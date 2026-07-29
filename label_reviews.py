"""
label_reviews.py  --  THE INTELLIGENCE CORE (at-scale)
-------------------------------------------------------
Labels each Blinkit review into structured JSON so insights can be COUNTED
and CITED, not hallucinated.

To keep the run cheap/fast we don't label all reviews. We label:
  * EVERY category-discovery-relevant review  (high value -> deep analysis)
  * PLUS a random CONTROL sample              (unbiased -> lets us estimate
                                                true corpus prevalence & topic mix)
Each row is tagged with `stratum` = relevant | control so analyze.py can report
an unbiased prevalence from the control stratum while using ALL discovery
reviews for evidence depth.

Guardrail: verbatim_quote must be a real substring of the review or it is
dropped -> fabricated quotes are impossible.

Setup (no keys in code or chat):
  1. Create a .env file (see .env.example):
        GROQ_API_KEY=gsk_your_key
        CONTROL_SAMPLE=500          # optional, default 500
  2. python label_reviews.py
Resumable: re-run after any interruption; it skips already-labeled reviews.
"""
import os
import re
import json
import time
import pandas as pd
from groq import Groq

MODEL = "llama-3.1-8b-instant"   # 70b hit its 100k tokens/day cap; 8b has a separate, larger budget
BATCH = 12
IN_FILE = "reviews_clean.csv"
OUT_FILE = "reviews_labeled.csv"

# High-precision pre-filter: "is this about category discovery, repeat purchases,
# or exploration behavior on Blinkit?"
DISCOVERY_RE = re.compile(
    r"(categor|explore|discover|same (product|item|order|thing)|repeat order|"
    r"habitual|routine|stuck|new (brand|product|item|categor)|recommend|suggest|"
    r"try (new|something|different)|browse|hidden gem|didn.?t know|never tried|"
    r"variety|personaliz|for you|similar item|also (buy|bought|order)|"
    r"impulse|experiment|boring|same every|nothing new)", re.I)

TOPICS = ["category_exploration", "repeat_purchase", "delivery_speed", "app_ux",
          "pricing", "product_quality", "customer_support", "other"]
CATEGORY_BARRIERS = ["habit_lock", "no_trigger", "trust_deficit", "price_anxiety",
                     "information_gap", "overwhelmed", "no_barrier", "not_applicable"]
DISCOVERY_THEMES = ["stuck_in_routine", "wants_to_explore", "price_blocks_exploration",
                    "needs_recommendation", "social_influence", "positive_discovery",
                    "needs_trust_signal", "not_applicable"]
USER_SEGMENTS = ["household_manager", "young_professional", "bachelor",
                 "health_conscious", "budget_shopper", "convenience_seeker",
                 "new_user", "unknown"]
SHOPPING_HABITS = ["planned_weekly", "impulse", "recurring_auto", "experimental",
                   "not_mentioned"]
SENTIMENT = ["positive", "negative", "neutral"]


def load_env(path=".env"):
    """Load KEY=VALUE lines from a local .env into the environment (no dependency)."""
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


SYSTEM = (
    "You are a product researcher analyzing Blinkit (quick commerce) "
    "app reviews. Label each review into structured JSON. "
    "Return ONLY valid JSON, no markdown."
)

PROMPT_TMPL = """Label each review below. Return a JSON array with one object per review:
{{
  "review_id": <int, copy exactly>,
  "topic": <one of {topics}>,
  "category_barrier": <one of {category_barriers}; habit_lock=stuck ordering same things; no_trigger=no prompt to try new; trust_deficit=doesn't trust new products; price_anxiety=afraid new items cost more; information_gap=can't find info about new products; overwhelmed=too many choices; no_barrier=no barrier present; not_applicable=not about category exploration>,
  "discovery_theme": <one of {discovery_themes}; stuck_in_routine=keeps ordering same; wants_to_explore=wants to try new; price_blocks_exploration=price stops them; needs_recommendation=wants suggestions; social_influence=influenced by others; positive_discovery=found something new; needs_trust_signal=needs reviews/ratings; not_applicable=not about discovery>,
  "user_segment": <one of {user_segments}>,
  "shopping_habit": <one of {shopping_habits}; planned_weekly=weekly grocery run; impulse=spontaneous orders; recurring_auto=subscription/auto-reorder; experimental=tries new things; not_mentioned=not clear>,
  "sentiment": <one of {sentiment}>,
  "verbatim_quote": <the EXACT span copied char-for-char (<=180 chars) from the review text that best supports your labeling, or null>
}}
Rules: copy verbatim_quote character-for-character (do NOT paraphrase or fix typos). If unsure use not_applicable/unknown/null. Output ONLY the JSON array.

REVIEWS:
{block}"""


SINGLE_PROMPT_TMPL = """Label this Blinkit app review:

Rating: {rating}/5
Review: {text}

Return JSON with EXACTLY this structure:
{{
  "topic": "category_exploration|repeat_purchase|delivery_speed|app_ux|pricing|product_quality|customer_support|other",
  "category_barrier": "habit_lock|no_trigger|trust_deficit|price_anxiety|information_gap|overwhelmed|no_barrier|not_applicable",
  "discovery_theme": "stuck_in_routine|wants_to_explore|price_blocks_exploration|needs_recommendation|social_influence|positive_discovery|needs_trust_signal|not_applicable",
  "user_segment": "household_manager|young_professional|bachelor|health_conscious|budget_shopper|convenience_seeker|new_user|unknown",
  "shopping_habit": "planned_weekly|impulse|recurring_auto|experimental|not_mentioned",
  "sentiment": "positive|negative|neutral",
  "verbatim_quote": "a direct word-for-word quote from the review text that best supports your labeling, or empty string"
}}"""


def build_prompt(batch_df):
    block = "\n".join(f'[review_id={r.review_id}] {r.text}' for r in batch_df.itertuples())
    return PROMPT_TMPL.format(topics=TOPICS, category_barriers=CATEGORY_BARRIERS,
                              discovery_themes=DISCOVERY_THEMES, user_segments=USER_SEGMENTS,
                              shopping_habits=SHOPPING_HABITS, sentiment=SENTIMENT,
                              block=block)


def parse_json_array(content):
    content = content.replace("```json", "").replace("```", "").strip()
    s, e = content.find("["), content.rfind("]")
    if s == -1 or e == -1:
        return []
    try:
        return json.loads(content[s:e + 1])
    except json.JSONDecodeError:
        return []


def norm(s):
    s = str(s).lower()
    for a, b in [("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"')]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def coerce(obj, text_by_id, stratum_by_id):
    rid = obj.get("review_id")
    if rid not in text_by_id:
        return None, False
    src = norm(text_by_id[rid])
    ev = obj.get("verbatim_quote")
    fabricated = False
    if ev and isinstance(ev, str):
        if norm(ev) not in src:   # GUARDRAIL
            ev, fabricated = None, True
    pick = lambda v, allowed, d: v if v in allowed else d
    row = {
        "review_id": rid,
        "topic": pick(obj.get("topic"), TOPICS, "other"),
        "category_barrier": pick(obj.get("category_barrier"), CATEGORY_BARRIERS, "not_applicable"),
        "discovery_theme": pick(obj.get("discovery_theme"), DISCOVERY_THEMES, "not_applicable"),
        "user_segment": pick(obj.get("user_segment"), USER_SEGMENTS, "unknown"),
        "shopping_habit": pick(obj.get("shopping_habit"), SHOPPING_HABITS, "not_mentioned"),
        "sentiment": pick(obj.get("sentiment"), SENTIMENT, "neutral"),
        "verbatim_quote": ev,
        "stratum": stratum_by_id.get(rid, "relevant"),
    }
    return row, fabricated


def build_target(df):
    """Relevant reviews + a random control sample, each tagged with a stratum."""
    control_n = int(os.environ.get("CONTROL_SAMPLE", "500"))
    mask = df["text"].astype(str).str.contains(DISCOVERY_RE)
    relevant = df[mask].copy()
    relevant["stratum"] = "relevant"
    rest = df[~mask]
    control = rest.sample(min(control_n, len(rest)), random_state=42).copy()
    control["stratum"] = "control"
    target = pd.concat([relevant, control]).drop_duplicates("review_id").reset_index(drop=True)
    print(f"Target set: {len(relevant)} relevant + {len(control)} control = {len(target)} reviews")
    return target


def main():
    load_env()
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not found. Add it to a .env file (see .env.example).")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    df = pd.read_csv(IN_FILE)
    target = build_target(df)
    stratum_by_id = dict(zip(target["review_id"], target["stratum"]))

    done = set()
    if os.path.exists(OUT_FILE):
        prev = pd.read_csv(OUT_FILE)
        if "stratum" in prev.columns:           # only resume a real at-scale run
            done = set(prev["review_id"].tolist())
            print(f"Resuming: {len(done)} already labeled.")
    todo = target[~target["review_id"].isin(done)].reset_index(drop=True)
    n_batches = (len(todo) + BATCH - 1) // BATCH
    print(f"To label: {len(todo)} reviews in {n_batches} batches (~{n_batches*0.7:.0f}s + API time)")

    fabricated = 0
    for i in range(0, len(todo), BATCH):
        batch = todo.iloc[i:i + BATCH]
        text_by_id = dict(zip(batch["review_id"], batch["text"].astype(str)))
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=0.1,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": build_prompt(batch)}])
            rows = []
            for obj in parse_json_array(resp.choices[0].message.content):
                row, fab = coerce(obj, text_by_id, stratum_by_id)
                fabricated += int(fab)
                if row:
                    rows.append(row)
            if rows:
                pd.DataFrame(rows).to_csv(OUT_FILE, mode="a", index=False,
                                          header=not os.path.exists(OUT_FILE))
            print(f"  batch {i//BATCH + 1}/{n_batches}: +{len(rows)} (fabricated blocked: {fabricated})")
            time.sleep(0.4)
        except Exception as ex:
            print(f"  ! batch {i//BATCH + 1} failed: {ex} -- rerun to resume")
            time.sleep(3)

    print(f"\nDone. Fabricated quotes blocked by guardrail: {fabricated}")
    print(f"Now run:  python analyze.py")


if __name__ == "__main__":
    main()
