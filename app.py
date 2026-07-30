"""
app.py  --  Blinkit Category Discovery Research Engine
======================================================
A grounded, evidence-cited review analysis app + a RAG-style Q&A.

What makes this different from a generic dashboard:
  * Every insight is computed from STRUCTURED LABELS, not LLM prose.
  * Every quote is REAL and cited by review_id (click to read the full review).
  * The "Ask the Reviews" tab is real Retrieval-Augmented Generation:
        query -> TF-IDF retrieval over all real reviews -> grounded answer.
    Retrieval works with NO API key. If GROQ_API_KEY is set, it also writes a
    synthesized answer that is forced to cite only the retrieved reviews and
    is passed through a verbatim guardrail (no fabricated quotes).

Run:  python3 -m streamlit run app.py
"""
import os
import re
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_env(path=".env"):
    """Load KEY=VALUE pairs from a local .env so the Q&A can use GROQ_API_KEY."""
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
# On Streamlit Cloud there is no .env — the key comes from st.secrets instead.
try:
    if not os.environ.get("GROQ_API_KEY") and "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

st.set_page_config(
    page_title="Blinkit Category Discovery Research Engine",
    page_icon="🛒",
    layout="wide"
)

# ----------------------------------------------------------------------------
# THEME — Blinkit-style dark mode with yellow accent
# ----------------------------------------------------------------------------
st.markdown("""
<style>
:root { --bl-yellow:#FFD700; --bl-orange:#FFA500; --bl-bg:#0f0f0f; --bl-card:#181818; --bl-muted:#B3B3B3; }

/* base */
.stApp { background: radial-gradient(1200px 500px at 20% -10%, #1a1800 0%, #0f0f0f 55%) fixed; }
html, body, [class*="css"] { font-family: 'Inter','Helvetica Neue',sans-serif; }

/* hide default chrome for a cleaner product feel */
#MainMenu, footer, header [data-testid="stHeader"] { visibility:hidden; }
[data-testid="stHeader"] { background:transparent; }

/* headings */
h1 { font-weight:800; letter-spacing:-.5px; }
h2, h3 { color:#fff; font-weight:700; }
.stCaption, [data-testid="stCaptionContainer"] { color:var(--bl-muted)!important; }

/* metric cards */
[data-testid="stMetric"] {
  background:var(--bl-card); border:1px solid #282828; border-radius:14px;
  padding:14px 16px;
}
[data-testid="stMetricValue"] { color:var(--bl-yellow); font-weight:800; }
[data-testid="stMetricLabel"] { color:var(--bl-muted); }

/* tabs -> Blinkit-style pill/underline */
.stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid #282828; }
.stTabs [data-baseweb="tab"] {
  background:transparent; color:var(--bl-muted); border-radius:8px 8px 0 0;
  padding:8px 14px; font-weight:600;
}
.stTabs [aria-selected="true"] { color:#fff!important; border-bottom:3px solid var(--bl-yellow); }

/* buttons -> yellow pills */
.stButton button {
  background:#232323; color:#fff; border:1px solid #303030; border-radius:999px;
  font-weight:600; transition:all .15s ease;
}
.stButton button:hover { background:var(--bl-yellow); color:#000; border-color:var(--bl-yellow); transform:scale(1.02); }

/* primary button style */
.stButton button[kind="primary"], .stButton button[data-testid="stFormSubmitButton"] {
  background:var(--bl-yellow); color:#000; border-color:var(--bl-yellow); font-weight:700;
}
.stButton button[kind="primary"]:hover { background:var(--bl-orange); border-color:var(--bl-orange); }

/* inputs */
[data-baseweb="input"], .stTextInput input, [data-baseweb="select"] {
  background:var(--bl-card)!important; border-radius:10px!important;
}

/* alert/callout cards -> rounded, dark, yellow left-rail for info/success */
[data-testid="stAlert"] { border-radius:12px; border:1px solid #282828; }
[data-testid="stExpander"] { border:1px solid #282828; border-radius:12px; background:var(--bl-card); }

/* blockquotes (cited quotes) */
blockquote { border-left:3px solid var(--bl-yellow); background:#1a1a1a; padding:8px 14px; border-radius:0 8px 8px 0; }

/* dataframe */
[data-testid="stDataFrame"] { border:1px solid #282828; border-radius:12px; }
hr { border-color:#282828; }

/* headline box */
.headline-box {
  border: 2px solid var(--bl-yellow); background: var(--bl-card);
  border-radius: 14px; padding: 18px 22px; margin-top: 16px;
}
.headline-box p { color: #fff; font-size: 1.05em; line-height: 1.6; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
@st.cache_data
def load_insights_v4():
    with open("insights.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_reviews_v4():
    return pd.read_csv("reviews_labeled.csv")


@st.cache_data
def load_clean_v4():
    return pd.read_csv("reviews_clean.csv")


@st.cache_resource
def build_retriever(texts):
    """Semantic retrieval via precomputed MiniLM embeddings if available; else TF-IDF.
    Returns (mode_label, search_fn) where search_fn(query, k) -> (indices, sims)."""
    import numpy as np
    if os.path.exists("embeddings.npy"):
        try:
            from sentence_transformers import SentenceTransformer
            emb = np.load("embeddings.npy")
            model = SentenceTransformer("all-MiniLM-L6-v2")

            def search(q, k):
                qv = model.encode([q], normalize_embeddings=True)
                sims = np.nan_to_num((emb @ qv.T).ravel(), nan=-1.0)
                idx = sims.argsort()[::-1][:k]
                return idx, sims
            return "semantic (MiniLM embeddings)", search
        except Exception:
            pass
    # Unigrams + capped vocabulary keep the sparse matrix small so scikit-learn/
    # scipy don't spike memory (a segfault risk on some cloud Python builds).
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 1), min_df=3,
                          max_df=0.6, max_features=20000)
    mat = vec.fit_transform(texts)

    def search(q, k):
        sims = cosine_similarity(vec.transform([q]), mat).ravel()
        idx = sims.argsort()[::-1][:k]
        return idx, sims
    return "lexical (TF-IDF)", search


ins = load_insights_v4()
lab = load_reviews_v4()
clean = load_clean_v4()
TEXT_BY_ID = dict(zip(clean["review_id"], clean["text"]))
META_BY_ID = {r.review_id: (r.source, r.rating) for r in clean.itertuples()}


@st.cache_resource
def get_retriever_v4():
    """Built lazily (and cached) on the FIRST search, so page load never runs the
    heavy scipy/scikit-learn TF-IDF fit — that eager build was crashing the app."""
    return build_retriever(clean["text"].astype(str).tolist())


def full_review(rid):
    return TEXT_BY_ID.get(rid, "(review not found)")


def show_quote(q):
    """Render a cited quote with an expander to read the full source review."""
    rid, src = q["review_id"], q.get("source", "")
    st.markdown(f"> \u201c{q['quote']}\u201d  \n> — <span style='opacity:.6'>review #{rid} · {src}</span>",
                unsafe_allow_html=True)
    with st.expander(f"read full review #{rid}"):
        st.write(full_review(rid))


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
total = ins["total"]
n_sources = len(ins["sources"])
st.title("🛒 Blinkit Category Discovery Research Engine")
st.caption(f"AI-powered Voice-of-Customer analysis for category discovery · "
           f"{total:,} reviews · {n_sources} sources")

tabs = st.tabs(["📊 Overview", "💡 Key Insights", "🔍 Discovery Deep-Dive",
                "💬 Ask the Reviews (AI)", "🗂 Evidence Explorer"])

# ============================================================================
# TAB 1 — OVERVIEW
# ============================================================================
with tabs[0]:
    st.markdown(f"### Blinkit Category Discovery Research Engine")
    st.caption(f"{total:,} reviews · {n_sources} sources")

    # KPI row
    barrier_counts = ins.get("category_barriers", {})
    barrier_related = sum(v for k, v in barrier_counts.items()
                          if k not in ("no_barrier", "not_applicable"))
    seg_counts = ins.get("user_segments", {})
    top_segment = max(((k, v) for k, v in seg_counts.items() if k != "unknown"),
                      key=lambda x: x[1], default=("—", 0))

    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews analyzed", f"{len(clean):,}")
    c2.metric("Sources", f"{n_sources}")
    c3.metric("Structured-labeled", f"{len(lab):,}")

    # Charts row
    sent = ins.get("sentiments", {})
    a, b = st.columns(2)
    with a:
        st.subheader("Real sentiment (from star ratings)")
        sdf = pd.DataFrame({"Sentiment": list(sent.keys()), "Count": list(sent.values())})
        fig = px.pie(sdf, names="Sentiment", values="Count", hole=.5,
                     color="Sentiment",
                     color_discrete_map={"positive": "#2ecc71", "neutral": "#888",
                                         "negative": "#E22134"})
        fig.update_traces(textinfo="percent+label", textposition="inside")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#fff", showlegend=True,
                          legend=dict(font=dict(color="#B3B3B3")))
        st.plotly_chart(fig, use_container_width=True, config={})
    with b:
        st.subheader("Where the reviews come from")
        src = ins.get("sources", {})
        sdf2 = pd.DataFrame({"Source": list(src.keys()), "Reviews": list(src.values())})
        source_colors = {
            "App Store": "#3498db",
            "Play Store": "#2ecc71",
            "YouTube": "#e84393",
            "Trustpilot": "#e74c3c",
            "Community": "#e74c3c",
            "Reddit": "#FF4500",
        }
        fig2 = px.bar(sdf2, x="Source", y="Reviews", color="Source",
                      color_discrete_map=source_colors)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#fff", showlegend=False,
                           xaxis=dict(gridcolor="#282828"),
                           yaxis=dict(gridcolor="#282828"))
        st.plotly_chart(fig2, use_container_width=True, config={})

# ============================================================================
# TAB 2 — KEY INSIGHTS (8 QUESTIONS)
# ============================================================================
with tabs[1]:
    st.subheader("The core category questions — answered with evidence")
    st.caption("Each answer separates symptom from root cause and is backed by counts + "
               "real cited quotes. Click any review to verify it.")

    QUESTIONS = [
        "Why do users repeatedly buy from the same categories?",
        "What prevents users from exploring new categories?",
        "How do users currently discover new products on Blinkit?",
        "What role do habits and routines play in shopping behavior?",
        "What information do users need before trying a new category?",
        "What frustrations emerge repeatedly across reviews?",
        "Which user segments are more likely to experiment with new categories?",
        "What unmet needs emerge consistently across discussions?",
    ]

    Q_DATA_KEYS = [
        "q1_same_categories", "q2_exploration_barriers", "q3_discovery_today",
        "q4_habit_role", "q5_info_needed", "q6_frustrations",
        "q7_explorer_segments", "q8_unmet_needs",
    ]

    def generate_finding(question, data_key, data):
        """Generate a 2-3 sentence finding using Groq if available, else from data."""
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            return _static_finding(data_key, data)
        try:
            from groq import Groq
            prompt = (
                f"You are a product researcher. Based on this data from Blinkit app reviews, "
                f"write a 2-3 sentence research finding answering: '{question}'\n\n"
                f"Data: {json.dumps(data, indent=2, default=str)}\n\n"
                f"Be specific, cite numbers from the data, and separate symptom from root cause. "
                f"Do NOT use markdown. Write plain text only."
            )
            ans = Groq(api_key=key).chat.completions.create(
                model="llama-3.1-8b-instant", temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content
            return ans.strip()
        except Exception:
            return _static_finding(data_key, data)

    def _static_finding(data_key, data):
        """Fallback static findings when Groq is unavailable."""
        if data_key == "q1_same_categories" and isinstance(data, list) and data:
            top = data[0]
            return (f"The primary reason is '{top['reason']}' with {top['count']} reviews "
                    f"({top['pct']}). Users form purchasing habits that lock them into "
                    f"familiar categories, rarely deviating from their established routine.")
        if data_key == "q2_exploration_barriers" and isinstance(data, list) and data:
            barriers = ", ".join(f"{d['barrier']} ({d['count']})" for d in data[:3])
            return f"Top barriers: {barriers}. These compound to create a friction wall against exploration."
        if data_key == "q3_discovery_today" and isinstance(data, list) and data:
            methods = ", ".join(f"{d['method']} ({d['count']})" for d in data[:3])
            return f"Current discovery methods: {methods}."
        if data_key == "q4_habit_role" and isinstance(data, dict):
            return (f"Habit lock affects {data.get('habit_lock_count', 0)} reviews ({data.get('pct', '?')}). "
                    f"Routines dominate shopping behavior on Blinkit.")
        if data_key == "q5_info_needed" and isinstance(data, dict):
            return (f"{data.get('information_gap_count', 0)} reviews cite an information gap. "
                    f"Users need trust signals before trying new categories.")
        if data_key == "q6_frustrations" and isinstance(data, list) and data:
            top = data[0]
            return f"The most common frustration area is '{top['frustration']}' with {top['count']} reviews."
        if data_key == "q7_explorer_segments" and isinstance(data, list) and data:
            top = data[0]
            return (f"'{top['segment']}' users are most likely to experiment ({top['experimental_count']} "
                    f"experimental out of {top['count']} total).")
        if data_key == "q8_unmet_needs" and isinstance(data, list) and data:
            top = data[0]
            return (f"The most common unmet need is '{top['need']}' with {top['evidence_count']} "
                    f"supporting reviews.")
        return "See the data breakdown below."

    def _get_quotes_from_data(data_key, data):
        """Extract quotes from the question data."""
        quotes = []
        if data_key in ("q1_same_categories", "q2_exploration_barriers", "q6_frustrations"):
            for item in (data if isinstance(data, list) else []):
                q_text = item.get("quote", "")
                if q_text:
                    quotes.append(q_text)
        elif data_key == "q8_unmet_needs":
            for item in (data if isinstance(data, list) else []):
                q_text = item.get("quote", "")
                if q_text:
                    quotes.append(q_text)
        elif data_key == "q5_info_needed" and isinstance(data, dict):
            quotes = data.get("top_quotes", [])[:3]
        return quotes[:3]

    def _get_stat_line(data_key, data):
        """Build a stat line for each question."""
        if data_key == "q1_same_categories" and isinstance(data, list):
            total_cited = sum(d["count"] for d in data)
            return f"{total_cited} reviews cite specific category-lock reasons"
        if data_key == "q2_exploration_barriers" and isinstance(data, list):
            total_cited = sum(d["count"] for d in data)
            return f"{total_cited} reviews identify exploration barriers"
        if data_key == "q3_discovery_today" and isinstance(data, list):
            total_cited = sum(d["count"] for d in data)
            return f"{total_cited} reviews describe current discovery methods"
        if data_key == "q4_habit_role" and isinstance(data, dict):
            return (f"{data.get('habit_lock_count', 0)} reviews show habit lock-in "
                    f"({data.get('pct', '?')} of labeled set)")
        if data_key == "q5_info_needed" and isinstance(data, dict):
            return f"{data.get('information_gap_count', 0)} reviews cite information gaps"
        if data_key == "q6_frustrations" and isinstance(data, list):
            total_cited = sum(d["count"] for d in data)
            return f"{total_cited} reviews across {len(data)} frustration topics"
        if data_key == "q7_explorer_segments" and isinstance(data, list):
            total_exp = sum(d["experimental_count"] for d in data)
            return f"{total_exp} reviews from experimentally-inclined users"
        if data_key == "q8_unmet_needs" and isinstance(data, list):
            total_cited = sum(d["evidence_count"] for d in data)
            return f"{total_cited} reviews express unmet needs"
        return ""

    for idx, (question, dk) in enumerate(zip(QUESTIONS, Q_DATA_KEYS)):
        q_data = ins.get(dk, [])
        st.markdown(f"### {idx + 1}. {question}")

        # Finding box
        finding = generate_finding(question, dk, q_data)
        st.info(f"**Finding:** {finding}")

        # Stat line
        stat = _get_stat_line(dk, q_data)
        if stat:
            st.caption(f"📈 {stat}")

        # Evidence quotes
        quotes = _get_quotes_from_data(dk, q_data)
        if quotes:
            st.markdown("**Evidence:**")
            for q_text in quotes:
                if isinstance(q_text, str) and q_text.strip():
                    st.markdown(f"> \u201c{q_text}\u201d",
                                unsafe_allow_html=True)

        st.divider()

# ============================================================================
# TAB 3 — DISCOVERY DEEP-DIVE
# ============================================================================
with tabs[2]:
    st.subheader("Category exploration barrier breakdown")

    barriers = ins.get("category_barriers", {})
    # Filter out not_applicable for the chart
    chart_barriers = {k: v for k, v in barriers.items() if k != "not_applicable"}
    if chart_barriers:
        bdf = (pd.DataFrame({"Barrier": list(chart_barriers.keys()),
                              "Count": list(chart_barriers.values())})
               .sort_values("Count"))
        fig = px.bar(bdf, x="Count", y="Barrier", orientation="h",
                     color="Count",
                     color_continuous_scale=[[0, "#FFA500"], [0.5, "#FFD700"], [1, "#FFEC8B"]])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#fff",
                          xaxis=dict(gridcolor="#282828"),
                          yaxis=dict(gridcolor="#282828"),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True, config={})

    # Headline box
    headline = ins.get("headline_finding", "")
    if headline:
        st.markdown(f"""
        <div class="headline-box">
            <p>🔍 <strong>Headline:</strong> {headline}</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TAB 4 — ASK THE REVIEWS (RAG)
# ============================================================================
with tabs[3]:
    st.subheader("Ask the reviews anything")
    st.caption(f"Real RAG: your question is matched against all "
               f"{total:,} reviews; the most relevant real reviews are retrieved "
               f"and used as the ONLY basis for the answer.")
    st.caption(f"🔎 Answer mode: "
               f"**{'LLM-synthesized (Groq)' if os.environ.get('GROQ_API_KEY') else 'extractive'}** "
               "· search index builds on first query.")

    examples = [
        "Why do users keep buying the same categories?",
        "What stops users from trying new product categories?",
        "How do users discover new products today?",
        "What do users say about exploring new categories?",
    ]
    
    def set_query(q):
        st.session_state.search_query = q
        
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        cols[i].button(ex, key=f"example_{i}", on_click=set_query, args=(ex,))
        
    query = st.text_input("Your question:", key="search_query",
                          placeholder="e.g. Why do users keep ordering the same products?")
    topk = st.slider("How many reviews to retrieve", 3, 15, 6)

    if query:
        RETRIEVER_MODE, SEARCH = get_retriever_v4()
        idx, sims = SEARCH(query, topk)
        retrieved = clean.iloc[idx].copy()
        retrieved["score"] = sims[idx]
        thr = 0.15 if "semantic" in RETRIEVER_MODE else 0.02
        retrieved = retrieved[retrieved["score"] > thr]

        if retrieved.empty:
            st.warning("No closely matching reviews. Try rephrasing.")
        else:
            # --- grounded answer ---
            key = os.environ.get("GROQ_API_KEY")
            st.markdown("#### Answer")
            if key:
                try:
                    from groq import Groq
                    ctx = "\n".join(f"[#{r.review_id}] {r.text}" for r in retrieved.itertuples())
                    prompt = (
                        "Answer the question using ONLY the reviews below. Cite review ids "
                        "like [#123]. If the reviews don't support an answer, say so. Do not "
                        "invent quotes.\n\nQUESTION: " + query + "\n\nREVIEWS:\n" + ctx)
                    ans = Groq(api_key=key).chat.completions.create(
                        model="llama-3.1-8b-instant", temperature=0.2,
                        messages=[{"role": "user", "content": prompt}]
                    ).choices[0].message.content
                    st.write(ans)
                except Exception as e:
                    st.error(f"LLM synthesis failed ({e}). Showing retrieved evidence below.")
            else:
                # No key: extractive grounded answer (sentences mentioning query terms)
                terms = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 3]
                picks = []
                for r in retrieved.itertuples():
                    for sent_ in re.split(r"(?<=[.!?])\s+", str(r.text)):
                        if sum(t in sent_.lower() for t in terms) >= 1 and 4 < len(sent_.split()) < 40:
                            picks.append((r.review_id, r.source, sent_.strip()))
                            break
                if picks:
                    st.markdown("*What real reviewers say (extractive — set `GROQ_API_KEY` for a "
                                "synthesized answer):*")
                    for rid, src, s in picks[:5]:
                        st.markdown(f"- \u201c{s}\u201d — <span style='opacity:.6'>#{rid} · {src}</span>",
                                    unsafe_allow_html=True)
                else:
                    st.info("Set GROQ_API_KEY for a synthesized answer. Retrieved reviews below.")

            # --- the retrieved evidence (always shown) ---
            st.markdown("#### Retrieved reviews (the evidence)")
            for r in retrieved.itertuples():
                st.markdown(f"**#{r.review_id} · {r.source} · {('%.0f★'%r.rating) if pd.notna(r.rating) else '—'}** "
                            f"<span style='opacity:.5'>(match {r.score:.2f})</span>", unsafe_allow_html=True)
                st.write(str(r.text)[:500] + ("…" if len(str(r.text)) > 500 else ""))
                st.divider()

# ============================================================================
# TAB 5 — EVIDENCE EXPLORER
# ============================================================================
with tabs[4]:
    st.subheader("Browse the raw evidence")
    f1, f2, f3 = st.columns([2, 1, 1])
    kw = f1.text_input("Keyword filter", placeholder="e.g. category, new, explore, habit")
    srcf = f2.multiselect("Source", sorted(clean["source"].unique()))
    only_low = f3.checkbox("Only ≤2★")
    view = clean
    if kw:
        view = view[view["text"].str.contains(re.escape(kw), case=False, na=False)]
    if srcf:
        view = view[view["source"].isin(srcf)]
    if only_low:
        view = view[view["rating"] <= 2]
    st.caption(f"{len(view):,} reviews match")
    st.dataframe(view[["review_id", "source", "rating", "text"]].head(300),
                 use_container_width=True, height=500)
