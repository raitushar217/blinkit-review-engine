"""
collect_reddit.py  —  Reddit scraper for Blinkit discussions
-------------------------------------------------------------
Uses Reddit's public JSON endpoints (no API key needed).
Falls back to PRAW if credentials are available.

Output: reddit_reviews.csv  with columns [text, rating, date, source]
"""
import os
import time
import requests
import pandas as pd


def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

SUBREDDITS = ["india", "bangalore", "delhi", "mumbai", "hyderabad"]
QUERIES = [
    "blinkit new categories",
    "blinkit same products every order",
    "quick commerce category discovery",
    "blinkit grocery habit india",
    "blinkit explore categories",
    "blinkit review",
    "blinkit experience",
    "blinkit vs zepto",
    "blinkit delivery",
    "blinkit app",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120 Safari/537.36"
}


def search_reddit_json(subreddit, query, limit=25):
    """Search Reddit using the public .json endpoint (no auth needed)."""
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {
        "q": query,
        "restrict_sr": "on",
        "sort": "relevance",
        "t": "all",
        "limit": limit,
    }
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if r.status_code == 429:
            print(f"  Rate limited on r/{subreddit}, waiting 10s...")
            time.sleep(10)
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if r.status_code != 200:
            print(f"  ! r/{subreddit} search returned {r.status_code}")
            return []
        data = r.json()
        posts = data.get("data", {}).get("children", [])
        return [p["data"] for p in posts if p.get("data")]
    except Exception as e:
        print(f"  ! r/{subreddit} search failed: {e}")
        return []


def get_comments_json(permalink, limit=5):
    """Fetch top comments from a post using the public .json endpoint."""
    url = f"https://www.reddit.com{permalink}.json"
    try:
        r = requests.get(url, headers=HEADERS, params={"limit": limit}, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        if len(data) < 2:
            return []
        comments = data[1].get("data", {}).get("children", [])
        return [c["data"].get("body", "") for c in comments
                if c.get("kind") == "t1" and c["data"].get("body")]
    except Exception:
        return []


def collect_with_json():
    """Collect Reddit posts and comments using public JSON API."""
    rows, seen = [], set()
    for sub in SUBREDDITS:
        for q in QUERIES:
            posts = search_reddit_json(sub, q, limit=25)
            for post in posts:
                pid = post.get("id", "")
                if pid in seen:
                    continue
                seen.add(pid)
                title = post.get("title", "")
                selftext = post.get("selftext", "")
                body = f"{title}. {selftext}".strip(". ").strip()
                if body:
                    rows.append({
                        "text": body,
                        "rating": None,
                        "date": post.get("created_utc", ""),
                        "source": "Reddit",
                    })
                # Fetch a few top comments
                permalink = post.get("permalink", "")
                if permalink:
                    comment_texts = get_comments_json(permalink, limit=5)
                    for ct in comment_texts:
                        if ct.strip() and ct not in seen:
                            seen.add(ct[:100])
                            rows.append({
                                "text": ct.strip(),
                                "rating": None,
                                "date": post.get("created_utc", ""),
                                "source": "Reddit",
                            })
            print(f"  r/{sub} q='{q}': {len(rows)} total rows so far")
            time.sleep(1.5)  # respect rate limits
    return rows


def collect_with_praw():
    """Collect Reddit posts and comments using PRAW (needs credentials)."""
    import praw
    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "blinkit-research"),
    )
    rows, seen = [], set()
    for sub in SUBREDDITS:
        for q in QUERIES:
            try:
                for post in reddit.subreddit(sub).search(q, limit=40, sort="relevance"):
                    if post.id in seen:
                        continue
                    seen.add(post.id)
                    body = (post.title or "") + ". " + (post.selftext or "")
                    rows.append({"text": body.strip(), "rating": None,
                                 "date": post.created_utc, "source": "Reddit"})
                    post.comments.replace_more(limit=0)
                    for c in post.comments[:5]:
                        rows.append({"text": c.body, "rating": None,
                                     "date": c.created_utc, "source": "Reddit"})
            except Exception as e:
                print(f"  ! r/{sub} q='{q}' failed: {e}")
        print(f"  r/{sub}: {len(rows)} total rows so far")
    return rows


# --- Main ---
print("Collecting Blinkit discussions from Reddit...")

if os.environ.get("REDDIT_CLIENT_ID") and os.environ.get("REDDIT_CLIENT_SECRET"):
    print("Using PRAW (authenticated API)")
    try:
        rows = collect_with_praw()
    except Exception as e:
        print(f"PRAW failed ({e}), falling back to public JSON API")
        rows = collect_with_json()
else:
    print("No Reddit API credentials — using public JSON API (no key needed)")
    rows = collect_with_json()

if rows:
    df = pd.DataFrame(rows)
    df = df[df["text"].str.split().str.len() >= 5].drop_duplicates("text")
    df.to_csv("reddit_reviews.csv", index=False)
    print(f"\nSaved reddit_reviews.csv | {len(df)} Reddit items about Blinkit")
else:
    print("\nNo Reddit data collected. Creating empty reddit_reviews.csv.")
    pd.DataFrame(columns=["text", "rating", "date", "source"]).to_csv(
        "reddit_reviews.csv", index=False)
