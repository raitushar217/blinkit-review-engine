"""
build_dataset.py
----------------
Merge ALL sources into ONE clean schema for Blinkit review analysis.

Reads from:
  - appstore_reviews.csv (App Store iOS reviews)
  - data/raw_playstore.csv (Play Store Android reviews)
  - youtube_reviews.csv / youtube_comments.csv (YouTube comments)
  - community_reviews.csv (Trustpilot reviews)
  - reddit_reviews.csv (Reddit discussions)
  - Any other raw_*.csv files in the data/ folder

Output: reviews_clean.csv  with columns:
    review_id | source | text | rating | date
Every row has non-empty text. review_id is stable and used for citations.
"""
import os
import glob
import pandas as pd


def longest_text_col(df):
    """Pick the column that looks most like free-text (longest avg length)."""
    best, best_len = None, 0
    for c in df.columns:
        if df[c].dtype == object:
            avg = df[c].dropna().astype(str).str.len().mean() or 0
            if avg > best_len:
                best, best_len = c, avg
    return best


def normalize_frame(df, source_name=None):
    """Normalize a DataFrame to the standard schema: text, rating, date, source."""
    # Determine text column
    text_col = None
    for candidate in ["text", "Review", "content", "Comment", "User comment"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        text_col = longest_text_col(df)
    if text_col is None:
        return None

    # Determine rating column
    rating_col = None
    for candidate in ["rating", "Rating", "score", "stars"]:
        if candidate in df.columns:
            rating_col = candidate
            break

    # Determine date column
    date_col = None
    for candidate in ["date", "Date", "at", "created_utc", "timestamp"]:
        if candidate in df.columns:
            date_col = candidate
            break

    # Determine source column
    source_val = source_name
    if "source" in df.columns:
        source_series = df["source"]
    elif "Source" in df.columns:
        source_series = df["Source"]
    else:
        source_series = pd.Series([source_val] * len(df))

    result = pd.DataFrame({
        "text": df[text_col].astype(str),
        "rating": df[rating_col] if rating_col else None,
        "date": df[date_col].astype(str) if date_col else None,
        "source": source_series,
    })
    return result


frames = []

# 1) App Store (iOS reviews) ------------------------------------------------
if os.path.exists("appstore_reviews.csv"):
    a = pd.read_csv("appstore_reviews.csv")
    f = normalize_frame(a, "App Store")
    if f is not None:
        frames.append(f)

# 2) Play Store (from data/ folder) -----------------------------------------
if os.path.exists("data/raw_playstore.csv"):
    p = pd.read_csv("data/raw_playstore.csv")
    f = normalize_frame(p, "Play Store")
    if f is not None:
        frames.append(f)

# Also check legacy Play Store files
for pf in ["spotify_reviews_filtered.csv"]:
    if os.path.exists(pf):
        p = pd.read_csv(pf)
        f = normalize_frame(p, "Play Store")
        if f is not None:
            frames.append(f)

# 3) YouTube ----------------------------------------------------------------
for yf in ["youtube_reviews.csv", "youtube_comments.csv"]:
    if os.path.exists(yf):
        y = pd.read_csv(yf)
        f = normalize_frame(y, "YouTube")
        if f is not None:
            frames.append(f)

# 4) Community / Trustpilot -------------------------------------------------
if os.path.exists("community_reviews.csv"):
    cr = pd.read_csv("community_reviews.csv")
    f = normalize_frame(cr, "Trustpilot")
    if f is not None:
        frames.append(f)

# 5) Reddit ------------------------------------------------------------------
if os.path.exists("reddit_reviews.csv") and os.path.getsize("reddit_reviews.csv") > 50:
    r = pd.read_csv("reddit_reviews.csv")
    f = normalize_frame(r, "Reddit")
    if f is not None:
        frames.append(f)

# 6) Any other raw CSVs in data/ folder --------------------------------------
already_loaded = {"raw_playstore.csv"}
for csv_path in glob.glob("data/raw_*.csv"):
    basename = os.path.basename(csv_path)
    if basename in already_loaded:
        continue
    already_loaded.add(basename)
    try:
        extra = pd.read_csv(csv_path)
        source_name = basename.replace("raw_", "").replace(".csv", "").title()
        f = normalize_frame(extra, source_name)
        if f is not None:
            frames.append(f)
    except Exception as e:
        print(f"Warning: could not read {csv_path}: {e}")

if not frames:
    raise SystemExit("No source CSV files found. Run the collection scripts first.")

df = pd.concat(frames, ignore_index=True)

# Clean: strip, drop empties / very short / dedupe
df["text"] = df["text"].str.strip()
df = df[df["text"].str.split().str.len() >= 5]
df = df[~df["text"].str.lower().isin(["nan", "none", ""])]
df["clean_text"] = df["text"].str.replace(r' \[\d+\]$', '', regex=True).str.lower()
df = df.drop_duplicates(subset="clean_text").drop(columns=["clean_text"]).reset_index(drop=True)
df.insert(0, "review_id", range(1, len(df) + 1))

# Ensure final column order
df = df[["review_id", "source", "text", "rating", "date"]]

df.to_csv("reviews_clean.csv", index=False)

print(f"reviews_clean.csv written | {len(df)} unique Blinkit reviews")
print("\nSource breakdown:")
print(df["source"].value_counts())
print("\nSanity check -- every row has text:", df["text"].notna().all(),
      "| empty texts:", (df["text"].str.len() == 0).sum())
