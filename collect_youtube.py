"""
collect_youtube.py
------------------
Pull comments from YouTube videos ABOUT Blinkit's quick commerce experience /
category discovery / grocery delivery. Comments on these videos are rich in
real user opinions about the app and service quality.

Writes incrementally (appends after each video) so a slow/interrupted run still
keeps progress. Output: youtube_reviews.csv  [text, rating, source, video_id]
"""
import os
import time
import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader

# Blinkit / quick-commerce review & comparison videos
VIDEO_IDS = [
    'dQw4w9WgXcQ',  # placeholder — replace with actual Blinkit review videos on YouTube
]

SEARCH_QUERIES = [
    'blinkit app review 2024',
    'blinkit vs zepto vs swiggy instamart',
    'blinkit grocery delivery experience india',
    'quick commerce india review',
]

PER_VIDEO = 180
MIN_WORDS = 8          # drop "love it 😍" style low-signal comments
OUT = "youtube_reviews.csv"

downloader = YoutubeCommentDownloader()
seen = set()
if os.path.exists(OUT):
    os.remove(OUT)

total = 0
for vid in VIDEO_IDS:
    rows = []
    try:
        n = 0
        for c in downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={vid}"):
            t = (c.get("text") or "").strip().replace("\n", " ")
            if len(t.split()) < MIN_WORDS or t in seen:
                continue
            seen.add(t)
            rows.append({"text": t, "rating": None, "source": "YouTube", "video_id": vid})
            n += 1
            if n >= PER_VIDEO:
                break
        if rows:
            pd.DataFrame(rows).to_csv(OUT, mode="a", index=False,
                                      header=not os.path.exists(OUT))
        total += len(rows)
        print(f"{vid}  +{len(rows)}  (total {total})")
        time.sleep(1)
    except Exception as e:
        print(f"! {vid} failed: {e}")

print(f"\nSaved {OUT} | {total} substantive comments from {len(VIDEO_IDS)} Blinkit-related videos")
