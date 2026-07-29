"""
collect_community.py
--------------------
Collect Blinkit reviews from Trustpilot. Since Blinkit does not have a
dedicated community forum like Spotify, we scrape Trustpilot as the
community-style review source.

Strategy:
  1. Fetch multiple pages of Trustpilot reviews for blinkit.com.
  2. Extract the review text and star rating from each review card.
  3. Filter for substance, dedupe.

Writes incrementally. Output: community_reviews.csv  [text, rating, source, url]
"""
import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
HEADERS = {"User-Agent": UA}
BASE_URL = "https://www.trustpilot.com/review/blinkit.com"
OUT = "community_reviews.csv"

MAX_PAGES = 10
MIN_WORDS = 5


def get(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        return r.text if r.status_code == 200 else ""
    except Exception:
        return ""


if os.path.exists(OUT):
    os.remove(OUT)

seen_text = set()
total = 0

for page in range(1, MAX_PAGES + 1):
    url = f"{BASE_URL}?page={page}" if page > 1 else BASE_URL
    html = get(url)
    if not html:
        print(f"  ! Page {page} returned empty — stopping.")
        break

    soup = BeautifulSoup(html, "lxml")
    rows = []

    # Trustpilot review cards — each card has a star rating image and review text
    review_cards = soup.select("article[data-service-review-card-paper]")
    if not review_cards:
        # Fallback: try alternative selectors
        review_cards = soup.select(".review-card") or soup.select("[data-review-id]")

    for card in review_cards:
        # Extract rating from the star image alt text (e.g., "Rated 5 out of 5 stars")
        rating = None
        star_el = card.select_one("img[alt*='Rated']") or card.select_one("[data-service-review-rating]")
        if star_el:
            alt = star_el.get("alt", "") or ""
            match = re.search(r"Rated (\d)", alt)
            if match:
                rating = int(match.group(1))
            else:
                # try data attribute
                rating_attr = star_el.get("data-service-review-rating")
                if rating_attr:
                    rating = int(rating_attr)

        # Extract review text
        text_el = (card.select_one("[data-service-review-text-typography]")
                   or card.select_one(".review-content__text")
                   or card.select_one("p"))
        if not text_el:
            continue
        text = text_el.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text.split()) < MIN_WORDS:
            continue

        key = text[:120].lower()
        if key in seen_text:
            continue
        seen_text.add(key)

        rows.append({
            "text": text,
            "rating": rating,
            "source": "Trustpilot",
            "url": url,
        })

    if rows:
        pd.DataFrame(rows).to_csv(OUT, mode="a", index=False,
                                  header=not os.path.exists(OUT))
        total += len(rows)

    print(f"Page {page}: +{len(rows)} reviews (total {total})")
    time.sleep(0.5)

    if len(rows) == 0:
        print("No more reviews found — stopping pagination.")
        break

print(f"\nSaved {OUT} | {total} Blinkit Trustpilot reviews")
