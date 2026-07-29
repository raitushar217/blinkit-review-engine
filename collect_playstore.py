"""
collect_playstore.py
--------------------
Collect Blinkit Play Store (Android) reviews using google-play-scraper.

Install dependency:
  pip install google-play-scraper

Output: data/raw_playstore.csv  with columns [review_id, source, text, rating, date, thumbs_up]
"""

from google_play_scraper import reviews, Sort
import csv, os, time


def collect(count=500):
    result, _ = reviews(
        'com.grofers.customerapp',
        lang='en',
        country='in',
        sort=Sort.NEWEST,
        count=count,
    )
    rows = []
    for r in result:
        if not r.get('content'):
            continue
        rows.append({
            'review_id': r['reviewId'],
            'source': 'Play Store',
            'text': r['content'],
            'rating': r['score'],
            'date': r['at'].isoformat(),
            'thumbs_up': r.get('thumbsUpCount', 0),
        })
    return rows


if __name__ == '__main__':
    rows = collect(500)
    os.makedirs('data', exist_ok=True)
    with open('data/raw_playstore.csv', 'w', newline='',
              encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Collected {len(rows)} Play Store reviews")
