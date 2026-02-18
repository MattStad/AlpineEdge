# src/news_rss_fetcher.py

from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime
import feedparser

from src.config import FEEDS


def fetch_rss_feeds() -> List[Dict[str, Any]]:
    articles: List[Dict[str, Any]] = []

    for category, urls in FEEDS.items():
        for url in urls:
            try:
                print(f"[RSS] Fetching {category}: {url}")
                feed = feedparser.parse(url)
            except Exception as e:
                print(f"[RSS] Fehler beim Holen von {url}: {e}")
                continue

            for entry in feed.entries:
                headline = entry.get("title")
                link = entry.get("link")
                summary = entry.get("summary") or entry.get("description")
                published = (
                    entry.get("published")
                    or entry.get("updated")
                    or entry.get("pubDate")
                )

                # published zu ISO-String normalisieren (best effort)
                published_iso = None
                if published:
                    try:
                        # feedparser kann parsed date liefern
                        if hasattr(entry, "published_parsed") and entry.published_parsed:
                            dt = datetime(*entry.published_parsed[:6])
                            published_iso = dt.isoformat()
                        else:
                            # fallback: einfach Rohstring speichern
                            published_iso = str(published)
                    except Exception:
                        published_iso = str(published)

                article = {
                    "source": "rss",
                    "category": category,
                    "ticker": None,  # RSS sind eher Makro / allgemein
                    "headline": headline,
                    "url": link,
                    "published_at": published_iso,
                    "summary": summary,
                    "raw_text": None,
                    "language": "de" if "AT_" in category else None,
                }
                articles.append(article)

    return articles

if __name__ == "__main__":
    arts = fetch_rss_feeds()
    print(f"Fetched {len(arts)} RSS-Artikel.")
    print(arts[:3])
