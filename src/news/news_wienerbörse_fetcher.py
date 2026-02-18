# src/news_wienerboerse_fetcher.py

from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime
import re

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.wienerborse.at"
NEWS_URL = f"{BASE_URL}/news/"


def _parse_meta_for_anchor(anchor) -> tuple[str | None, str | None]:
    """
    Sucht die Meta-Zeile direkt vor dem Link, z.B.:
    'APA News  ·  04.02.2026, 18:23:00'
    und gibt (quelle, published_iso) zurück.
    """
    # Textknoten vor dem <a>, der ein Datum enthält
    meta_node = anchor.find_previous(
        string=re.compile(r"\d{2}\.\d{2}\.\d{4},\s*\d{2}:\d{2}:\d{2}")
    )
    if not meta_node:
        return None, None

    raw = meta_node.strip()
    # Split an "·"
    parts = [p.strip() for p in raw.split("·")]
    quelle = parts[0] if parts else None

    published_iso = None
    if len(parts) >= 2:
        dt_txt = parts[1]  # z.B. "04.02.2026, 18:23:00"
        try:
            dt = datetime.strptime(dt_txt, "%d.%m.%Y, %H:%M:%S")
            published_iso = dt.isoformat()
        except Exception:
            # Fallback: Rohstring speichern
            published_iso = dt_txt

    return quelle, published_iso


def fetch_wienerborse_news(limit: int | None = 100) -> List[Dict[str, Any]]:
    """
    Scrapt die allgemeine News-Seite der Wiener Börse.

    Erkennt News-Links an:
      <a class="sxp-no-ajax" href="/news/...">
         Headline...
      </a>
    """
    print(f"[WB] Fetching {NEWS_URL}")
    resp = requests.get(NEWS_URL, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    articles: List[Dict[str, Any]] = []

    # Alle News-Links: class="sxp-no-ajax" und href beginnt mit /news/
    anchors = soup.find_all(
        "a",
        class_="sxp-no-ajax",
        href=re.compile(r"^/news/"),
    )

    for a in anchors:
        headline = a.get_text(strip=True)
        if not headline:
            continue

        href = a.get("href") or ""
        if href.startswith("/"):
            url = BASE_URL + href
        else:
            url = href

        quelle, published_iso = _parse_meta_for_anchor(a)

        article = {
            "source": "wienerboerse",        # Datenquelle
            "category": "AT_Boerse",        # dein eigenes Kategorien-Label
            "ticker": None,                 # Mapping Firma -> Ticker kannst du später ergänzen
            "headline": headline,
            "url": url,
            "published_at": published_iso,  # ISO-String oder None
            "summary": None,
            "raw_text": None,               # wird später vom Enricher gefüllt
            "language": "de",
        }
        articles.append(article)

        if limit is not None and len(articles) >= limit:
            break

    print(f"[WB] Found {len(articles)} articles on {NEWS_URL}")
    return articles


if __name__ == "__main__":
    arts = fetch_wienerborse_news(limit=10)
    for a in arts:
        print(a["published_at"], "-", a["headline"], "->", a["url"])
