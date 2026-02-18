# src/news_db.py

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import sqlite3
from datetime import datetime
import hashlib
from src.config import DB_PATH

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            category TEXT,
            ticker TEXT,
            headline TEXT NOT NULL,
            url TEXT,
            published_at TEXT,
            fetched_at TEXT NOT NULL,
            summary TEXT,
            raw_text TEXT,
            language TEXT,
            unique_key TEXT UNIQUE
        );
        """
    )
    conn.commit()
    conn.close()


def make_unique_key(source: str, url: str | None, headline: str) -> str:
    base = f"{source}|{url or ''}|{headline}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def store_news_batch(articles: List[Dict[str, Any]]) -> int:
    """Speichert mehrere news; Duplikate (unique_key) werden ignoriert."""
    if not articles:
        return 0

    conn = get_connection()
    cur = conn.cursor()
    now_iso = datetime.utcnow().isoformat()

    inserted = 0

    for a in articles:
        source = a.get("source")
        category = a.get("category")
        ticker = a.get("ticker")
        headline = a.get("headline")
        url = a.get("url")
        published_at = a.get("published_at")
        summary = a.get("summary")
        raw_text = a.get("raw_text")
        language = a.get("language")

        if not headline or not source:
            continue

        unique_key = make_unique_key(source, url, headline)

        try:
            cur.execute(
                """
                INSERT OR IGNORE INTO news_articles
                (source, category, ticker, headline, url, published_at, fetched_at,
                 summary, raw_text, language, unique_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    source,
                    category,
                    ticker,
                    headline,
                    url,
                    published_at,
                    now_iso,
                    summary,
                    raw_text,
                    language,
                    unique_key,
                ),
            )
            if cur.rowcount > 0:
                inserted += 1
        except Exception as e:
            print(f"[NEWS_DB] Fehler beim Insert: {e}")

    conn.commit()
    conn.close()
    return inserted


if __name__ == "__main__":
    init_db()
    print(f"Initialisiert DB unter {DB_PATH}")
