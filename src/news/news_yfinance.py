from typing import Iterable
import sqlite3
from datetime import datetime, timezone, timedelta
import ast
import yfinance as yf
from src.config import DB_PATH, TICKERS

def _unwrap_yf_content(item: dict) -> dict:
    """
    yfinance-news kommen aktuell als {'id': ..., 'content': '<python-dict-as-string>'}.
    Diese Funktion gibt immer ein dict der inneren Struktur zurück.
    """
    content = item.get("content")

    # Falls yfinance irgendwann wieder echte dicts liefert:
    if isinstance(content, dict):
        return content

    # Aktuell: content ist ein String mit einem Python-dict-Literal
    if isinstance(content, str):
        try:
            return ast.literal_eval(content)
        except Exception as e:
            print(f"[YF] Konnte content nicht parsen: {e}")
            return {}

    return {}

def _parse_yf_datetime(inner: dict) -> datetime:
    """
    Erwartet das 'innere' news-Dict (nach _unwrap_yf_content).
    Nutzt pubDate oder displayTime. Fallback: jetzt.
    """
    # pubDate: "2026-01-29T10:38:12Z"
    pub = inner.get("pubDate") or inner.get("displayTime")
    if isinstance(pub, str):
        try:
            return datetime.fromisoformat(pub.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            pass

    # Fallback: jetzt
    return datetime.now(timezone.utc)


def _extract_yf_url(inner: dict) -> str | None:
    """
    Erwartet das innere news-Dict. Sucht eine brauchbare URL.
    """
    # 1) direktes link-Feld (falls es mal existiert)
    link = inner.get("link")
    if isinstance(link, str):
        return link

    # 2) canonicalUrl
    canon = inner.get("canonicalUrl") or inner.get("canonical_url")
    if isinstance(canon, dict) and isinstance(canon.get("url"), str):
        return canon["url"]

    # 3) clickThroughUrl
    click = inner.get("clickThroughUrl")
    if isinstance(click, dict) and isinstance(click.get("url"), str):
        return click["url"]

    # Wenn nichts gefunden wurde → Debug
    print(f"[YF] Kein URL-Feld im inneren dict, keys={list(inner.keys())}")
    return None


def fetch_yf_news_for_tickers(
    tickers: Iterable[str],
    lookback_days: int = 7,
) -> None:
    """
    Holt news via yfinance und schreibt sie direkt in die news_articles-DB.

    Erwartetes Schema der Tabelle news_articles (mindestens):
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT,
        category TEXT,
        title TEXT,
        url TEXT UNIQUE,
        published_at TEXT,
        summary TEXT,
        raw_text TEXT,
        created_at TEXT,   -- optional mit DEFAULT CURRENT_TIMESTAMP
        updated_at TEXT    -- optional

    raw_text wird hier NULL gelassen, damit der Enricher später
    den Artikel-Text per HTTP/PDF etc. nachzieht.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    total_new = 0
    total_seen = 0

    for symbol in tickers:
        print(f"[YF] Fetching news for {symbol}")
        try:
            t = yf.Ticker(symbol)
            raw_news = t.news or []
        except Exception as e:
            print(f"[YF] Fehler bei {symbol}: {e}")
            continue

        print(f"[YF] {symbol}: {len(raw_news)} raw-items")
        total_seen += len(raw_news)

        for item in raw_news:
            inner = _unwrap_yf_content(item)


            dt = _parse_yf_datetime(inner)
            url = _extract_yf_url(inner)
            title = (inner.get("title") or "").strip()
            summary = (inner.get("summary") or "").strip()

            # Publisher / Quelle
            provider = None
            prov = inner.get("provider") or inner.get("publisher")
            if isinstance(prov, dict):
                provider = prov.get("displayName")
            elif isinstance(prov, str):
                provider = prov

            source = f"yfinance:{provider}" if provider else "yfinance"
            feed_group = f"ticker:{symbol}"

            fetched_at = datetime.now(timezone.utc).isoformat()

            try:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO news_articles
                        (source, category, headline, url, published_at, summary, raw_text, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, NULL, ?);
                    """,
                    (
                        source,
                        feed_group,  # category
                        title,  # headline
                        url,
                        dt.isoformat(),  # published_at
                        summary,
                        fetched_at,  # NEU
                    ),
                )
                if cur.rowcount > 0:
                    total_new += 1
            except Exception as e:
                print(f"[YF] Insert-Fehler für {url}: {e}")



    conn.commit()
    conn.close()

    print(f"[YF] Seen raw-items (gesamt): {total_seen}")
    print(f"[YF] Neu eingefügte yfinance-news: {total_new}")
