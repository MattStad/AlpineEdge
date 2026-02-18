# src/news_enricher.py
#
# Lädt aus news.db alle Artikel ohne raw_text,
# scrapt die URL, extrahiert den Haupttext und
# speichert raw_text (und ggf. summary) in die DB.

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import sqlite3
import time
from datetime import datetime
import re
from urllib.error import HTTPError

import requests
from bs4 import BeautifulSoup
from src.config import DB_PATH

from requests.exceptions import HTTPError


# -------------------------------------------
# DB-Helper
# -------------------------------------------

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


# -------------------------------------------
# HTTP + HTML Parsing
# -------------------------------------------

import requests
from requests.exceptions import HTTPError
from io import BytesIO
from PyPDF2 import PdfReader

# falls du schon extract_main_text(html: str) hast, lassen wir die weiterleben


from io import BytesIO
from PyPDF2 import PdfReader
from PyPDF2.errors import FileNotDecryptedError

def extract_pdf_text_from_bytes(data: bytes) -> str:
    """Extrahiert Text aus einem PDF-Byte-Stream, inkl. Handling für verschlüsselte PDFs."""
    try:
        reader = PdfReader(BytesIO(data))

        # Falls verschlüsselt: zuerst versuchen ohne Passwort zu entschlüsseln
        if reader.is_encrypted:
            try:
                # Viele IR-/Börse-PDFs sind "owner-geschützt", aber mit leerem Passwort lesbar
                decrypt_result = reader.decrypt("")
                print(f"[ENRICH] PDF ist verschlüsselt, decrypt_result={decrypt_result}")
                # decrypt_result kann 0/1/2 sein, 0 = fail
                if decrypt_result == 0:
                    print("[ENRICH] PDF-Verschlüsselung konnte nicht entschlüsselt werden.")
                    return ""
            except Exception as e:
                print(f"[ENRICH] Fehler beim Entschlüsseln des PDFs: {e}")
                return ""

        pages = []
        for page in reader.pages:
            try:
                txt = page.extract_text()
            except FileNotDecryptedError as e:
                print(f"[ENRICH] Seite nicht entschlüsselt lesbar: {e}")
                return ""
            if txt:
                pages.append(txt)

        full_text = "\n\n".join(pages)
        return full_text

    except Exception as e:
        print(f"[ENRICH] Fehler beim PDF-Parsing: {e}")
        return ""


def fetch_article_text(url: str, timeout: int = 10) -> str | None:
    """
    Holt den Artikelinhalt:
    - Wenn HTML -> extract_main_text(resp.text)
    - Wenn PDF  -> extract_pdf_text_from_bytes(resp.content)
    Gibt fertigen Rohtext zurück.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except HTTPError as e:
        print(f"[ENRICH] HTTP-Fehler {e.response.status_code} bei {url}")
        return None
    except Exception as e:
        print(f"[ENRICH] Allgemeiner Fehler beim Holen von {url}: {e}")
        return None

    content_type = resp.headers.get("Content-Type", "").lower()
    # Debug:
    # print(f"[ENRICH] Content-Type für {url}: {content_type}")

    # 📄 PDF-Fall (Wiener Börse, IR-Dokumente etc.)
    if "pdf" in content_type:
        print(f"[ENRICH] PDF erkannt bei {url}")
        text = extract_pdf_text_from_bytes(resp.content)
        return text if text else None

    # Sonst: HTML → existierende HTML-Extraktion
    html = resp.text
    text = extract_main_text(html)
    return text if text else None


def fetch_html(url: str, timeout: int = 10) -> str | None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except HTTPError as e:
        print(f"[ENRICH] HTTP-Fehler {e.response.status_code} bei {url}")
        return None
    except Exception as e:
        print(f"[ENRICH] Allgemeiner Fehler beim Holen von {url}: {e}")
        return None

    return resp.text

def extract_main_text(html: str) -> str:
    """
    Versucht den Haupttext aus einer Nachrichten-Seite zu extrahieren.
    Generic Heuristik: <article>, dann typische Content-Divs, dann Fallback.
    """
    soup = BeautifulSoup(html, "html.parser")

    # offensichtliche Noise-Tags entfernen
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form", "aside"]):
        tag.decompose()

    candidates = []

    # 1) <article>
    article = soup.find("article")
    if article:
        candidates.append(article)

    # 2) divs mit typischen Klassen
    key_classes = [
        "article", "article-body", "articleBody",
        "content", "main-content", "content-main",
        "post-content", "entry-content",
        "text", "body__inner", "story",
    ]

    def class_matches(cls: Optional[str]) -> bool:
        if not cls:
            return False
        if isinstance(cls, list):
            return any(any(k.lower() in c.lower() for k in key_classes) for c in cls)
        return any(k.lower() in cls.lower() for k in key_classes)

    for div in soup.find_all("div", class_=class_matches):
        candidates.append(div)

    # 3) Fallback: body
    if soup.body:
        candidates.append(soup.body)

    # längsten Text nehmen
    best_node = None
    best_len = 0
    for node in candidates:
        text = node.get_text(" ", strip=True)
        if len(text) > best_len:
            best_len = len(text)
            best_node = node

    if not best_node:
        return ""

    text = best_node.get_text(" ", strip=True)

    # bisschen aufräumen: mehrfach Spaces, Zeilenumbrüche usw.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_summary_from_text(text: str, max_chars: int = 400) -> str:
    """
    Sehr einfache Zusammenfassung: erste 1–3 Sätze bis max_chars.
    Später kannst du hier was Schlaueres einbauen.
    """
    if not text:
        return ""

    # Spilt rudimentär nach Satzzeichen
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = ""
    for s in sentences:
        if not s:
            continue
        if len(summary) + len(s) + 1 > max_chars:
            break
        summary = (summary + " " + s).strip()
        if len(summary) >= max_chars * 0.7:  # reicht meist
            break

    if not summary:
        summary = text[:max_chars]

    return summary.strip()


# -------------------------------------------
# Enrichment-Logic
# -------------------------------------------

def enrich_one_article(conn: sqlite3.Connection, article_id: int, url: str, old_summary: str | None) -> bool:
    print(f"[ENRICH] ID={article_id}, URL={url}")
    cur = conn.cursor()

    # 1) Text holen (HTML oder PDF)
    raw_text = fetch_article_text(url)

    if raw_text is None:
        # z.B. 404, 401, Netzwerkfehler, PDF-Parsing kaputt, etc.
        print(f"[ENRICH] Kein Text für ID={article_id}, markiere als HTTP/NO_CONTENT_ERROR")
        cur.execute(
            "UPDATE news_articles SET raw_text = ? WHERE id = ?;",
            ("__HTTP_OR_NO_CONTENT__", article_id),
        )
        conn.commit()
        return False

    if len(raw_text.strip()) < 100:
        # Sehr kurze Texte als „nutzlos“ behandeln
        print(f"[ENRICH] Zu wenig Text extrahiert ({len(raw_text)} chars), markiere als NO_TEXT_FOUND.")
        cur.execute(
            "UPDATE news_articles SET raw_text = ? WHERE id = ?;",
            ("__NO_TEXT_FOUND__", article_id),
        )
        conn.commit()
        return False

    # 2) Summary bauen, falls nötig
    new_summary = old_summary
    if not new_summary or len(str(new_summary)) < 30:
        new_summary = make_summary_from_text(raw_text)

    # 3) In DB schreiben
    cur.execute(
        """
        UPDATE news_articles
        SET raw_text = ?, summary = ?
        WHERE id = ?;
        """,
        (raw_text, new_summary, article_id),
    )
    conn.commit()
    print(f"[ENRICH] Aktualisiert ID={article_id} (Text {len(raw_text)} chars)")
    return True


# news_enricher.py

def run_enrichment(batch_size: int = 20, sleep_seconds: float = 1.0, max_articles: int | None = None):
    conn = sqlite3.connect(DB_PATH)
    processed = 0

    while True:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, url, summary
            FROM news_articles
            WHERE raw_text IS NULL
              AND url IS NOT NULL
            ORDER BY id ASC
            LIMIT ?;
            """,
            (batch_size,),
        )
        rows = cur.fetchall()

        if not rows:
            print("[ENRICH] Keine weiteren Artikel ohne raw_text gefunden. Fertig.")
            break

        print(f"[ENRICH] Verarbeite Batch mit {len(rows)} Artikeln...")

        for article_id, url, summary in rows:

            ok = enrich_one_article(conn, article_id, url, summary)
            processed += 1

            if max_articles is not None and processed >= max_articles:
                print(f"[ENRICH] max_articles={max_articles} erreicht, Stop.")
                conn.close()
                return

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    conn.close()


if __name__ == "__main__":
    start = datetime.now().isoformat(timespec="seconds")
    print(f"[ENRICH] Start: {start}")
    run_enrichment(batch_size=20, sleep_seconds=1.0)
    end = datetime.now().isoformat(timespec="seconds")
    print(f"[ENRICH] Ende: {end}")
