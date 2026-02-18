# db.py
import sqlite3
from datetime import datetime

DB_NAME = "trading_bot.db"


def get_connection():
    """Erstellt eine Verbindung zur Datenbank"""
    return sqlite3.connect(DB_NAME)


def init_db():
    """Erstellt die Tabellen, falls sie noch nicht existieren"""
    conn = get_connection()
    c = conn.cursor()

    # 1. Tabelle für einzelne news-Artikel (damit wir wissen, was wir schon gelesen haben)
    c.execute('''CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    source TEXT,
                    title TEXT UNIQUE,  -- UNIQUE verhindert doppelte Einträge
                    url TEXT,
                    sentiment_score REAL
                )''')

    # 2. Tabelle für das "Gedächtnis" (Memory) des Bots
    c.execute('''CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    summary_text TEXT,
                    market_score REAL
                )''')

    conn.commit()
    conn.close()
    print("[DB] Datenbank bereit und Tabellen geprüft.")


# --- FUNKTIONEN FÜR NEWS ---

def article_exists(title):
    """Prüft, ob wir den Artikel schon kennen"""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT 1 FROM news WHERE title = ?", (title,))
    exists = c.fetchone() is not None
    conn.close()
    return exists


def save_news(source, title, url, sentiment):
    """Speichert einen neuen Artikel"""
    if article_exists(title):
        return  # Schon da, nichts tun

    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO news (timestamp, source, title, url, sentiment_score) VALUES (?, ?, ?, ?, ?)",
                  (datetime.now(), source, title, url, sentiment))
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # Sollte durch article_exists eigentlich nicht passieren
    finally:
        conn.close()


# --- FUNKTIONEN FÜR DAS GEDÄCHTNIS ---

def get_latest_memory():
    """Holt den letzten Statusbericht. Wenn keiner da ist, gibt es einen Startwert."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT summary_text, market_score FROM memory ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()

    if row:
        return {"summary": row[0], "score": row[1]}
    else:
        # Startzustand, wenn der Bot zum allerersten Mal läuft
        return {"summary": "Der Markt ist neutral. Keine vorherigen Daten.", "score": 0.0}


def update_memory(summary, score):
    """Speichert das neue Gedächtnis"""
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO memory (timestamp, summary_text, market_score) VALUES (?, ?, ?)",
              (datetime.now(), summary, score))
    conn.commit()
    conn.close()


# In db.py hinzufügen:

def init_tech_db():
    conn = get_connection()
    c = conn.cursor()
    # Tabelle für Technische Signale
    c.execute('''CREATE TABLE IF NOT EXISTS tech_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    ticker TEXT,
                    price REAL,
                    rsi REAL,
                    sma_200 REAL,
                    signal_score REAL, -- Von -1 (Strong Sell) bis +1 (Strong Buy)
                    details TEXT
                )''')
    conn.commit()
    conn.close()
    print("[DB] Tech-Tabelle bereit.")

def save_tech_signal(ticker, price, rsi, sma, score, details):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO tech_signals (timestamp, ticker, price, rsi, sma_200, signal_score, details) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (datetime.now(), ticker, price, rsi, sma, score, details))
    conn.commit()
    conn.close()


# In db.py hinzufügen:

def get_latest_tech_signal(ticker):
    """Holt das aktuellste Tech-Signal für eine Aktie"""
    conn = get_connection()
    c = conn.cursor()
    # Wir sortieren nach ID absteigend, um das neuste zu kriegen
    c.execute("SELECT signal_score, details FROM tech_signals WHERE ticker = ? ORDER BY id DESC LIMIT 1", (ticker,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"score": row[0], "details": row[1]}
    return None


def get_latest_ticker_sentiment(ticker):
    """
    Versucht, spezifische news zu einer Aktie zu finden (letzte 24h).
    Falls keine da sind, gibt es 0 zurück.
    """
    conn = get_connection()
    c = conn.cursor()
    # Wir suchen im Titel nach dem Ticker-Namen (z.B. OMV)
    # Hinweis: Das ist ein einfacher Filter. In Produktion bräuchte man echtes 'Entity Recognition'.
    # Wir nehmen den 'cleanen' Namen ohne .VI
    clean_name = ticker.replace(".VI", "")

    query = f"%{clean_name}%"
    c.execute("SELECT sentiment_score FROM news WHERE title LIKE ? ORDER BY id DESC LIMIT 5", (query,))
    rows = c.fetchall()
    conn.close()

    if rows:
        # Durchschnitt der letzten 5 Artikel
        scores = [r[0] for r in rows]
        return sum(scores) / len(scores)
    return 0.0

# Ganz unten in db.py:

if __name__ == '__main__':
    print("Initialisiere Datenbank...")
    init_db()       # Erstellt 'news' und 'memory' Tabellen
    init_tech_db()  # Erstellt 'tech_signals' Tabelle
    print("Fertig. Alle Tabellen sind bereit.")