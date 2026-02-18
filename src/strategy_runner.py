# src/strategy_runner.py

import asyncio
import json
import sqlite3
import time
import sys
from pathlib import Path
from typing import List, Dict, Any

# Pfad-Setup für Imports
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from trade_brain import HttpClient, decide_for_ticker
    from config import DB_PATH
    from sector_strategies import get_strategy_for_ticker
    from news.news_sentiment import calculate_news_sentiment, format_sentiment_for_prompt
except ImportError as e:
    print(f"[CRITICAL] Import Error: {e}")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# Datenbank-Helper
# ---------------------------------------------------------------------------

def get_db_connection():
    if not Path(DB_PATH).exists():
        return None
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_top_news_from_db(
        categories: List[str] = None,
        ticker: str = None,
        limit: int = 5,
        min_importance: int = 0
) -> List[str]:
    """
    Holt News flexibel aus der DB.
    """
    conn = get_db_connection()
    if not conn:
        return []

    cursor = conn.cursor()

    # 1. Prüfen welche Spalten existieren
    cursor.execute("PRAGMA table_info(news_articles)")
    cols = [r['name'] for r in cursor.fetchall()]
    has_importance = 'importance' in cols
    has_classified_ticker = 'classified_ticker' in cols
    has_classified_category = 'classified_category' in cols

    # 2. SELECT-Felder dynamisch aufbauen
    select_fields = ["headline", "source", "published_at", "summary", "category"]
    if has_classified_category:
        select_fields.append("classified_category")

    query = f"SELECT {', '.join(select_fields)} FROM news_articles WHERE 1=1"
    params = []

    # 3. Ticker Filter
    if ticker:
        # Suche in 'ticker', 'category' (ticker:XYZ) und 'classified_ticker'
        ticker_clauses = ["ticker = ?", "category = ?"]
        params.append(ticker)
        params.append(f"ticker:{ticker}")

        if has_classified_ticker:
            ticker_clauses.append("classified_ticker = ?")
            params.append(ticker)

        query += " AND (" + " OR ".join(ticker_clauses) + ")"

    # 4. Kategorie Filter
    if categories:
        cat_clauses = []
        for cat in categories:
            cat_clauses.append("category LIKE ?")
            params.append(f"%{cat}%")

            if has_classified_category:
                cat_clauses.append("classified_category LIKE ?")
                params.append(f"%{cat}%")

        if cat_clauses:
            query += " AND (" + " OR ".join(cat_clauses) + ")"

    # 5. Sortierung & Limit
    if has_importance:
        if min_importance > 0:
            query += " AND COALESCE(importance, 0) >= ?"
            params.append(min_importance)
        query += " ORDER BY importance DESC, published_at DESC LIMIT ?"
    else:
        query += " ORDER BY published_at DESC LIMIT ?"

    params.append(limit)

    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
    except Exception as e:
        print(f"[DB QUERY ERROR] {e}")
        return []
    finally:
        conn.close()

    # 6. Ergebnisse formatieren
    news_list = []
    for row in rows:
        head = row["headline"]
        src = row["source"]

        # Kategorie sicher bestimmen (ohne .get())
        cat = "NoCat"
        if has_classified_category and row["classified_category"]:
            cat = row["classified_category"]
        elif row["category"]:
            cat = row["category"]

        # Summary sicher lesen
        summ = row["summary"] if row["summary"] else ""
        summ_short = (summ[:120] + '..') if len(summ) > 120 else summ

        entry = f"- {head} ({src}) [{cat}]: {summ_short}"
        news_list.append(entry)

    return news_list


def get_macro_context() -> Dict[str, str]:
    print("[INIT] Fetching Macro News...")

    # 1. GLOBAL: Quell-Kategorien (Welt_Macro) + Keywords
    global_keywords = ["Welt_Macro", "Europa_Macro", "World", "Global", "US"]
    global_news = fetch_top_news_from_db(categories=global_keywords, limit=5, min_importance=6)

    if not global_news:
        print("  [INFO] No high-importance global news, fetching latest...")
        global_news = fetch_top_news_from_db(categories=global_keywords, limit=5, min_importance=0)

    # 2. AUSTRIA: Quell-Kategorien (AT_...) + Keywords
    austria_keywords = ["AT_Boerse", "AT_Finanzen", "AT_Wirtschaft", "ATX", "Austria", "Wien"]
    austria_news = fetch_top_news_from_db(categories=austria_keywords, limit=5, min_importance=5)

    if not austria_news:
        print("  [INFO] No high-importance Austria news, fetching latest...")
        austria_news = fetch_top_news_from_db(categories=austria_keywords, limit=5, min_importance=0)

    return {
        "global": "\n".join(global_news) if global_news else "No global news found in DB.",
        "austria": "\n".join(austria_news) if austria_news else "No Austrian news found in DB."
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

async def run_swarm_pipeline(enable_sentiment: bool = True):
    # 1. Macro Context laden
    try:
        macro_ctx = get_macro_context()
    except Exception as e:
        print(f"[ERROR] Macro Context failed: {e}")
        macro_ctx = {"global": "Error loading", "austria": "Error loading"}

    print("==================================================")
    print("GLOBAL CONTEXT:\n" + macro_ctx["global"])
    print("\nAUSTRIA CONTEXT:\n" + macro_ctx["austria"])
    print("==================================================\n")

    client = HttpClient()

    processed_files = list(PROCESSED_DIR.glob("*_processed.json"))
    if not processed_files:
        print(f"[ERROR] No processed files found in {PROCESSED_DIR}")
        return

    for p_file in processed_files:
        try:
            ticker = p_file.name.replace("_processed.json", "")
            print(f"[RUN] Processing {ticker}...")

            with open(p_file, "r", encoding="utf-8") as f:
                p_data = json.load(f)

            # 3. Company News aus DB
            company_news = fetch_top_news_from_db(ticker=ticker, limit=5, min_importance=1)

            if not company_news:
                company_news = fetch_top_news_from_db(ticker=ticker, limit=5, min_importance=0)

            if not company_news:
                company_news = ["No specific news found in DB for this ticker."]
            
            # 3.1 NEWS SENTIMENT (neu! - kann fehlschlagen wenn Ollama überlastet)
            sentiment_text = ""
            if enable_sentiment:
                try:
                    news_sentiment = calculate_news_sentiment(
                        ticker, 
                        days=7, 
                        min_importance=3,
                        max_headlines=3  # Limit to 3 für Stabilität
                    )
                    
                    # Check if sentiment analysis worked
                    if news_sentiment.get('error'):
                        print(f"[INFO] {ticker}: Sentiment skipped ({news_sentiment['error']})")
                    else:
                        sentiment_text = format_sentiment_for_prompt(news_sentiment)
                        
                except Exception as e:
                    print(f"[WARN] News sentiment failed for {ticker}: {str(e)[:100]}")
                    # Continue without sentiment
            else:
                print(f"[INFO] {ticker}: Sentiment disabled")
            
            # 3.2 SECTOR CONTEXT (neu!)
            try:
                sector_strategy = get_strategy_for_ticker(ticker)
                sector_context = sector_strategy.build_sector_context()
            except Exception as e:
                print(f"[WARN] Sector context failed for {ticker}: {e}")
                sector_context = ""

            # 4. SWARM ENTSCHEIDUNG (jetzt mit Sector Context + Sentiment)
            final, all_votes = await decide_for_ticker(
                client,
                ticker,
                p_data,
                macro_ctx,
                company_news,
                sector_context=sector_context + "\n" + sentiment_text
            )

            if final:
                color = "\033[92m" if final["action"] == "BUY" else "\033[91m" if final[
                                                                                      "action"] == "SELL" else "\033[93m"
                reset = "\033[0m"
                print(
                    f"  => RESULT: {color}{final['action']}{reset} (Conf: {final['avg_confidence']}) | Votes: {final['vote_count']}")
            else:
                print("  => RESULT: NO CONSENSUS (HOLD)")

            print("-" * 50)
            time.sleep(1)

        except Exception as e:
            print(f"[ERROR] Failed {p_file.name}: {e}")


if __name__ == "__main__":
    import sys
    
    # Optional: Disable sentiment analysis via command line
    # Usage: python strategy_runner.py --no-sentiment
    enable_sentiment = "--no-sentiment" not in sys.argv
    
    if not enable_sentiment:
        print("[INFO] News sentiment analysis DISABLED (faster, more stable)")
    
    try:
        asyncio.run(run_swarm_pipeline(enable_sentiment=enable_sentiment))
    except KeyboardInterrupt:
        print("\n[STOP] User interrupted.")