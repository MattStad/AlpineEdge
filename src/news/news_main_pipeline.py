from news_rss_fetcher import fetch_rss_feeds
from news_yfinance import fetch_yf_news_for_tickers
from news_enricher import run_enrichment
from news_classifier import classify_unclassified_news, print_classification_stats
from src.config import TICKERS
from news_db import store_news_batch
from src.news.news_wienerbörse_fetcher import fetch_wienerborse_news


def run_once(classify: bool = True, min_importance: int = 3):
    """
    Führt die komplette News-Pipeline aus.

    Args:
        classify: Ob automatische Klassifizierung durchgeführt werden soll
        min_importance: Min. Wichtigkeit - Artikel darunter werden gelöscht
    """
    print("========== NEWS PIPELINE RUN ==========")

    # 1) RSS
    print("[PIPELINE] Step 1: Fetch RSS/news-Feeds")
    rss_articles = fetch_rss_feeds()
    print(f"[PIPELINE] RSS: {len(rss_articles)} Artikel geholt")

    # 2) yfinance → direkt in DB
    print("[PIPELINE] Step 2: Fetch yfinance news")
    fetch_yf_news_for_tickers(TICKERS, lookback_days=365)

    # 3) Wiener Börse
    print("[PIPELINE] Step 3: WienerBörse-Feeds")
    wb_articles = fetch_wienerborse_news()
    print(f"[PIPELINE] WB: {len(wb_articles)} Artikel geholt")

    # Speichere RSS + WB
    articles = rss_articles + wb_articles
    store_news_batch(articles)

    # 4) Enrichment (Text-Extraktion)
    print("[PIPELINE] Step 4: Enrich articles without raw_text")
    run_enrichment(batch_size=20, sleep_seconds=0.5, max_articles=200)

    # 5) Klassifizierung (NEU!)
    if classify:
        print("[PIPELINE] Step 5: Classify articles with AI")
        try:
            classified_count = classify_unclassified_news(
                batch_size=20,
                max_articles=None,  # Alle klassifizieren
                model_name=None,  # Auto-detect Modell
                min_importance=min_importance,
            )
            print(f"[PIPELINE] ✓ {classified_count} Artikel klassifiziert")

            # Zeige Statistiken
            print_classification_stats()

        except Exception as e:
            print(f"[PIPELINE] FEHLER bei Klassifizierung: {e}")
            print("[PIPELINE] Mögliche Lösungen:")
            print("[PIPELINE]   1. Prüfe ob Ollama läuft: ollama serve")
            print("[PIPELINE]   2. Installiere Modell: ollama pull llama3.2:3b")
            print("[PIPELINE]   3. Nutze Fix-Script: python fix_ollama_model.py")
            print("[PIPELINE] Pipeline läuft ohne Klassifizierung weiter...")

    print("[PIPELINE] Done.")


def run_daemon(interval_minutes: int = 60, classify: bool = True):
    """
    Führt die Pipeline regelmäßig aus.

    Args:
        interval_minutes: Intervall in Minuten
        classify: Ob Klassifizierung aktiviert sein soll
    """
    import time

    while True:
        try:
            run_once(classify=classify)
        except Exception as e:
            print(f"[PIPELINE] Fehler in run_once: {e}")

        print(f"[PIPELINE] Warte {interval_minutes} Minuten bis zum nächsten Lauf...")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    import sys

    # Kommandozeilen-Argumente
    if len(sys.argv) > 1 and sys.argv[1] == "daemon":
        # Daemon-Modus
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        run_daemon(interval_minutes=interval, classify=True)
    else:
        # Einmaliger Lauf
        run_once(classify=True, min_importance=3)
