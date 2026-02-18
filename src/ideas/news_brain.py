import feedparser
import requests
import json
import time

# Wir importieren deine anderen Dateien
# (Stelle sicher, dass sources.py und db.py im selben Ordner liegen!)
from sources import FEEDS
from src.ideas.db import init_db, save_news, get_latest_memory, update_memory, article_exists


# --- LLM FUNKTION ---
def analyze_market_situation(current_memory, new_articles):
    """
    Sendet neue Artikel + altes Gedächtnis an Llama.
    """
    # Wir fassen die neuen Artikel zu einem Textblock zusammen
    news_text = ""
    for art in new_articles:
        # --- KORREKTUR START ---
        # Wir nutzen .get(), das stürzt nicht ab, wenn das Feld fehlt.
        # Priorität: 1. summary, 2. description, 3. Leerer String
        summary_text = art.get('summary', art.get('description', ''))

        # Falls auch der Title fehlt (sehr selten), fangen wir das auch ab
        title_text = art.get('title', 'Ohne Titel')

        news_text += f"- {title_text}: {summary_text[:200]}\n"
        # --- KORREKTUR ENDE ---

    if not news_text:
        return None

    prompt = f"""
    Du bist ein KI-Strategist für den österreichischen Aktienmarkt (ATX).

    1. AKTUELLE LAGE (Was wir bisher wussten):
    {current_memory['summary']}

    2. NEUE NACHRICHTEN (Letzte 15 Min):
    {news_text}

    AUFGABE:
    Update die "Aktuelle Lage" basierend auf den neuen Nachrichten. 
    Ignoriere unwichtige News. Fokus auf Zinsen, Energie, ATX-Firmen, Österreichische Wirtschaft.

    Antworte NUR im JSON Format:
    {{
        "updated_summary": "Ein Text, der die alte Lage mit den neuen Infos verschmilzt (max 100 Wörter).",
        "market_sentiment": (Zahl zwischen -1.0 Bärenmarkt und +1.0 Bullenmarkt),
        "trade_signals": ["KAUFEN OMV", "VERKAUFEN ERSTE"] (oder leer lassen [])
    }}
    """

    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": "llama3.1",
            "prompt": prompt,
            "format": "json",
            "stream": False
        }, timeout=120)

        return json.loads(response.json()['response'])
    except Exception as e:
        print(f"LLM Fehler: {e}")
        return None

# --- MAIN LOOP ---
def run_cycle():
    print("\n--- Zyklus Start ---")

    # 1. Altes Gedächtnis laden
    current_memory = get_latest_memory()
    # Sicherstellen, dass wir Strings haben, falls DB leer war
    summary_text = current_memory['summary'] if current_memory else "Neutral"
    score = current_memory['score'] if current_memory else 0.0

    print(f"Status davor: {summary_text[:60]}... (Score: {score})")

    # 2. news Scrapen
    new_relevant_articles = []

    print("Scanne Feeds...")

    # HIER WAR DER FEHLER: Wir müssen durch die FEEDS iterieren!
    for category, urls in FEEDS.items():
        for url in urls:
            try:
                # Hier wird 'feed' definiert!
                feed = feedparser.parse(url)

                # Jetzt können wir feed.entries benutzen
                for entry in feed.entries[:15]:  # Nur die neuesten 5 pro Feed

                    # A) Check: Haben wir den schon in der DB?
                    if article_exists(entry.title):
                        continue  # Überspringen, kennen wir schon

                    # B) Check: Ist es relevant? (Keyword Watchlist)
                    # Wir kombinieren Titel und Beschreibung für die Suche

                    #full_text_search = (entry.title + " " + entry.get('description', '')).lower()

                    #if any(w.lower() in full_text_search for w in WATCHLIST):
                    #    print(f" -> Neuer Treffer: {entry.title}")
                    #    new_relevant_articles.append(entry)
                    print(entry)
                    new_relevant_articles.append(entry)

            except Exception as e:
                print(f"Fehler beim Feed {url}: {e}")
                continue

    # 3. Wenn neue Artikel da sind -> LLM fragen
    if new_relevant_articles:
        print(f"{len(new_relevant_articles)} neue relevante Artikel gefunden. Frage Llama...")

        # Hier übergeben wir das Dictionary aus der DB
        result = analyze_market_situation(current_memory, new_relevant_articles)

        if result:
            print("\n>>> AI ERGEBNIS:")
            print(f"Sentiment: {result.get('market_sentiment')}")
            print(f"Update: {result.get('updated_summary')}")
            print(f"Tradesignals:{result.get('trade_signals')}")

            # 4. Alles in die DB speichern!

            # A) Das neue Gedächtnis speichern
            update_memory(result.get('updated_summary'), result.get('market_sentiment'))
            print("[DB] Markt-Gedächtnis aktualisiert.")

            # B) Die Artikel als "gelesen" markieren
            for art in new_relevant_articles:
                # Wir speichern 0.0 als Sentiment für den einzelnen Artikel,
                # weil wir hier nur das Gesamt-Sentiment berechnet haben.
                save_news("RSS", art.title, art.link, 0.0)
            print(f"[DB] {len(new_relevant_articles)} Artikel gespeichert.")

        else:
            print("Fehler: Keine gültige Antwort vom LLM erhalten.")

    else:
        print("Keine neuen relevanten Nachrichten gefunden.")


if __name__ == "__main__":
    init_db()  # Initialisiert die DB beim Start

    # Endlosschleife
    while True:
        try:
            run_cycle()
        except Exception as e:
            print(f"Kritischer Fehler im Loop: {e}")

        wait_period=300

        print(f"Warte {wait_period/60} Minuten...")
        time.sleep(wait_period)