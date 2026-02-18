import feedparser
import trafilatura
import requests
import json
import time

# --- KONFIGURATION ---
# Hier definieren wir, was wir scannen wollen
RSS_FEEDS = [
    "https://www.ots.at/rss/b0012",  # APA OTS Wirtschaft (Ad-hoc Meldungen!)
    "https://www.derstandard.at/rss/wirtschaft",
    "https://diepresse.com/rss/wirtschaft"
]

# Nach diesen Firmen suchen wir (ATX Prime Auswahl)
WATCHLIST = ["OMV", "voestalpine", "Erste Group", "Raiffeisen", "Verbund", "Andritz", "Wienerberger"]


def ask_ollama(text):
    """Sendet den Text an dein lokales Llama 3.1"""
    prompt = f"""
    Du bist ein harter Finanzanalyst für den österreichischen Aktienmarkt.
    Analysiere diesen Nachrichtenartikel.

    Ausgabeformat (JSON):
    {{
        "sentiment_score": (Zahl zwischen -1.0 sehr negativ und +1.0 sehr positiv),
        "company": (Name der betroffenen Firma aus dem Text),
        "summary": (Ein kurzer Satz Zusammenfassung auf Deutsch)
    }}

    Artikel:
    {text[:2500]} 
    """

    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": "llama3.1",
            "prompt": prompt,
            "format": "json",  # WICHTIG: Erzwingt strukturiertes Datenformat
            "stream": False
        })
        return json.loads(response.json()['response'])
    except Exception as e:
        print(f"Fehler bei Ollama: {e}")
        return None


def run_bot():
    print("--- Trading Bot 'ServusAlpha' gestartet ---")
    print(f"Überwache {len(WATCHLIST)} Firmen in {len(RSS_FEEDS)} News-Quellen.")

    # Wir merken uns Artikel, die wir schon kennen, um Doppelungen zu vermeiden
    seen_links = set()

    while True:
        print(f"\n[{time.strftime('%H:%M:%S')}] Scanne Feeds...")

        for url in RSS_FEEDS:
            try:
                feed = feedparser.parse(url)

                for entry in feed.entries[:5]:  # Nur die 5 neuesten pro Feed
                    if entry.link in seen_links:
                        continue

                    # Checken ob eine unserer Firmen in der Headline vorkommt
                    # (Einfacher Filter, später machen wir das besser)
                    found_companies = [comp for comp in WATCHLIST if comp.lower() in entry.title.lower()]

                    if found_companies:
                        print(f"\n>>> TREFFER: {entry.title}")
                        print(f"    Firma: {found_companies}")

                        # Text holen
                        downloaded = trafilatura.fetch_url(entry.link)
                        text = trafilatura.extract(downloaded)

                        if text:
                            print("    Analysiere mit AI...")
                            analysis = ask_ollama(text)
                            print(f"    ERGEBNIS: {analysis}")
                            # Hier würde später der 'Kauf'-Befehl stehen

                        seen_links.add(entry.link)
            except Exception as e:
                print(f"Fehler beim Feed {url}: {e}")

        print("Schlafe 60 Sekunden...")
        time.sleep(60)


if __name__ == "__main__":
    run_bot()