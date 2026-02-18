import time
import sys
from datetime import datetime

# Wir importieren deine Module
# WICHTIG: Die Dateien müssen im selben Ordner liegen!
try:
    from db import init_db, init_tech_db
    from news_brain import run_cycle as run_news_cycle
    # Falls news_brain.py direkt im Ordner liegt, nimm diese Zeile stattdessen:
    # from news_brain import run_cycle as run_news_cycle

    from tech_brain import run_tech_cycle
    # Falls tech_brain.py direkt im Ordner liegt:
    # from tech_brain import run_tech_cycle

    from main_brain import run_decision_engine
    # Falls main_brain.py direkt im Ordner liegt:
    # from main_brain import run_decision_engine

except ImportError as e:
    print("FEHLER BEIM IMPORTIEREN:")
    print(f"{e}")
    print("Tipp: Stelle sicher, dass 'run_bot.py' im selben Ordner wie die anderen Skripte liegt")
    print("oder passe die Import-Pfade oben im Code an.")
    sys.exit(1)

# Konfiguration
SLEEP_MINUTES = 5
SLEEP_SECONDS = SLEEP_MINUTES * 60


def main():
    print("========================================")
    print(f"   SERVUS ALPHA - AUTOPILOT GESTARTET   ")
    print(f"   Intervall: Alle {SLEEP_MINUTES} Minuten")
    print("========================================\n")

    # 1. Datenbanken einmalig initialisieren
    print("[INIT] Prüfe Datenbanken...")
    init_db()
    init_tech_db()
    print("[INIT] Datenbanken bereit.\n")

    # 2. Endlosschleife
    while True:
        start_time = datetime.now()
        timestamp = start_time.strftime("%H:%M:%S")

        print(f">>> START ZYKLUS um {timestamp} <<<")

        # --- SCHRITT 1: NEWS (Das Ohr) ---
        try:
            print("\n[1/3] Starte news-Analyse...")
            # Wir rufen die Funktion auf, die einen Durchlauf macht
            run_news_cycle()
        except Exception as e:
            print(f"!!! FEHLER bei news-Cycle: {e}")

        # --- SCHRITT 2: TECH (Das Auge) ---
        try:
            print("\n[2/3] Starte Technische Analyse...")
            run_tech_cycle()
        except Exception as e:
            print(f"!!! FEHLER bei Tech-Cycle: {e}")

        # --- SCHRITT 3: BRAIN (Der Chef) ---
        try:
            print("\n[3/3] Starte Decision Engine...")
            run_decision_engine()
        except Exception as e:
            print(f"!!! FEHLER bei Decision Engine: {e}")

        # --- WARTEZEIT ---
        end_time = datetime.now()
        duration = (end_time - start_time).seconds

        print(f"\n>>> ZYKLUS BEENDET (Dauer: {duration}s) <<<")
        print(f"Schlafe {SLEEP_MINUTES} Minuten bis zum nächsten Run...")
        print("-" * 40)

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    # Strg+C zum Beenden abfangen
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAutopilot vom Benutzer beendet. Pfiat di!")