import time
from db import get_latest_memory, get_latest_tech_signal, get_latest_ticker_sentiment
from tech_brain import TICKER_MAP

# --- KONFIGURATION ---
# Wir senken die Hürde leicht, aber bleiben sicher
MIN_SCORE_STRONG_BUY = 0.6
MIN_SCORE_WEAK_BUY = 0.35  # War vorher strenger
CAPITAL = 10000


def calculate_position_size(capital, risk_factor):
    """
    Berechnet die Positionsgröße basierend auf dem Risiko.
    risk_factor: 1.0 = volle Größe, 0.5 = halbe Größe
    """
    base_size = capital * 0.10  # Max 10% pro Trade
    return base_size * risk_factor


def run_decision_engine():
    print("\n=== DER CEO BOT (Decision Engine v2) ===")

    # 1. Makro-Check
    market_memory = get_latest_memory()
    market_score = market_memory['score']

    # Wenn Markt-Score 0 ist (weil DB neu), nehmen wir 0 an statt Fehler
    if market_score is None: market_score = 0.0

    print(f"Allgemeine Marktstimmung (Makro): {market_score:.2f}")

    # NOTBREMSE: Wenn der Gesamtmarkt crasht, kaufen wir gar nichts.
    if market_score < -0.6:
        print("!!! WARNUNG: Markt-Crash Modus. Trading ausgesetzt. !!!")
        return

    print("\n{:<10} | {:<6} | {:<6} | {:<6} | {:<15}".format("TICKER", "TECH", "NEWS", "TOTAL", "ACTION"))
    print("-" * 55)

    opportunities = []

    for ticker in TICKER_MAP:
        # A) Tech Score holen
        tech_data = get_latest_tech_signal(ticker)
        if not tech_data:
            continue  # Keine Daten -> Überspringen

        tech_score = tech_data['score']

        # B) news Score holen
        news_score = get_latest_ticker_sentiment(ticker)

        # C) Die neue "Smart-Gewichtung"
        # Wenn news fehlen (0.0), vertrauen wir Tech mehr, aber reduzieren die Positionsgröße
        if news_score == 0.0:
            combined_score = (tech_score * 0.8) + (market_score * 0.2)
            confidence = 0.5  # Geringere Confidence, weil news fehlen
        else:
            combined_score = (tech_score * 0.5) + (news_score * 0.3) + (market_score * 0.2)
            confidence = 1.0

        # D) Entscheidung
        action = "HOLD"
        risk_factor = 0.0

        if combined_score >= MIN_SCORE_STRONG_BUY:
            action = "STRONG BUY"
            risk_factor = 1.0 * confidence
        elif combined_score >= MIN_SCORE_WEAK_BUY:
            action = "BUY (Weak)"
            risk_factor = 0.5 * confidence  # Vorsichtiger Einstieg
        elif combined_score <= -0.2:
            action = "SELL"

        print(f"{ticker:<10} | {tech_score:>6.1f} | {news_score:>6.1f} | {combined_score:>6.2f} | -> {action}")

        if "BUY" in action:
            opportunities.append((ticker, combined_score, risk_factor, action))

    # --- ORDER BUCH ---
    print("\n--- ORDER EMPFEHLUNGEN ---")
    if not opportunities:
        print("Heute keine Trades.")
    else:
        opportunities.sort(key=lambda x: x[1], reverse=True)
        for ticker, score, risk, action in opportunities:
            money = calculate_position_size(CAPITAL, risk)
            print(f"💰 {ticker}: Score {score:.2f} ({action}) -> Kaufe für € {money:.0f}")


if __name__ == "__main__":
    run_decision_engine()