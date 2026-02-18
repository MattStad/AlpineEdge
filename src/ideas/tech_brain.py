import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from db import init_tech_db, save_tech_signal

# -----------------------------
# Ticker Mapping (ATX + ATX Prime)
# -----------------------------
# SPI.VI habe ich entfernt, da es oft Datenprobleme hat (Delisting/Fusion)
TICKER_MAP = [
    "ANDR.VI", "BG.VI", "CAI.VI", "EBS.VI", "FLU.VI", "LNZ.VI", "MMK.VI",
    "OMV.VI", "POST.VI", "RBI.VI", "SBO.VI", "STR.VI", "TKA.VI", "UQA.VI",
    "VOE.VI", "VIG.VI", "VER.VI", "ATS.VI", "DOC.VI", "ROS.VI",
    "WIE.VI", "FQT.VI", "MARI.VI"
]


# -----------------------------
# RSI Berechnung
# -----------------------------
def compute_rsi(close_series, period=14):
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# -----------------------------
# Analyse für jeden Ticker
# -----------------------------
def analyze_ticker(symbol):
    print(f"Analysiere {symbol}...")

    try:
        # Download
        df = yf.download(symbol, period="1y", interval="1d", progress=False)

        # 1. FIX FÜR YFINANCE MULTIINDEX PROBLEM
        # Wenn Spalten verschachtelt sind (z.B. ('Close', 'OMV.VI')), ebnen wir sie ein.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Check: Ist die Tabelle leer?
        if df is None or df.empty:
            print(f" -> Keine Daten erhalten für {symbol}")
            return

        if len(df) < 200:
            print(f" -> Zu wenig Daten für SMA200 bei {symbol} (Nur {len(df)} Tage)")
            return

        # 2. Indikatoren berechnen
        df["SMA_200"] = df["Close"].rolling(200).mean()
        df["RSI_14"] = compute_rsi(df["Close"], 14)

        # 3. Werte extrahieren (und sicherstellen, dass es wirklich Zahlen sind)
        # .item() zwingt Pandas, den Wert als reine Python-Zahl (float) zurückzugeben
        current_price = df["Close"].iloc[-1]
        if hasattr(current_price, "item"): current_price = current_price.item()

        sma200 = df["SMA_200"].iloc[-1]
        if hasattr(sma200, "item"): sma200 = sma200.item()

        rsi = df["RSI_14"].iloc[-1]
        if hasattr(rsi, "item"): rsi = rsi.item()

        # Falls RSI NaN ist (passiert manchmal am allerletzten Tag)
        if pd.isna(rsi) or pd.isna(sma200):
            print(f" -> Indikatoren nicht berechenbar (NaN) für {symbol}")
            return

        # 4. Score berechnen
        score = 0
        details = []

        # Trend: Preis über SMA200?
        if current_price > sma200:
            score += 0.5
            details.append("Bullish Trend (>SMA)")
        else:
            score -= 0.5
            details.append("Bearish Trend (<SMA)")

        # RSI
        if rsi < 30:
            score += 0.5
            details.append(f"RSI Oversold ({rsi:.0f})")
        elif rsi > 70:
            score -= 0.5
            details.append(f"RSI Overbought ({rsi:.0f})")
        else:
            details.append(f"RSI Neutral ({rsi:.0f})")

        final_score = max(min(score, 1.0), -1.0)

        print(f" -> Preis: {current_price:.2f} | Score: {final_score} | {', '.join(details)}")

        # 5. Speichern
        save_tech_signal(
            symbol,
            float(current_price),
            float(rsi),
            float(sma200),
            float(final_score),
            ", ".join(details)
        )

    except Exception as e:
        print(f"!! Kritischer Fehler bei {symbol}: {e}")


# -----------------------------
# Hauptloop
# -----------------------------
def run_tech_cycle():
    print("\n--- Starte Technische Analyse ---")
    for symbol in TICKER_MAP:
        analyze_ticker(symbol)
    print("--- Analyse fertig ---")

if __name__ == "__main__":
    init_tech_db()
    run_tech_cycle()