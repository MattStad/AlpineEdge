"""
pair_scan_cointegration.py

Scant eine Liste von Aktien-/ETF-Paaren und testet für jedes Paar
die Cointegration (Engle-Granger-Test). Gibt eine sortierte Tabelle
mit p-Werten aus (kleiner = besser).

Verwendung:
    python pair_scan_cointegration.py
"""

from datetime import datetime, date
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint


# =========================
# 1. Hilfsfunktionen
# =========================

def load_pair_prices(
    ticker1: str,
    ticker2: str,
    start: str | date,
    end: str | date,
) -> pd.DataFrame:
    """
    Lädt Daily-Schlusskurse für zwei Ticker und gibt ein DataFrame
    mit zwei Spalten zurück: [ticker1, ticker2].
    """

    def _load_one(ticker: str) -> pd.Series:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            progress=False,
        )
        if df.empty:
            raise ValueError(f"Keine Daten für {ticker} im Zeitraum {start} bis {end}.")
        # MultiIndex-Fall abfangen
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"].iloc[:, 0]
        else:
            close = df["Close"]
        close.name = ticker
        close.index = pd.to_datetime(close.index)
        return close

    s1 = _load_one(ticker1)
    s2 = _load_one(ticker2)

    df = pd.concat([s1, s2], axis=1).dropna()
    return df


def test_cointegration(df: pd.DataFrame, t1: str, t2: str) -> float:
    """
    Engle-Granger Cointegration-Test.
    Gibt p-Wert zurück (p < 0.05 → signifikante Cointegration).
    """
    score, pvalue, _ = coint(df[t1], df[t2])
    return float(pvalue)


# =========================
# 2. Pair-Liste definieren
# =========================

def get_candidate_pairs() -> List[Tuple[str, str, str]]:
    """
    Gibt eine Liste von (ticker1, ticker2, beschreibung) zurück.
    Du kannst hier beliebig weitere Paare ergänzen.
    """
    pairs = [
        # Energie / Öl
        ("OMV.VI", "BNO",   "OMV vs Brent Oil ETF (BNO)"),
        ("OMV.VI", "OIH",   "OMV vs Oil Services ETF (OIH)"),
        ("OMV.VI", "XOP",   "OMV vs Oil & Gas ETF (XOP)"),

        # Stahl / Metalle
        ("VOE.VI", "MT.AS", "Voestalpine vs ArcelorMittal"),
        ("VOE.VI", "S32.L", "Voestalpine vs South32"),

        # Europäische Banken
        ("INGA.AS", "BNP.PA", "ING vs BNP Paribas"),
        ("BARC.L", "HSBA.L",  "Barclays vs HSBC"),
        ("DBK.DE", "CBK.DE",  "Deutsche Bank vs Commerzbank"),

        # Österreichische Banken vs große EU-Banken
        ("EBS.VI", "INGA.AS", "Erste Group vs ING"),
        ("RBI.VI", "INGA.AS", "Raiffeisen vs ING"),

        ("KO","PEP","Pepsi vs Coke")
    ]
    return pairs


# =========================
# 3. Scan-Logik
# =========================

def scan_pairs(
    start: str = "2010-01-01",
    end: str | date = datetime.today().date(),
):
    results = []

    for t1, t2, desc in get_candidate_pairs():
        print(f"Teste Paar: {t1} / {t2} – {desc}")
        try:
            prices = load_pair_prices(t1, t2, start, end)
            if len(prices) < 252:  # weniger als 1 Jahr Daten → unbrauchbar
                print(f"  → zu wenige Daten ({len(prices)} Punkte), überspringe.\n")
                continue

            pvalue = test_cointegration(prices, t1, t2)
            results.append({
                "ticker1": t1,
                "ticker2": t2,
                "beschreibung": desc,
                "datenpunkte": len(prices),
                "p_value": pvalue,
            })
            print(f"  → p-Wert: {pvalue:.4f}\n")

        except Exception as e:
            print(f"  ⚠️ Fehler bei {t1}/{t2}: {e}\n")

    if not results:
        print("Keine gültigen Paare gefunden.")
        return

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("p_value", ascending=True)

    print("\n================= Scan-Ergebnis (sortiert nach p-Wert) =================")
    print(res_df.to_string(index=False))
    print("========================================================================\n")

    print("Interpretation:")
    print("- p < 0.05  → starke Evidenz für Cointegration (sehr interessante Pair-Trading-Kandidaten)")
    print("- 0.05–0.10 → schwache bis moderate Evidenz (mit Vorsicht testen)")
    print("- > 0.10    → eher kein Cointegration-Pair, besser ignorieren.")


# =========================
# 4. Main
# =========================

if __name__ == "__main__":
    START = "2010-01-01"
    END = datetime.today().date()
    scan_pairs(start=START, end=END)
