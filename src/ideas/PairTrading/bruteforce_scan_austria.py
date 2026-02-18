import itertools
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
import numpy as np
# ========================
# 1. Liste der österreichischen Aktien (ATX + weitere Prime Market)
# ========================

tickers = [
    "ANDR.VI", "BG.VI", "CAI.VI", "EBS.VI", "FLU.VI", "LNZ.VI", "MMK.VI",
    "OMV.VI", "POST.VI", "RBI.VI", "SBO.VI", "STR.VI", "TKA.VI", "UQA.VI",
    "VOE.VI", "VIG.VI", "VER.VI", "ATS.VI", "FLU.VI", "SPI.VI",
    "DOC.VI", "ROS.VI", "WIE.VI", "FQT.VI", "MARI.VI",
]

START = "2010-01-01"
END = "2025-12-31"

# ========================
# 2. Historische Kursdaten laden
# ========================

def load_close_series(ticker: str) -> pd.Series:
    df = yf.download(ticker, start=START, end=END, progress=False)
    if df.empty:
        raise ValueError(f"Keine Daten für {ticker}")
    df = df.rename(columns=str.lower)
    s = df["close"].dropna()
    s.name = ticker
    return s

price_data: dict[str, pd.Series] = {}

print("Lade Kursdaten...")

for t in tickers:
    try:
        s = load_close_series(t)
        print(f"{t}: {len(s)} Datenpunkte")
        # nur minimaler Filter: mindestens 252 Tage (1 Handelsjahr)
        if len(s) >= 252:
            price_data[t] = s
        else:
            print(f"  -> zu kurz, wird nicht verwendet.")
    except Exception as e:
        print(f"{t}: Fehler beim Laden ({e})")

print(f"\n{len(price_data)} von {len(tickers)} Aktien erfolgreich mit ausreichender Länge geladen.\n")

# ========================
# 3. Bruteforce Cointegration Test
# ========================

results = []
pairs = list(itertools.combinations(sorted(price_data.keys()), 2))

print(f"Teste {len(pairs)} Paare...\n")

for t1, t2 in pairs:
    s1_raw = price_data[t1].astype(float)
    s2_raw = price_data[t2].astype(float)

    # gemeinsame Daten, NaNs raus
    df_pair = pd.concat([s1_raw, s2_raw], axis=1, keys=[t1, t2]).dropna()

    n = len(df_pair)
    if n < 252:
        print(f"{t1} / {t2}: nur {n} gemeinsame Punkte, übersprungen.")
        continue

    # als numpy arrays, nur endliche Werte
    x = df_pair[t1].to_numpy()
    y = df_pair[t2].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 252:
        print(f"{t1} / {t2}: nach Cleaning nur {len(x)} Punkte, übersprungen.")
        continue

    try:
        score, pvalue, _ = coint(x, y)
        results.append((t1, t2, float(pvalue), len(x)))
        print(f"{t1} / {t2}: p = {pvalue:.4f}, n = {len(x)}")
    except Exception as e:
        print(f"{t1} / {t2}: Fehler im Cointegrationstest ({e})")
# ========================
# 4. Sortieren & Ausgabe
# ========================

if not results:
    print("\n⚠️ Keine gültigen Cointegration-Resultate – entweder alle Paare wurden gefiltert oder es gab Fehler.")
else:
    df_res = pd.DataFrame(results, columns=["ticker1", "ticker2", "p_value", "length"])
    df_res = df_res.sort_values("p_value")

    print("\n============ Top 15 beste Paare (niedrigster p-Wert) ============\n")
    print(df_res.head(15).to_string(index=False))
    print("\n=================================================================\n")
    print("Interpretation:")
    print("- p < 0.05 → stark cointegrated → sehr gut für Pair Trading")
    print("- 0.05–0.10 → akzeptabel, aber vorsichtig")
    print("- > 0.10 → keine Cointegration (für stat. Arbitrage eher uninteressant)")
