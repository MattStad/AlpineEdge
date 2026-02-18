"""
Multi-Faktor-Strategie für österreichische Aktien:
Trend (SMA200) + Volatilität + Mean Reversion (Z-Score um SMA20).

- Nur Long.
- Entry: Aufwärtstrend, "gesunde" Volatilität, starker Dip (z < -1).
- Exit: Mean-Reversion (z >= -0.2) oder Trendbruch (Close < SMA200) oder Stop-Loss.

Kapital:
- INITIAL_CAPITAL wird gleichmäßig auf alle Titel verteilt.
- Pro Titel Risk-based Position Sizing mit ATR.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # falls Probleme: "Agg" verwenden
import matplotlib.pyplot as plt


# ---------- Universum & Parameter ----------

TICKERS = [
    "OMV.VI", "VOE.VI", "EBS.VI", "RBI.VI", "VIG.VI",
    "STR.VI", "ANDR.VI", "VER.VI", "POST.VI", "WIE.VI", "TKA.VI",
]

START = "2008-01-01"
END = "2025-01-01"

INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.02       # 2% Risiko pro Titel
STOP_ATR_MULT = 2.0         # Stop-Distanz: 2 * ATR
VOL_MIN = 0.005             # 0.5% ATR/Preis
VOL_MAX = 0.04              # 4% ATR/Preis
Z_ENTRY = -1.0              # starker Dip
Z_EXIT = -0.2               # Rückkehr Richtung SMA20


# ---------- 1. Daten laden ----------

def download_data() -> dict:
    data = {}
    for t in TICKERS:
        print(f"Lade {t}...")
        df = yf.download(t, start=START, end=END, progress=False)

        if df is None or df.empty:
            print(f"⚠️ {t}: keine Daten, übersprungen.")
            continue

        # MultiIndex flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].title() for c in df.columns]
        else:
            df = df.rename(columns=str.title)

        needed = {"Close", "High", "Low"}
        if not needed.issubset(set(df.columns)):
            print(f"⚠️ {t}: falsche Spalten ({df.columns}), übersprungen.")
            continue

        df.index = pd.to_datetime(df.index)
        data[t] = df

    if not data:
        raise ValueError("Keine Kursdaten für irgendeinen Ticker geladen.")
    print(f"\n✅ {len(data)} von {len(TICKERS)} Tickern erfolgreich geladen.\n")
    return data


# ---------- 2. Faktoren / Signale ----------

def add_factors_and_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # ATR20
    tr = (high - low).abs()
    atr20 = tr.rolling(20, min_periods=20).mean()
    df["ATR"] = atr20

    # SMA200 & SMA20
    df["SMA200"] = close.rolling(200, min_periods=200).mean()
    df["SMA20"] = close.rolling(20, min_periods=20).mean()

    # Std20 und Z-Score
    std20 = close.rolling(20, min_periods=20).std()
    df["STD20"] = std20
    df["Z"] = (close - df["SMA20"]) / df["STD20"]

    # Normalisierte Volatilität
    df["ATR_NORM"] = df["ATR"] / close

    # Trend-Faktor
    df["trend_up"] = close > df["SMA200"]

    # Volatilitäts-Filter
    df["vol_ok"] = (df["ATR_NORM"] >= VOL_MIN) & (df["ATR_NORM"] <= VOL_MAX)

    # Mean-Reversion Entry / Exit
    df["entry_signal"] = df["trend_up"] & df["vol_ok"] & (df["Z"] < Z_ENTRY)
    df["exit_signal"] = (df["Z"] >= Z_EXIT) | (close < df["SMA200"])

    # Nur Zeilen, wo alles definiert ist
    df = df.dropna(subset=["ATR", "SMA200", "SMA20", "STD20", "ATR_NORM"])

    return df


# ---------- 3. Backtest für EINEN Titel (Multi-Faktor) ----------

def backtest_single_multi_factor(df: pd.DataFrame,
                                 initial_capital: float) -> pd.Series:
    df = add_factors_and_signals(df)

    if df.empty:
        print("⚠️ backtest_single_multi_factor: keine Daten nach Signal-Generierung.")
        return pd.Series(dtype=float)

    equity = initial_capital
    position = 0.0
    entry_price = 0.0
    stop_price = np.nan

    equity_list = []

    for dt, row in df.iterrows():
        price = float(row["Close"])
        atr = float(row["ATR"])
        entry_sig = bool(row["entry_signal"])
        exit_sig = bool(row["exit_signal"])

        # --- EXIT-Logik ---
        if position > 0:
            # Stop-Loss
            if (price <= stop_price) or exit_sig:
                pnl = (price - entry_price) * position
                equity += pnl
                position = 0.0
                entry_price = 0.0
                stop_price = np.nan

        # --- ENTRY-Logik ---
        if position == 0 and entry_sig and atr > 0:
            risk_amount = equity * RISK_PER_TRADE
            stop_distance = STOP_ATR_MULT * atr
            if stop_distance > 0:
                size = risk_amount / stop_distance
                if size > 0:
                    position = size
                    entry_price = price
                    stop_price = entry_price - stop_distance

        # --- Mark-to-market Equity ---
        if position > 0:
            equity_marked = equity + (price - entry_price) * position
        else:
            equity_marked = equity

        equity_list.append((dt, equity_marked))

    if not equity_list:
        return pd.Series(dtype=float)

    equity_series = pd.Series(
        [e for _, e in equity_list],
        index=[d for d, _ in equity_list],
        name="equity_multi_factor",
    )
    return equity_series


# ---------- 4. Portfolio-Backtest ----------

def backtest_portfolio_multi_factor(data: dict) -> pd.Series:
    valid_equities = {}
    n = len(data)
    capital_per_ticker = INITIAL_CAPITAL / n

    for t, df in data.items():
        print(f"Backteste Multi-Faktor für {t} ...")
        s = backtest_single_multi_factor(df, capital_per_ticker)
        if s.empty:
            print(f"⚠️ {t}: keine gültige Equity-Kurve, übersprungen.")
            continue
        valid_equities[t] = s

    if not valid_equities:
        raise ValueError("Keine Equity-Kurve im Multi-Faktor-Backtest.")

    eq_df = pd.DataFrame(valid_equities)
    portfolio_equity = eq_df.sum(axis=1)
    portfolio_equity.name = "MultiFactor_ATX"
    return portfolio_equity.sort_index().dropna()


# ---------- 5. Kennzahlen ----------

def compute_stats(equity: pd.Series) -> dict:
    equity = equity.dropna()
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    total_return = end / start - 1

    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    cagr = (1 + total_return) ** (1 / years) - 1 if years and years > 0 else np.nan

    daily_ret = equity.pct_change().dropna()
    if len(daily_ret) > 2 and daily_ret.std() > 0:
        sharpe = np.sqrt(252) * daily_ret.mean() / daily_ret.std()
    else:
        sharpe = np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    return {
        "start": start,
        "end": end,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


# ---------- 6. Main ----------

if __name__ == "__main__":
    data = download_data()

    portfolio_equity = backtest_portfolio_multi_factor(data)

    stats = compute_stats(portfolio_equity)

    print("\n===== Resultate – Multi-Faktor Österreich =====")
    print(f"Startkapital (gesamt): {stats['start']:,.2f} EUR")
    print(f"Endkapital (gesamt):   {stats['end']:,.2f} EUR")
    print(f"Gesamtrendite:         {stats['total_return']*100:,.2f} %")
    print(f"CAGR:                  {stats['cagr']*100:,.2f} %")
    print(f"Sharpe (ungefähr):     {stats['sharpe']:,.2f}")
    print(f"Max Drawdown:          {stats['max_drawdown']*100:,.2f} %")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_equity.index, portfolio_equity.values, label="Multi-Faktor-Portfolio")
    plt.title("Multi-Faktor-Strategie – Trend + Volatilität + Mean Reversion (ATX-Universum)")
    plt.xlabel("Zeit")
    plt.ylabel("Kontowert (EUR)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
