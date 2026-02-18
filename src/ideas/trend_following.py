"""
Vergleich von 3 Trend-Following-Strategien auf österreichischen Aktien:

1) SMA200 + fixer 2*ATR-Stop (Baseline)
2) SMA200 + ATR-Trailing-Stop
3) Donchian 55/20 (Turtle-Style) + ATR-Stop

Kapital:
- INITIAL_CAPITAL wird gleichmäßig auf alle Titel verteilt.
- Pro Titel wird mit ATR-basiertem Risiko RISK_PER_TRADE gearbeitet.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # falls Probleme: "Agg" verwenden
import matplotlib.pyplot as plt


TICKERS = [
    "OMV.VI", "VOE.VI", "EBS.VI", "RBI.VI", "VIG.VI",
    "STR.VI", "ANDR.VI", "VER.VI", "POST.VI", "WIE.VI", "TKA.VI",
]

START = "2008-01-01"
END = "2025-01-01"
INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.02  # 2 % Risiko pro Titel


# ----------------------------------------------------------------------
# Daten laden (robust)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Hilfsfunktionen für Indikatoren
# ----------------------------------------------------------------------
def add_atr(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    df = df.copy()
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    tr = (high - low).abs()
    df["ATR"] = tr.rolling(length, min_periods=length).mean()
    return df


def add_sma200_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"].astype(float)
    df["SMA200"] = close.rolling(200, min_periods=200).mean()
    df["long_signal"] = close > df["SMA200"]
    df["exit_signal"] = close < df["SMA200"]
    df = df.dropna(subset=["SMA200", "ATR"])
    return df


def add_donchian_signals(df: pd.DataFrame,
                          period_entry: int = 55,
                          period_exit: int = 20) -> pd.DataFrame:
    df = df.copy()
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    df["DC_high"] = high.rolling(period_entry, min_periods=period_entry).max()
    df["DC_low"] = low.rolling(period_exit, min_periods=period_exit).min()

    df["long_signal"] = close > df["DC_high"]      # Entry
    df["exit_signal"] = close < df["DC_low"]       # Exit
    df = df.dropna(subset=["DC_high", "DC_low", "ATR"])
    return df


# ----------------------------------------------------------------------
# 3 Backtest-Varianten für EINEN Titel
# ----------------------------------------------------------------------
def backtest_baseline(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    """SMA200 + fixer 2*ATR-Stop."""
    df = add_atr(df)
    df = add_sma200_signals(df)
    if df.empty:
        return pd.Series(dtype=float)

    equity = initial_capital
    position = 0.0
    entry_price = 0.0
    stop_price = np.nan

    equity_list = []

    for dt, row in df.iterrows():
        price = float(row["Close"])
        atr = float(row["ATR"])
        long_sig = bool(row["long_signal"])
        exit_sig = bool(row["exit_signal"])

        # EXIT
        if position > 0:
            if (price <= stop_price) or exit_sig:
                pnl = (price - entry_price) * position
                equity += pnl
                position = 0.0
                entry_price = 0.0
                stop_price = np.nan

        # ENTRY
        if position == 0 and long_sig and atr > 0:
            risk_amount = equity * RISK_PER_TRADE
            size = risk_amount / (2.0 * atr)  # Stop-Entfernung = 2*ATR
            if size > 0:
                position = size
                entry_price = price
                stop_price = entry_price - 2.0 * atr

        # Mark-to-market
        if position > 0:
            equity_marked = equity + (price - entry_price) * position
        else:
            equity_marked = equity

        equity_list.append((dt, equity_marked))

    if not equity_list:
        return pd.Series(dtype=float)
    return pd.Series(
        [e for _, e in equity_list],
        index=[d for d, _ in equity_list],
        name="equity_baseline",
    )


def backtest_trailing(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    """SMA200 + ATR-Trailing-Stop."""
    df = add_atr(df)
    df = add_sma200_signals(df)
    if df.empty:
        return pd.Series(dtype=float)

    equity = initial_capital
    position = 0.0
    entry_price = 0.0
    stop_price = np.nan
    highest_close_since_entry = np.nan

    equity_list = []

    for dt, row in df.iterrows():
        price = float(row["Close"])
        atr = float(row["ATR"])
        long_sig = bool(row["long_signal"])
        exit_sig = bool(row["exit_signal"])

        # EXIT
        if position > 0:
            # Trailing-Stop nachziehen
            highest_close_since_entry = max(highest_close_since_entry, price)
            trailing_stop = highest_close_since_entry - 3.0 * atr
            stop_price = max(stop_price, trailing_stop)

            if (price <= stop_price) or exit_sig:
                pnl = (price - entry_price) * position
                equity += pnl
                position = 0.0
                entry_price = 0.0
                stop_price = np.nan
                highest_close_since_entry = np.nan

        # ENTRY
        if position == 0 and long_sig and atr > 0:
            risk_amount = equity * RISK_PER_TRADE
            size = risk_amount / (3.0 * atr)  # initial Stop ~3*ATR
            if size > 0:
                position = size
                entry_price = price
                stop_price = entry_price - 3.0 * atr
                highest_close_since_entry = price

        # Mark-to-market
        if position > 0:
            equity_marked = equity + (price - entry_price) * position
        else:
            equity_marked = equity

        equity_list.append((dt, equity_marked))

    if not equity_list:
        return pd.Series(dtype=float)
    return pd.Series(
        [e for _, e in equity_list],
        index=[d for d, _ in equity_list],
        name="equity_trailing",
    )


def backtest_donchian(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    """Donchian 55/20 + ATR-Stop."""
    df = add_atr(df)
    df = add_donchian_signals(df, period_entry=55, period_exit=20)
    if df.empty:
        return pd.Series(dtype=float)

    equity = initial_capital
    position = 0.0
    entry_price = 0.0
    stop_price = np.nan

    equity_list = []

    for dt, row in df.iterrows():
        price = float(row["Close"])
        atr = float(row["ATR"])
        long_sig = bool(row["long_signal"])
        exit_sig = bool(row["exit_signal"])

        # EXIT
        if position > 0:
            if (price <= stop_price) or exit_sig:
                pnl = (price - entry_price) * position
                equity += pnl
                position = 0.0
                entry_price = 0.0
                stop_price = np.nan

        # ENTRY
        if position == 0 and long_sig and atr > 0:
            risk_amount = equity * RISK_PER_TRADE
            size = risk_amount / (2.0 * atr)
            if size > 0:
                position = size
                entry_price = price
                stop_price = entry_price - 2.0 * atr

        # Mark-to-market
        if position > 0:
            equity_marked = equity + (price - entry_price) * position
        else:
            equity_marked = equity

        equity_list.append((dt, equity_marked))

    if not equity_list:
        return pd.Series(dtype=float)
    return pd.Series(
        [e for _, e in equity_list],
        index=[d for d, _ in equity_list],
        name="equity_donchian",
    )


# ----------------------------------------------------------------------
# Portfolio-Backtest: selbe Strategie über alle Titel
# ----------------------------------------------------------------------
def backtest_portfolio(data: dict, bt_func) -> pd.Series:
    valid_equities = {}
    n = len(data)
    capital_per_ticker = INITIAL_CAPITAL / n

    for t, df in data.items():
        print(f"Backteste {t} ... ({bt_func.__name__})")
        s = bt_func(df, capital_per_ticker)
        if s.empty:
            print(f"⚠️ {t}: keine gültige Equity-Kurve ({bt_func.__name__}), übersprungen.")
            continue
        valid_equities[t] = s

    if not valid_equities:
        raise ValueError(f"Keine valide Equity-Kurve für {bt_func.__name__}")

    eq_df = pd.DataFrame(valid_equities)
    portfolio_equity = eq_df.sum(axis=1)
    portfolio_equity.name = bt_func.__name__
    return portfolio_equity.sort_index().dropna()


# ----------------------------------------------------------------------
# Kennzahlen
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    data = download_data()

    # Drei Strategien
    eq_baseline = backtest_portfolio(data, backtest_baseline)
    eq_trailing = backtest_portfolio(data, backtest_trailing)
    eq_donchian = backtest_portfolio(data, backtest_donchian)

    # Gemeinsame Zeitachse
    all_eq = pd.concat(
        [eq_baseline.rename("Baseline_SMA200"),
         eq_trailing.rename("Trailing_SMA200"),
         eq_donchian.rename("Donchian_55_20")],
        axis=1
    ).dropna()

    # Stats
    print("\n===== Vergleich der Strategien =====")
    for col in all_eq.columns:
        stats = compute_stats(all_eq[col])
        print(f"\n>> {col}")
        print(f"  Startkapital:  {stats['start']:,.2f} EUR")
        print(f"  Endkapital:    {stats['end']:,.2f} EUR")
        print(f"  Gesamtrendite: {stats['total_return']*100:,.2f} %")
        print(f"  CAGR:          {stats['cagr']*100:,.2f} %")
        print(f"  Sharpe:        {stats['sharpe']:,.2f}")
        print(f"  Max Drawdown:  {stats['max_drawdown']*100:,.2f} %")

    # Plot
    plt.figure(figsize=(12, 6))
    for col in all_eq.columns:
        plt.plot(all_eq.index, all_eq[col], label=col)
    plt.title("Trend-Following – Vergleich Baseline / Trailing / Donchian")
    plt.xlabel("Zeit")
    plt.ylabel("Kontowert (EUR)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
