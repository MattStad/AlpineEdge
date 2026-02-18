# Momentum + Saisonalität Strategie für ATX-Aktien

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # falls Probleme: "Agg" verwenden
import matplotlib.pyplot as plt
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ------------------------------
# Parameter
# ------------------------------
TICKERS = [
    "ANDR.VI", "BG.VI", "CAI.VI", "EBS.VI", "FLU.VI", "LNZ.VI", "MMK.VI",
    "OMV.VI", "POST.VI", "RBI.VI", "SBO.VI", "STR.VI", "TKA.VI", "UQA.VI",
    "VOE.VI", "VIG.VI", "VER.VI", "ATS.VI", "FLU.VI", "SPI.VI",
    "DOC.VI", "ROS.VI", "WIE.VI", "FQT.VI", "MARI.VI"
]
START = "2008-01-01"
END = "2025-12-01"
LOOKBACK_DAYS = 126  # ca. 6 Monate Momentum
TOP_N = 3  # nur Top-N Momentum-Aktien kaufen
INITIAL_CAPITAL = 10_000.0
SEASONALITY_MONTHS = [4, 5, 11, 12]  # April, Mai, Nov, Dez


# ------------------------------
# Daten herunterladen
# ------------------------------
def download_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    df = df.dropna(axis=1, how="any")
    return df


# ------------------------------
# Momentum berechnen (annualisiert)
# ------------------------------
def calculate_momentum(data, lookback):
    ret = data.pct_change(lookback)
    momentum = (1 + ret) ** (252 / lookback) - 1
    return momentum


# ------------------------------
# Monatlich investieren mit Momentum + Saisonalität
# ------------------------------
def backtest_strategy(data):
    monthly_data = data.resample("M").last()
    momentum = calculate_momentum(monthly_data, LOOKBACK_DAYS // 21)

    equity_curve = pd.Series(index=monthly_data.index, dtype=float)
    capital = INITIAL_CAPITAL
    equity_curve.iloc[0] = capital
    for i in range(1, len(monthly_data)):
        date = monthly_data.index[i]
        month = date.month

        # Saisonalität aktiv?
        if month not in SEASONALITY_MONTHS:
            equity_curve[date] = capital
            continue

        # Top N Momentum Aktien
        current_mom = momentum.iloc[i - 1]
        if current_mom.isna().all():
            equity_curve[date] = capital
            continue

        top = current_mom.dropna().sort_values(ascending=False).head(TOP_N).index
        prev_prices = monthly_data.iloc[i - 1][top]
        curr_prices = monthly_data.iloc[i][top]

        returns = (curr_prices / prev_prices - 1).mean()
        capital *= np.exp(np.log1p(returns))  # stabiler bei kleinen/negativen Werten
        equity_curve[date] = capital

    equity_curve.ffill(inplace=True)
    return equity_curve


# ------------------------------
# Auswertung & Plot
# ------------------------------
def analyze_performance(equity_curve):
    start = equity_curve.index[0]
    end = equity_curve.index[-1]
    total_return = equity_curve[-1] / equity_curve[0] - 1
    cagr = (equity_curve[-1] / equity_curve[0]) ** (1 / ((end - start).days / 365.25)) - 1
    daily_returns = equity_curve.pct_change().dropna()
    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else np.nan
    drawdown = equity_curve / equity_curve.cummax() - 1
    max_dd = drawdown.min()

    print("\n===== Performance: Momentum + Saisonalität (ATX) =====")
    print(f"Gesamtrendite: {total_return * 100:.2f} %")
    print(f"CAGR:          {cagr * 100:.2f} %")
    print(f"Sharpe:        {sharpe:.2f}")
    print(f"Max Drawdown:  {max_dd * 100:.2f} %")

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Momentum + Saisonstrategie")
    plt.title("Equity-Kurve: Momentum + Saisonalität (ATX-Titel)")
    plt.xlabel("Datum")
    plt.ylabel("Kontostand (relativ)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    data = download_data(TICKERS, START, END)
    equity = backtest_strategy(data)
    analyze_performance(equity)
