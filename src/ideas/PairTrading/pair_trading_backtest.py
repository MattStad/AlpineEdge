"""
pair_trading_backtest.py

Pair Trading / Statistical Arbitrage zwischen:
- Erste Group Bank AG (EBS.VI)
- Raiffeisen Bank International (RBI.VI)

Pipeline:
1. Daten laden (Daily Close)
2. Cointegration-Test (Engle-Granger)
3. Hedge-Ratio via OLS
4. Spread & z-Score berechnen
5. Trading-Regeln (Mean Reversion des Spreads)
6. Market-neutraler Backtest (long/short beide Legs)
7. Trade-Log & Performance-Kennzahlen
8. Plots: Equity Curve & Spread/z-Score
"""

from dataclasses import dataclass
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf

import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

import matplotlib
matplotlib.use("TkAgg")  # falls Probleme: "Agg" verwenden
import matplotlib.pyplot as plt


# =========================
# 1. Daten laden
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


# =========================
# 2. Cointegration & Hedge-Ratio
# =========================

def test_cointegration(df: pd.DataFrame, t1: str, t2: str) -> float:
    """
    Engle-Granger Cointegration-Test.
    Gibt p-Wert zurück (p < 0.05 → signifikante Cointegration).
    """
    score, pvalue, _ = coint(df[t1], df[t2])
    return pvalue


def compute_hedge_ratio(df: pd.DataFrame, t1: str, t2: str) -> float:
    """
    Schätzt Hedge-Ratio beta aus OLS-Regression:
    t1 ≈ alpha + beta * t2
    Spread = t1 - beta * t2
    """
    y = df[t1]
    X = sm.add_constant(df[t2])
    model = sm.OLS(y, X).fit()
    beta = model.params[t2]
    return float(beta)


def compute_spread_zscore(
    df: pd.DataFrame,
    t1: str,
    t2: str,
    beta: float,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Berechnet Spread = t1 - beta * t2 und z-Score des Spreads
    über ein Rolling-Fenster.
    """
    data = df.copy()
    data["spread"] = data[t1] - beta * data[t2]
    data["spread_mean"] = data["spread"].rolling(lookback).mean()
    data["spread_std"] = data["spread"].rolling(lookback).std()
    data["zscore"] = (data["spread"] - data["spread_mean"]) / data["spread_std"]
    return data


# =========================
# 3. Strategie-Konfiguration
# =========================

@dataclass
class PairTradingConfig:
    lookback: int = 60      # Rolling-Fenster für z-Score-Berechnung
    entry_z: float = 2.0    # Entry-Schwelle |z| > entry_z
    exit_z: float = 0.5     # Exit-Schwelle |z| < exit_z
    max_holding_days: int | None = None  # Optionales Maximum an Haltezeit


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    commission_per_leg: float = 2.0
    slippage_bps: float = 1.0         # 1 bp = 0.01 %
    allocation_per_trade: float = 1.0  # Anteil des Kapitals pro Pair-Trade (1.0 = full)


# =========================
# 4. Pair-Trading-Backtester
# =========================

class PairTradingBacktester:
    """
    Market-neutraler Pair-Trading-Backtester:
    - Long/Short beide Beine in gleicher Dollargröße
    - Entry:
        z > entry_z  → short t1, long t2
        z < -entry_z → long t1, short t2
    - Exit:
        |z| < exit_z oder max_holding_days überschritten
    - Trades werden zum Open des folgenden Tages ausgeführt
      auf Basis des z-Scores vom Vortag (kein Lookahead).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        t1: str,
        t2: str,
        beta: float,
        strat_conf: PairTradingConfig,
        bt_conf: BacktestConfig,
    ):
        self.data = data
        self.t1 = t1
        self.t2 = t2
        self.beta = beta
        self.strat_conf = strat_conf
        self.bt_conf = bt_conf

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self.data.dropna(subset=["zscore"]).copy()

        capital = self.bt_conf.initial_capital
        cash = capital

        position = 0  # 0 = keine Position, +1 = long t1 / short t2, -1 = short t1 / long t2
        shares1 = 0.0
        shares2 = 0.0
        entry_index = None
        entry_equity = None
        entry_z = None

        equity_curve: list[dict] = []
        trade_log: list[dict] = []

        slip_factor = 1 + self.bt_conf.slippage_bps / 10_000

        dates = df.index

        for i in range(1, len(df)):
            prev_row = df.iloc[i - 1]
            row = df.iloc[i]

            date_prev = dates[i - 1]
            date_curr = dates[i]

            z_prev = prev_row["zscore"]

            price1_open = float(row[self.t1])
            price2_open = float(row[self.t2])

            price1_close = float(row[self.t1])
            price2_close = float(row[self.t2])

            # ===== 1) Exit-Entscheidung =====
            holding_days = (date_curr - entry_index).days if (position != 0 and entry_index is not None) else 0

            exit_signal = False
            exit_reason = None

            if position != 0:
                if abs(z_prev) < self.strat_conf.exit_z:
                    exit_signal = True
                    exit_reason = "z_exit"
                elif self.strat_conf.max_holding_days is not None and holding_days >= self.strat_conf.max_holding_days:
                    exit_signal = True
                    exit_reason = "max_hold"

            if position != 0 and exit_signal:
                # Exit zum Open von heute
                if position == 1:
                    # long t1, short t2
                    # Long t1 verkaufen
                    cash += shares1 * (price1_open / slip_factor)
                    # Short t2 zurückkaufen
                    cash -= shares2 * (price2_open * slip_factor)
                elif position == -1:
                    # short t1, long t2
                    cash -= shares1 * (price1_open * slip_factor)  # short t1 zurückkaufen
                    cash += shares2 * (price2_open / slip_factor)  # long t2 verkaufen

                # Kommissionen
                cash -= 2 * self.bt_conf.commission_per_leg

                equity_after = cash
                pnl = equity_after - entry_equity
                ret = pnl / entry_equity if entry_equity != 0 else 0.0

                trade_log.append({
                    "entry_time": entry_index,
                    "exit_time": date_curr,
                    "direction": position,  # +1 = long t1/short t2; -1 = short t1/long t2
                    "entry_z": entry_z,
                    "exit_z": z_prev,
                    "pnl": pnl,
                    "return_pct": ret,
                    "holding_days": holding_days,
                    "exit_reason": exit_reason,
                })

                position = 0
                shares1 = shares2 = 0.0
                entry_index = None
                entry_equity = None
                entry_z = None

            # ===== 2) Entry-Entscheidung =====
            if position == 0:
                # Spread oben → t1 teuer, t2 billig → short t1, long t2
                if z_prev > self.strat_conf.entry_z:
                    # Dollar-Exposure pro Leg
                    leg_capital = cash * self.bt_conf.allocation_per_trade / 2.0
                    shares1 = leg_capital / (price1_open * slip_factor)
                    shares2 = leg_capital / (price2_open / slip_factor)

                    # Position aufbauen: short t1, long t2
                    # short t1: wir erhalten Cash
                    cash += shares1 * (price1_open * (1 - self.bt_conf.slippage_bps / 10_000))
                    # long t2: wir zahlen Cash
                    cash -= shares2 * (price2_open * slip_factor)

                    # Kommissionen
                    cash -= 2 * self.bt_conf.commission_per_leg

                    position = -1
                    entry_index = date_curr
                    entry_equity = cash
                    entry_z = z_prev

                # Spread unten → t1 billig, t2 teuer → long t1, short t2
                elif z_prev < -self.strat_conf.entry_z:
                    leg_capital = cash * self.bt_conf.allocation_per_trade / 2.0
                    shares1 = leg_capital / (price1_open / slip_factor)
                    shares2 = leg_capital / (price2_open * slip_factor)

                    # long t1
                    cash -= shares1 * (price1_open * slip_factor)
                    # short t2
                    cash += shares2 * (price2_open * (1 - self.bt_conf.slippage_bps / 10_000))

                    # Kommissionen
                    cash -= 2 * self.bt_conf.commission_per_leg

                    position = 1
                    entry_index = date_curr
                    entry_equity = cash
                    entry_z = z_prev

            # ===== 3) Equity berechnen (mark-to-market am Close) =====
            if position == 0:
                equity = cash
            elif position == 1:
                # long t1, short t2
                equity = cash + shares1 * price1_close - shares2 * price2_close
            else:
                # short t1, long t2
                equity = cash - shares1 * price1_close + shares2 * price2_close

            equity_curve.append({
                "timestamp": date_curr,
                "equity": equity,
                "cash": cash,
                "position": position,
                "shares1": shares1,
                "shares2": shares2,
                "zscore": z_prev,
            })

        equity_df = pd.DataFrame(equity_curve).set_index("timestamp")
        trades_df = pd.DataFrame(trade_log)
        return equity_df, trades_df


# =========================
# 5. Performance & Plots
# =========================

def compute_performance_stats(equity_df: pd.DataFrame) -> dict:
    equity = equity_df["equity"]
    total_return = equity.iloc[-1] / equity.iloc[0] - 1

    days = (equity.index[-1] - equity.index[0]).days
    if days > 0:
        daily_equity = equity.resample("1D").last().dropna()
        daily_returns = daily_equity.pct_change().dropna()
        annual_return = (1 + daily_returns.mean()) ** 252 - 1
        annual_vol = daily_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
    else:
        annual_return = np.nan
        sharpe = np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


def plot_equity_curve(equity_df: pd.DataFrame, title: str):
    plt.figure(figsize=(12, 5))
    plt.plot(equity_df.index, equity_df["equity"], label="Pair Strategy", linewidth=2)
    plt.title(title)
    plt.xlabel("Zeit")
    plt.ylabel("Kontowert (EUR)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_spread_zscore(data: pd.DataFrame, title: str, t1: str, t2: str):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(data.index, data[t1], label=t1)
    axes[0].plot(data.index, data[t2], label=t2)
    axes[0].set_title(f"Preise: {t1} & {t2}")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(data.index, data["spread"], label="Spread")
    axes[1].plot(data.index, data["spread_mean"], label="Spread Mean", linestyle="--")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Spread & Rolling-Mean")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    # z-Score separat
    plt.figure(figsize=(12, 4))
    plt.plot(data.index, data["zscore"], label="z-Score")
    plt.axhline(0, color="black", linewidth=1)
    plt.axhline(2, color="red", linestyle="--")
    plt.axhline(-2, color="red", linestyle="--")
    plt.axhline(0.5, color="grey", linestyle=":")
    plt.axhline(-0.5, color="grey", linestyle=":")
    plt.title("z-Score des Spreads")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# 6. Main
# =========================

if __name__ == "__main__":
    T1 = "EBS.VI"
    T2 = "STR.VI"
    START = "2010-01-01"
    END = datetime.today().date().isoformat()

    print(f"Lade Daten für {T1} und {T2} ...")
    prices = load_pair_prices(T1, T2, START, END)
    print(prices.tail())

    # 1) Cointegration-Test
    pvalue = test_cointegration(prices, T1, T2)
    print(f"\nEngle-Granger-Cointegrationstest p-Wert: {pvalue:.4f}")
    if pvalue >= 0.05:
        print("⚠️ Warnung: p >= 0.05 → keine starke Evidenz für Cointegration.")

    # 2) Hedge-Ratio
    beta = compute_hedge_ratio(prices, T1, T2)
    print(f"Hedge-Ratio (beta) geschätzt: {beta:.4f}")

    # 3) Spread & z-Score
    strat_conf = PairTradingConfig(
        lookback=60,
        entry_z=2.0,
        exit_z=0.5,
        max_holding_days=30,  # z.B. nach 30 Tagen zwangsweise schließen
    )
    data = compute_spread_zscore(prices, T1, T2, beta, lookback=strat_conf.lookback)

    # 4) Backtest
    bt_conf = BacktestConfig(
        initial_capital=10_000.0,
        commission_per_leg=2.0,
        slippage_bps=1.0,
        allocation_per_trade=1.0,
    )

    backtester = PairTradingBacktester(
        data=data,
        t1=T1,
        t2=T2,
        beta=beta,
        strat_conf=strat_conf,
        bt_conf=bt_conf,
    )

    equity_df, trades_df = backtester.run()

    # 5) Performance
    stats = compute_performance_stats(equity_df)
    print("\n===== Pair-Trading-Resultate =====")
    print(f"Startkapital:     {bt_conf.initial_capital:,.2f} EUR")
    print(f"Endkapital:       {equity_df['equity'].iloc[-1]:,.2f} EUR")
    print(f"Gesamtrendite:    {stats['total_return']*100:,.2f} %")
    print(f"Annualisierte R.: {stats['annual_return']*100:,.2f} %" if not np.isnan(stats['annual_return']) else "Annualisierte R.: n/a")
    print(f"Max Drawdown:     {stats['max_drawdown']*100:,.2f} %")
    print(f"Sharpe Ratio:     {stats['sharpe']:.2f}" if not np.isnan(stats['sharpe']) else "Sharpe Ratio: n/a")
    print(f"Anzahl Trades:    {len(trades_df)}")

    if not trades_df.empty:
        print("\nErste Trades:")
        print(trades_df.head())
        print("\nDurchschnittliche Trade-Return: "
              f"{100 * trades_df['return_pct'].mean():.2f} %")
        print("Win-Rate: "
              f"{100 * (trades_df['pnl'] > 0).mean():.2f} %")

    # 6) Plots
    plot_equity_curve(
        equity_df,
        title=f"Pair Trading Equity – {T1} vs {T2}",
    )

    plot_spread_zscore(
        data,
        title=f"Spread & z-Score – {T1} vs {T2}",
        t1=T1,
        t2=T2,
    )
