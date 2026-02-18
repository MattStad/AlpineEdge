"""
intraday_mean_reversion_backtest.py

Mean-Reversion Backtest für eine ATX-Aktie (z.B. EBS.VI = Erste Group).
Strategie-Idee (aktuell auf Tagesdaten):
- Wir berechnen einen z-Score des Close-Preises relativ zu einem gleitenden Durchschnitt.
- Wenn der Preis stark unter dem Durchschnitt liegt (z < -z_entry), gehen wir long.
- Wenn der z-Score wieder Richtung 0 geht (|z| < z_exit), gehen wir aus der Position.
- Zusätzlich gibt es ein Stop-Loss pro Trade (risk management).
- Am Ende plotten wir die Strategie gegen S&P500 und ATX.

Du kannst später easy auf Intraday-Bars umstellen (Interval="5m" und kürzerer Zeitraum).
"""

from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass

import matplotlib
matplotlib.use("TkAgg")  # falls Probleme → "Agg" verwenden
import matplotlib.pyplot as plt


# =========================
# 1. Daten laden
# =========================

def download_intraday_data(ticker: str,
                           start,
                           end,
                           interval: str = "5m") -> pd.DataFrame:
    """
    Lädt (Intra-)Tagesdaten von Yahoo Finance.
    Für ATX-Aktien z.B.: "EBS.VI", "RBI.VI", "VOE.VI", "OMV.VI".
    Für Indizes z.B.: "^GSPC", "^ATX".
    start / end können Strings oder date-Objekte sein.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        progress=False
    )

    if df.empty:
        raise ValueError(
            f"Keine Daten für {ticker} im Zeitraum {start} bis {end} mit Interval {interval}."
        )

    # MultiIndex-Columns (z.B. ('Close','EBS.VI')) auf eine Ebene flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df = df.rename(columns=str.lower)

    # Sicherstellen, dass Index ein DatetimeIndex ist
    df.index = pd.to_datetime(df.index)

    # Nur relevante Spalten
    df = df[["open", "high", "low", "close", "volume"]]

    return df


# =========================
# 2. Strategie: Mean Reversion
# =========================

@dataclass
class MeanReversionConfig:
    lookback: int = 20          # Anzahl Bars für Rolling-Mean/Std
    z_entry: float = 1.5        # z-Score-Schwelle für Entry
    z_exit: float = 0.3         # z-Score-Schwelle für Exit (Richtung 0)


class MeanReversionIntraday:
    def __init__(self, config: MeanReversionConfig):
        self.config = config

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet z-Score und generiert eine 0/1-Signalkurve:
        1 = long sein, 0 = flat.
        """
        data = df.copy()

        # Sicherstellen, dass 'close' wirklich eine Series ist
        close = pd.Series(data["close"], index=data.index, dtype=float)

        # Rolling Mean/Std nur auf der Close-Series
        ma = close.rolling(self.config.lookback,
                           min_periods=self.config.lookback).mean()
        std = close.rolling(self.config.lookback,
                            min_periods=self.config.lookback).std()

        # Optional ultra-konservativ gegen Lookahead:
        # ma = ma.shift(1)
        # std = std.shift(1)

        data["ma"] = ma
        data["std"] = std

        # z-Score berechnen – garantiert eine 1D-Series
        zscore = (close - ma) / std
        if isinstance(zscore, pd.DataFrame):
            zscore = zscore.iloc[:, 0]
        data["zscore"] = zscore

        # Signale initialisieren
        data["signal"] = 0

        long_condition = data["zscore"] < -self.config.z_entry
        exit_condition = data["zscore"].abs() < self.config.z_exit

        position = 0
        signals = []
        prev_day = None

        for ts, row in data.iterrows():
            current_day = ts.date()

            # Für echte Intraday-Daten könntest du hier am Tagesanfang flatten
            if prev_day is not None and current_day != prev_day:
                # position = 0   # für striktes Intraday-Only
                pass
            prev_day = current_day

            if position == 0:
                if bool(long_condition.loc[ts]):
                    position = 1
            else:
                if bool(exit_condition.loc[ts]):
                    position = 0

            signals.append(position)

        data["signal"] = signals
        return data


# =========================
# 3. Backtester + Risk Management
# =========================

@dataclass
class BacktestConfig:
    initial_cash: float = 10_000.0
    commission_per_trade: float = 3.0    # fixe Gebühren pro Trade (Hin/Her je 3€)
    slippage_bps: float = 1.0            # Slippage in Basispunkten (1 bp = 0,01 %)
    stop_loss_pct: float = 0.08          # 8 % Stop-Loss pro Trade (vom Entrypreis)


class IntradayBacktester:
    """
    Einfacher Backtester mit:
    - Full-In / Full-Out Positionierung (maximale Positionsgröße).
    - Nur Long (Position >= 0).
    - Signal-basierte Entries/Exits (am Open nach dem Signal).
    - Stop-Loss pro Trade (wenn Low unter Stop-Loss-Preis fällt).
    - Trade-Log.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(self, data_with_signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = data_with_signals.copy()

        cash = self.config.initial_cash
        position = 0.0   # Anzahl Aktien
        equity_curve = []
        trades = 0

        # Variablen für aktuell offenen Trade (falls vorhanden)
        open_trade = None  # dict mit entry_time, entry_price, size, etc.
        stop_price = None  # Stop-Loss-Level pro Trade

        trade_log: list[dict] = []

        # Wir brauchen den Signal-Wert der VORIGEN Kerze, um am aktuellen Open zu handeln
        for i in range(1, len(df)):
            prev_row = df.iloc[i - 1]
            row = df.iloc[i]

            price_open = float(row["open"])
            price_low = float(row["low"])
            price_close = float(row["close"])
            signal_prev = int(prev_row["signal"])
            timestamp = row.name
            day = timestamp.date()

            slip_factor = 1 + (self.config.slippage_bps / 10_000)

            # ===== 1) Risk Management: Stop-Loss pro Trade =====
            if position > 0 and open_trade is not None and stop_price is not None:
                # Falls das Tagestief unter den Stop-Loss fällt, exit zum Stop-Preis
                if price_low <= stop_price:
                    exit_price = stop_price
                    cash += position * exit_price
                    cash -= self.config.commission_per_trade

                    trades += 1

                    # Trade-Log updaten
                    entry_price = open_trade["entry_price"]
                    size = open_trade["size"]
                    pnl = (exit_price - entry_price) * size - 2 * self.config.commission_per_trade
                    ret = pnl / (entry_price * size)

                    open_trade["exit_time"] = timestamp
                    open_trade["exit_price"] = exit_price
                    open_trade["pnl"] = pnl
                    open_trade["return_pct"] = ret
                    open_trade["exit_reason"] = "stop_loss"
                    trade_log.append(open_trade)

                    # Position schließen
                    position = 0.0
                    open_trade = None
                    stop_price = None

            # ===== 2) Signal-basierte Entries/Exits (am Open) =====
            # Nur wenn nach Stop-Loss ggf. noch eine Position existiert / Flat ist
            if position == 0 and signal_prev == 1:
                # Neuer Entry
                trade_price = price_open * slip_factor
                size = (cash - self.config.commission_per_trade) / trade_price
                if size > 0:
                    cash -= size * trade_price
                    cash -= self.config.commission_per_trade
                    position = size
                    trades += 1

                    # Trade-Info speichern
                    open_trade = {
                        "entry_time": timestamp,
                        "entry_price": trade_price,
                        "size": size,
                        "exit_time": None,
                        "exit_price": None,
                        "pnl": None,
                        "return_pct": None,
                        "exit_reason": None,
                    }
                    stop_price = trade_price * (1.0 - self.config.stop_loss_pct)

            elif position > 0 and signal_prev == 0 and open_trade is not None:
                # Exit per Strategie-Signal
                trade_price = price_open / slip_factor
                cash += position * trade_price
                cash -= self.config.commission_per_trade

                trades += 1

                entry_price = open_trade["entry_price"]
                size = open_trade["size"]
                pnl = (trade_price - entry_price) * size - 2 * self.config.commission_per_trade
                ret = pnl / (entry_price * size)

                open_trade["exit_time"] = timestamp
                open_trade["exit_price"] = trade_price
                open_trade["pnl"] = pnl
                open_trade["return_pct"] = ret
                open_trade["exit_reason"] = "signal_exit"
                trade_log.append(open_trade)

                position = 0.0
                open_trade = None
                stop_price = None

            # ===== 3) Equity berechnen (Bewertung am Close) =====
            equity = cash + position * price_close
            equity_curve.append(
                {
                    "timestamp": timestamp,
                    "equity": equity,
                    "cash": cash,
                    "position_shares": position,
                    "close": price_close,
                    "signal": signal_prev,
                    "day": day,
                }
            )

        equity_df = pd.DataFrame(equity_curve).set_index("timestamp")
        equity_df.attrs["trades"] = trades

        trades_df = pd.DataFrame(trade_log)
        return equity_df, trades_df


# =========================
# 4. Auswertung: Kennzahlen & Plot
# =========================

def compute_performance_stats(equity_df: pd.DataFrame) -> dict:
    """
    Berechnet ein paar Standard-Kennzahlen:
    - Gesamtrendite
    - annualisierte Rendite (ungefähr, auf Basis von Trading-Tagen)
    - Maximaler Drawdown
    - Sharpe Ratio (sehr grob)
    """
    equity = equity_df["equity"]

    total_return = equity.iloc[-1] / equity.iloc[0] - 1

    days = (equity.index[-1].date() - equity.index[0].date()).days
    if days > 0:
        daily_equity = equity.resample("1D").last().dropna()
        daily_returns = daily_equity.pct_change(fill_method=None).dropna()
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


def plot_equity_curve(
    equity_df: pd.DataFrame,
    benchmark_curves: dict | None = None,
    title: str = "Equity Curve",
):
    """
    Plottet die Strategie-Equity und optional mehrere Benchmarks
    (z.B. S&P 500, ATX), alle bereits auf das gleiche Startkapital normiert.
    """
    plt.figure(figsize=(12, 6))

    # Strategie
    plt.plot(equity_df.index, equity_df["equity"], label="Strategy", linewidth=2)

    # Benchmarks
    if benchmark_curves:
        for name, series in benchmark_curves.items():
            plt.plot(series.index, series.values, label=name, linestyle="--")

    plt.title(title)
    plt.xlabel("Zeit")
    plt.ylabel("Kontowert (EUR)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# 5. Optional: Grid Search für Parameter
# =========================

def grid_search_mean_reversion(df: pd.DataFrame,
                               bt_config: BacktestConfig,
                               lookbacks=(5, 10, 20, 40),
                               z_entries=(0.5, 1.0, 1.5, 2.0),
                               z_exits=(0.2, 0.3)):
    """
    Einfache Grid-Search über Parameter-Kombinationen.
    Gibt eine DataFrame mit Kennzahlen pro Kombination zurück.
    """
    results = []

    for lb in lookbacks:
        for ze in z_entries:
            for zx in z_exits:
                mr_conf = MeanReversionConfig(
                    lookback=lb,
                    z_entry=ze,
                    z_exit=zx,
                )
                strat = MeanReversionIntraday(mr_conf)
                df_sig = strat.generate_signals(df)

                bt = IntradayBacktester(bt_config)
                equity_df, trades_df = bt.run(df_sig)
                stats = compute_performance_stats(equity_df)

                results.append({
                    "lookback": lb,
                    "z_entry": ze,
                    "z_exit": zx,
                    "total_return": stats["total_return"],
                    "annual_return": stats["annual_return"],
                    "max_drawdown": stats["max_drawdown"],
                    "sharpe": stats["sharpe"],
                    "num_trades": len(trades_df),
                })

    res_df = pd.DataFrame(results)
    # Wichtig: bei NaNs sort_values vorsichtig, daher fillna(-inf) für Sharpe
    res_df = res_df.sort_values("sharpe", ascending=False)
    return res_df


# =========================
# 6. Main: Alles zusammenstecken
# =========================

if __name__ == "__main__":
    # -------------------------
    # Parameter, die du leicht ändern kannst:
    # -------------------------
    TICKER = "RBI.VI"           # Erste Group an der Wiener Börse
    END = datetime.today().date()
    START = END - timedelta(days=6000)   # langer Zeitraum
    INTERVAL = "1d"

    RUN_GRID_SEARCH = False  # auf True setzen, um Grid-Search zu starten

    # 1. Daten laden – Ziel-Aktie
    print(f"Lade Daten für {TICKER} ...")
    df = download_intraday_data(TICKER, START, END, INTERVAL)
    print(f"{len(df)} Bars geladen.")

    # 2. Strategie konfigurieren & Signale generieren (Baseline)
    mr_config = MeanReversionConfig(
        lookback=20,
        z_entry=1.5,
        z_exit=0.3,
    )
    strategy = MeanReversionIntraday(mr_config)
    df_with_signals = strategy.generate_signals(df)

    # 3. Backtester konfigurieren & laufen lassen
    bt_config = BacktestConfig(
        initial_cash=10_000.0,
        commission_per_trade=3.0,
        slippage_bps=1.0,
        stop_loss_pct=0.3,   # Stop-Loss pro Trade (8 %)
    )
    backtester = IntradayBacktester(bt_config)
    equity_df, trades_df = backtester.run(df_with_signals)

    # 4. Benchmarks: S&P 500 und ATX
    bench_start = equity_df.index[0].date()
    bench_end = equity_df.index[-1].date()

    print("Lade Benchmark-Daten (S&P 500, ATX) ...")
    spx_df = download_intraday_data("^GSPC", bench_start, bench_end, "1d")
    atx_df = download_intraday_data("^ATX", bench_start, bench_end, "1d")

    spx_close = spx_df["close"].reindex(equity_df.index, method="ffill")
    atx_close = atx_df["close"].reindex(equity_df.index, method="ffill")

    init = bt_config.initial_cash
    spx_equity = spx_close / spx_close.iloc[0] * init
    atx_equity = atx_close / atx_close.iloc[0] * init

    benchmark_curves = {
        "S&P 500 (^GSPC)": spx_equity,
        "ATX (^ATX)": atx_equity,
    }

    # 5. Performance auswerten
    stats = compute_performance_stats(equity_df)

    print("\n===== Backtest-Resultate =====")
    print(f"Startkapital:      {bt_config.initial_cash:,.2f} EUR")
    print(f"Endkapital:        {equity_df['equity'].iloc[-1]:,.2f} EUR")
    print(f"Gesamtrendite:     {stats['total_return']*100:,.2f} %")
    print(
        f"Annualisierte R.:  {stats['annual_return']*100:,.2f} %"
        if not np.isnan(stats['annual_return']) else "Annualisierte R.:  n/a"
    )
    print(f"Max Drawdown:      {stats['max_drawdown']*100:,.2f} %")
    print(
        f"Sharpe Ratio:      {stats['sharpe']:,.2f}"
        if not np.isnan(stats['sharpe']) else "Sharpe Ratio:      n/a"
    )
    print(f"Anzahl Trades:     {len(trades_df)}")

    if not trades_df.empty:
        print("\nErste Trades:")
        print(trades_df.head())
        print("\nDurchschnittliche Trade-Return: "
              f"{100 * trades_df['return_pct'].mean():.2f} %")
        print("Win-Rate: "
              f"{100 * (trades_df['pnl'] > 0).mean():.2f} %")

    # 6. Equity Curve inkl. Benchmarks plotten
    plot_equity_curve(
        equity_df,
        benchmark_curves=benchmark_curves,
        title=f"Equity Curve – {TICKER} vs. S&P 500 & ATX",
    )

    # 7. Optional: Grid Search laufen lassen
    if RUN_GRID_SEARCH:
        print("\nStarte Grid-Search über Parameter ...")
        res_df = grid_search_mean_reversion(df, bt_config)
        print(res_df.head(20))
