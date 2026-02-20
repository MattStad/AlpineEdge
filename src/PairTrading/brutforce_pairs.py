import itertools
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

import warnings

warnings.filterwarnings("ignore")  # Unterdrückt Warnungen bei vielen Regressionen

# =============================================================================
# 1. KONFIGURATION
# =============================================================================

TICKERS = [
    "AGR.VI",     # Agrana
    "ADKO.VI",    #Addiko Bank AG
    "AMAG.VI",    # AMAG Austria Metall
    "ANDR.VI",    # Andritz
    "ATS.VI",     # AT&S
    "ACAG.AT",       #Austriacard Holdings AG
    "BG.VI",   # BAWAG Group
    "BKS.VI",     # BKS Bank
    "BTS.VI",     # Bank für Tirol und Vorarlberg
    "BHD.VI",      # Burgenland Holding
    "BMAG.VI",      #Bajaj Mobility AG
    "CAI.VI",     # CA Immo
    "CPI.VI",     # CPI Property Group
    "DOC.VI",     # DO & CO
    "EBS.VI",     # Erste Group Bank
    "ETS.VI",     # EuroTeleSites
    "EVN.VI",     # EVN
    "FLU.VI",     # Flughafen Wien
    "FQT.VI",     # Frequentis
    "FACC.VI",      #FACC AG
    "GAGV.VI",     # Gurktaler AG
    "KTCG.VI",    # Kapsch TrafficCom
    "KTN.VI",     # Kontron
    "LNZ.VI",     # Lenzing
    "MARI.VI",    # Marinomed Biotech
    "MMK.VI",     # Mayr-Melnhof Karton
    "OBS.VI",     # Oberbank
    "OMV.VI",     # OMV
    "POST.VI",    # Österreichische Post
    "PAL.VI",     # Palfinger
    "PYT.VI",     # Polytec Holding
    "POS.VI",    # PORR
    "RBI.VI",     # Raiffeisen Bank International
    "RHIM.VI",     # RHI Magnesita
    "ROS.VI",     # Rosenbauer
    "SBO.VI",     # Schoeller-Bleckmann
    "SEM.VI",     # Semperit
    "STR.VI",     # Strabag
    "SWUT.VI",     # SW Umwelttechnik
    "TKA.VI",     # Telekom Austria
    "UBS.VI",     # UBM Development
    "UQA.VI",     # UNIQA Insurance Group
    "VLA.VI",     # Valneva
    "VER.VI",     # Verbund
    "VIG.VI",     # Vienna Insurance Group
    "VOE.VI",     # voestalpine
    "WIE.VI",     # Wienerberger
    "WOL.VI",     # Wolford
    "ZAG.VI"      # Zumtobel
]

START = "2022-01-01"  # Etwas späterer Start, damit fast alle Aktien Daten haben
END = "2025-12-31"


@dataclass
class PairTradingConfig:
    lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    max_holding_days: int | None = 30


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    commission_per_leg: float = 2.0
    slippage_bps: float = 1.0
    allocation_per_trade: float = 1.0


# =============================================================================
# 2. BACKTEST ENGINE
# =============================================================================

class PairTradingBacktester:
    def __init__(self, data: pd.DataFrame, t1: str, t2: str, strat_conf: PairTradingConfig, bt_conf: BacktestConfig):
        self.data = data
        self.t1 = t1
        self.t2 = t2
        self.strat_conf = strat_conf
        self.bt_conf = bt_conf

    def run(self) -> float:
        # Dies ist eine optimierte Backtest-Schleife, die nur den Endgewinn berechnet
        df = self.data.dropna(subset=["zscore"]).copy()

        cash = self.bt_conf.initial_capital
        position = 0
        shares1, shares2 = 0.0, 0.0
        entry_index = None
        slip_factor = 1 + self.bt_conf.slippage_bps / 10_000

        dates = df.index
        z_scores = df["zscore"].values
        p1_open = df[self.t1].values
        p2_open = df[self.t2].values

        for i in range(1, len(df)):
            z_prev = z_scores[i - 1]
            date_curr = dates[i]

            price1 = float(p1_open[i])
            price2 = float(p2_open[i])

            # Exit logic
            holding_days = (date_curr - entry_index).days if (position != 0 and entry_index is not None) else 0
            exit_signal = False

            if position != 0:
                if abs(z_prev) < self.strat_conf.exit_z:
                    exit_signal = True
                elif self.strat_conf.max_holding_days and holding_days >= self.strat_conf.max_holding_days:
                    exit_signal = True

            if position != 0 and exit_signal:
                if position == 1:
                    cash += shares1 * (price1 / slip_factor)
                    cash -= shares2 * (price2 * slip_factor)
                elif position == -1:
                    cash -= shares1 * (price1 * slip_factor)
                    cash += shares2 * (price2 / slip_factor)

                cash -= 2 * self.bt_conf.commission_per_leg
                position = 0
                shares1, shares2 = 0.0, 0.0
                entry_index = None

            # Entry logic
            if position == 0:
                if z_prev > self.strat_conf.entry_z:
                    leg_capital = cash * self.bt_conf.allocation_per_trade / 2.0
                    shares1 = leg_capital / (price1 * slip_factor)
                    shares2 = leg_capital / (price2 / slip_factor)

                    cash += shares1 * (price1 * (1 - self.bt_conf.slippage_bps / 10_000))
                    cash -= shares2 * (price2 * slip_factor)
                    cash -= 2 * self.bt_conf.commission_per_leg

                    position = -1
                    entry_index = date_curr

                elif z_prev < -self.strat_conf.entry_z:
                    leg_capital = cash * self.bt_conf.allocation_per_trade / 2.0
                    shares1 = leg_capital / (price1 / slip_factor)
                    shares2 = leg_capital / (price2 * slip_factor)

                    cash -= shares1 * (price1 * slip_factor)
                    cash += shares2 * (price2 * (1 - self.bt_conf.slippage_bps / 10_000))
                    cash -= 2 * self.bt_conf.commission_per_leg

                    position = 1
                    entry_index = date_curr

        # Position am Ende schließen (Mark-to-Market)
        if position == 1:
            equity = cash + shares1 * p1_open[-1] - shares2 * p2_open[-1]
        elif position == -1:
            equity = cash - shares1 * p1_open[-1] + shares2 * p2_open[-1]
        else:
            equity = cash

        profit = equity - self.bt_conf.initial_capital
        return profit


# =============================================================================
# 3. DATEN LADEN & PIPELINE
# =============================================================================

def main():
    print("Lade Kursdaten von Yahoo Finance...")
    price_data = {}

    for t in TICKERS:
        df = yf.download(t, start=START, end=END, progress=False)
        if not df.empty:
            # Handle yfinance multiindex columns format
            if isinstance(df.columns, pd.MultiIndex):
                s = df["Close"].iloc[:, 0]
            else:
                s = df["Close"]

            s = s.dropna()
            if len(s) >= 252:
                s.name = t
                price_data[t] = s

    print(f"{len(price_data)} Aktien erfolgreich geladen.\n")

    pairs = list(itertools.combinations(sorted(price_data.keys()), 2))
    print(f"Teste {len(pairs)} Paare auf Cointegration und simuliere Trades...")

    strat_conf = PairTradingConfig()
    bt_conf = BacktestConfig()

    results = []

    for t1, t2 in pairs:
        df_pair = pd.concat([price_data[t1], price_data[t2]], axis=1, keys=[t1, t2]).dropna()
        if len(df_pair) < 252:
            continue

        x = df_pair[t1].to_numpy()
        y = df_pair[t2].to_numpy()

        try:
            # 1. Cointegration Test
            score, pvalue, _ = coint(x, y)

            # Beschleunigung: Nur wenn Cointegration vorhanden (p < 0.05), wird der Backtest gestartet
            if pvalue < 0.05:
                # 2. Hedge Ratio (OLS)
                Y_reg = df_pair[t1]
                X_reg = sm.add_constant(df_pair[t2])
                beta = sm.OLS(Y_reg, X_reg).fit().params[t2]

                # 3. Z-Score berechnen
                data = df_pair.copy()
                data["spread"] = data[t1] - beta * data[t2]
                data["spread_mean"] = data["spread"].rolling(strat_conf.lookback).mean()
                data["spread_std"] = data["spread"].rolling(strat_conf.lookback).std()
                data["zscore"] = (data["spread"] - data["spread_mean"]) / data["spread_std"]

                # 4. Backtest
                bt = PairTradingBacktester(data, t1, t2, strat_conf, bt_conf)
                profit = bt.run()

                results.append({
                    "Pair": f"{t1} / {t2}",
                    "P-Value": pvalue,
                    "Profit (EUR)": profit
                })

        except Exception:
            continue

    # =============================================================================
    # 4. ERGEBNISSE AUSWERTEN
    # =============================================================================

    if not results:
        print("Keine profitablen, kointegrierten Paare gefunden.")
        return

    df_res = pd.DataFrame(results)
    # Sortieren nach dem höchsten Gewinn absteigend
    df_res = df_res.sort_values(by="Profit (EUR)", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 50)
    print("🏆 DIE TOP 10 PAARE (NACH GEWINN)")
    print("=" * 50)

    for i in range(min(10, len(df_res))):
        pair = df_res.iloc[i]
        print(f"Platz {i + 1}: {pair['Pair']}")
        print(f"  > P-Wert:  {pair['P-Value']:.6f} (Kointegriert)")
        print(f"  > Gewinn:  {pair['Profit (EUR)']:,.2f} EUR")
        print("-" * 50)


if __name__ == "__main__":
    main()