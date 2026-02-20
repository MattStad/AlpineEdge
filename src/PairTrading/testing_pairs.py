import itertools
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. KONFIGURATION & ZEITRÄUME
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

START_TRAIN = "2015-01-01"
END_TRAIN = "2022-12-31"

START_TEST = "2023-01-01"
END_TEST = "2025-12-31"

MIN_DAILY_TURNOVER_EUR = 200_000  # Mindestens 200k Euro Tagesumsatz im Median


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
        df = self.data.dropna(subset=["zscore"]).copy()
        if len(df) == 0: return 0.0

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

            holding_days = (date_curr - entry_index).days if (position != 0 and entry_index is not None) else 0
            exit_signal = False

            # Exit logic
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

        # Finale Abrechnung
        if position == 1:
            equity = cash + shares1 * p1_open[-1] - shares2 * p2_open[-1]
        elif position == -1:
            equity = cash - shares1 * p1_open[-1] + shares2 * p2_open[-1]
        else:
            equity = cash

        profit = equity - self.bt_conf.initial_capital
        return profit


# =============================================================================
# 3. DATEN LADEN & FILTERN
# =============================================================================

def main():
    print(f"Lade Kurs- und Volumendaten von {START_TRAIN} bis {END_TEST}...\n")
    price_data = {}

    for t in TICKERS:
        df = yf.download(t, start=START_TRAIN, end=END_TEST, progress=False)
        if not df.empty:
            # Handle Single vs MultiIndex columns von yfinance
            if isinstance(df.columns, pd.MultiIndex):
                close = df["Close"].iloc[:, 0]
                volume = df["Volume"].iloc[:, 0]
            else:
                close = df["Close"]
                volume = df["Volume"]

            close = close.dropna()
            volume = volume.dropna()

            # Liquiditäts-Check auf den Trainingsdaten (verhindert Lookahead Bias!)
            train_close = close.loc[:END_TRAIN]
            train_volume = volume.loc[:END_TRAIN]

            if len(train_close) >= 500:
                # Berechne den täglichen Umsatz in Euro
                daily_turnover = train_close * train_volume
                median_turnover = daily_turnover.median()

                if median_turnover >= MIN_DAILY_TURNOVER_EUR:
                    close.name = t
                    price_data[t] = close
                else:
                    print(f"  Aussortiert: {t} (Zu illiquide, Median-Umsatz: {median_turnover:,.0f} EUR)")

    print(f"\n{len(price_data)} liquide Aktien für den Backtest qualifiziert.\n")

    pairs = list(itertools.combinations(sorted(price_data.keys()), 2))
    print(f"Prüfe {len(pairs)} verbleibende Paare auf Cointegration (In-Sample) und simuliere OOS-Trades...")

    strat_conf = PairTradingConfig()
    bt_conf = BacktestConfig()
    results = []

    for t1, t2 in pairs:
        df_pair = pd.concat([price_data[t1], price_data[t2]], axis=1, keys=[t1, t2]).dropna()

        df_train = df_pair.loc[:END_TRAIN]
        df_test_raw = df_pair.loc[START_TEST:]

        if len(df_train) < 500 or len(df_test_raw) < 100:
            continue

        x_train = df_train[t1].to_numpy()
        y_train = df_train[t2].to_numpy()

        try:
            score, pvalue, _ = coint(x_train, y_train)

            if pvalue < 0.05:
                Y_reg = df_train[t1]
                X_reg = sm.add_constant(df_train[t2])
                beta = sm.OLS(Y_reg, X_reg).fit().params[t2]

                data = df_pair.copy()
                data["spread"] = data[t1] - beta * data[t2]
                data["spread_mean"] = data["spread"].rolling(strat_conf.lookback).mean()
                data["spread_std"] = data["spread"].rolling(strat_conf.lookback).std()
                data["zscore"] = (data["spread"] - data["spread_mean"]) / data["spread_std"]

                data_test = data.loc[START_TEST:]

                bt = PairTradingBacktester(data_test, t1, t2, strat_conf, bt_conf)
                profit = bt.run()

                results.append({
                    "Pair": f"{t1} / {t2}",
                    "Train P-Value": pvalue,
                    "OOS Profit (EUR)": profit
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
    df_res = df_res.sort_values(by="OOS Profit (EUR)", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("🏆 DIE TOP 5 ROBUSTEN & LIQUIDEN PAARE (NACH OOS GEWINN)")
    print("=" * 60)

    for i in range(min(5, len(df_res))):
        pair = df_res.iloc[i]
        profit = pair['OOS Profit (EUR)']

        print(f"Platz {i + 1}: {pair['Pair']}")
        print(f"  > In-Sample P-Wert (bis 2022): {pair['Train P-Value']:.6f}")
        print(f"  > Echter OOS Gewinn (ab 2023): {profit:,.2f} EUR")
        print("-" * 60)


if __name__ == "__main__":
    main()