# src/tdata/metrics_builder.py
#
# Liest ein TICKER.json (Raw Data), berechnet Performance-Metriken (in %)
# und erstellt ein neues, kompaktes JSON im 'processed' Ordner.
#
# Inhalt:
#  - Summary Metriken (Vol %, Sharpe, Drawdown %)
#  - Performance Tabelle (1T, 1W, 1M, 6M, YTD, 1J, 3J, 5J, MAX) in Prozent
#  - Preise der letzten Woche
#  - Sonstige Infos (Profile, Financials, Empfehlungen)

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import timedelta

import json
import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# Pfade
# --------------------------------------------------------------------

# __file__ = src/tdata/metrics_builder.py
# parents[0] = src/tdata
# parents[1] = src
# parents[2] = Project Root (wo auch 'data' liegen sollte)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Sicherstellen, dass die Ordner existieren
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------
# Load / Save
# --------------------------------------------------------------------

def _load_raw_json(ticker: str) -> Dict[str, Any]:
    path = DATA_DIR / f"{ticker}.json"
    if not path.exists():
        raise FileNotFoundError(f"JSON for {ticker} not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw


def _save_processed_json(ticker: str, data: Dict[str, Any]) -> None:
    """Speichert das verarbeitete, kleinere JSON in data/processed/"""
    path = PROCESSED_DIR / f"{ticker}_processed.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[METRICS] Saved processed JSON for {ticker}: {path}")


def _df_from_block(block: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """Nimmt einen JSON-Block vom Typ dataframe/series/list_of_dicts und baut einen DataFrame."""
    if not block:
        return None
    # Flexible Typ-Prüfung
    if block.get("type") not in ("dataframe", "series", "list_of_dicts"):
        # Fallback: Versuch, direkt 'data' zu lesen
        if "data" not in block:
            return None

    data = block.get("data")
    if not data:
        return None
    df = pd.DataFrame(data)
    return df


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Konvertiert DataFrame in eine Liste von Dicts (für JSON), alles Strings."""
    df_out = df.reset_index()
    df_out.columns = [str(c) for c in df_out.columns]
    # Um sicherzugehen, dass Timestamps etc. lesbar bleiben:
    df_out = df_out.map(lambda x: str(x))
    return df_out.to_dict(orient="records")


# --------------------------------------------------------------------
# Daten-Aufbereitung
# --------------------------------------------------------------------

def _prepare_history_df(history_block: Dict[str, Any]) -> pd.DataFrame:
    """History-Block in sauberen DataFrame umwandeln (Datetime-Index, numerische Spalten)."""
    df = _df_from_block(history_block)
    if df is None or df.empty:
        raise ValueError("History block is empty or invalid")

    # Datums-Spalte erkennen
    for date_col in ["Date", "date", "Datetime", "datetime"]:
        if date_col in df.columns:
            # utc=True wichtig für .asof() Vergleiche
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
            df.set_index(date_col, inplace=True)
            break

    # Numerische Spalten konvertieren
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    # Nach Datum sortieren
    df.sort_index(inplace=True)

    return df


# --------------------------------------------------------------------
# Berechnung der Performance & Metriken
# --------------------------------------------------------------------

def compute_performance_table(history_df: pd.DataFrame, price_col: str) -> Dict[str, Any]:
    """
    Berechnet prozentuale Änderungen für verschiedene Zeiträume:
    1d, 1w, 1m, 6m, YTD, 1y, 3y, 5y, MAX.

    WICHTIG: Rückgabe in Prozent (z.B. 5.23 für 5.23%).
    """
    if history_df.empty:
        return {}

    prices = history_df[price_col]
    current_date = prices.index[-1]
    current_price = float(prices.iloc[-1])

    performance = {}

    def calc_pct_change_as_percent(old_price, new_price):
        if pd.isna(old_price) or old_price == 0:
            return None
        raw_ret = float((new_price - old_price) / old_price)
        return round(raw_ret * 100, 2)  # * 100 und runden

    # 1. Zeiträume definieren (in Tagen)
    periods = {
        "1d": 1,
        "1w": 7,
        "1m": 30,
        "6m": 182,
        "1y": 365,
        "3y": 365 * 3,
        "5y": 365 * 5
    }

    # 2. Lookups für die Zeiträume
    for label, days in periods.items():
        target_date = current_date - timedelta(days=days)
        try:
            if target_date < prices.index[0]:
                performance[label] = None
            else:
                past_price = prices.asof(target_date)
                performance[label] = calc_pct_change_as_percent(past_price, current_price)
        except Exception:
            performance[label] = None

    # 3. YTD (Year To Date)
    last_year_end = pd.Timestamp(year=current_date.year - 1, month=12, day=31).tz_localize(current_date.tz)

    if last_year_end < prices.index[0]:
        ytd_base = prices.iloc[0]  # Falls Historie dieses Jahr beginnt
    else:
        ytd_base = prices.asof(last_year_end)

    performance["YTD"] = calc_pct_change_as_percent(ytd_base, current_price)

    # 4. MAX (Gesamte Historie) -> Return über die maximale Dauer
    first_price = float(prices.iloc[0])
    performance["MAX"] = calc_pct_change_as_percent(first_price, current_price)

    return performance


def compute_risk_metrics(history_df: pd.DataFrame, price_col: str) -> Dict[str, Any]:
    """
    Berechnet Volatilität, Sharpe Ratio und Max Drawdown.
    Volatilität und Drawdown werden in Prozent (x100) zurückgegeben.
    Sharpe bleibt eine Ratio.
    """
    prices = history_df[price_col].astype(float)
    ret_1d = prices.pct_change().dropna()

    if len(prices) < 2:
        return {}

    # Annualisierte Volatilität
    vol_annual_raw = ret_1d.std() * np.sqrt(252)
    vol_annual_pct = round(vol_annual_raw * 100, 2)

    # Sharpe Ratio (Mean / Std * sqrt(252))
    # Sharpe ist keine Prozentzahl, daher keine *100 hier (außer für Basiswerte)
    mean_return_daily = ret_1d.mean()
    if vol_annual_raw > 0:
        sharpe_annual = float(mean_return_daily * 252 / vol_annual_raw)
        sharpe_annual = round(sharpe_annual, 2)
    else:
        sharpe_annual = 0.0

    # Max Drawdown
    running_max = prices.cummax()
    drawdown = (prices / running_max) - 1.0
    max_drawdown_raw = float(drawdown.min())
    max_drawdown_pct = round(max_drawdown_raw * 100, 2)

    return {
        "vol_annual": vol_annual_pct,  # in %
        "sharpe_annual": sharpe_annual,  # Ratio
        "max_drawdown": max_drawdown_pct  # in %
    }


# --------------------------------------------------------------------
# Haupt-Funktion
# --------------------------------------------------------------------

def add_metrics_to_json(ticker: str) -> None:
    """
    Lädt Raw-JSON, berechnet Metriken, extrahiert wichtige Infos
    und speichert ein 'processed' JSON.
    """
    try:
        raw = _load_raw_json(ticker)
    except FileNotFoundError:
        print(f"[METRICS] Skipping {ticker}, file not found.")
        return

    history_block = raw.get("history")
    if not history_block:
        print(f"[METRICS] No 'history' in JSON for {ticker}, skipping.")
        return

    # 1. History DataFrame vorbereiten
    try:
        history_df = _prepare_history_df(history_block)
    except Exception as e:
        print(f"[METRICS] Error processing history for {ticker}: {e}")
        return

    # Preis-Spalte finden (Adj Close bevorzugt)
    price_col = None
    for c in ["Adj Close", "AdjClose", "Close", "close"]:
        if c in history_df.columns:
            price_col = c
            break

    if not price_col:
        print(f"[METRICS] No price column found for {ticker}")
        return

    # 2. Berechnungen durchführen
    performance = compute_performance_table(history_df, price_col)
    risk_metrics = compute_risk_metrics(history_df, price_col)

    # 3. Preise der letzten Woche extrahieren (letzte 7 Einträge)
    last_week_df = history_df.tail(7).copy()
    last_week_data = _df_to_records(last_week_df)

    # 4. Neues JSON zusammenbauen
    current_price_val = float(history_df[price_col].iloc[-1])

    processed_data = {
        "ticker": ticker,
        "generated_at": str(pd.Timestamp.now()),
        "metrics": {
            **risk_metrics,  # Vol%, Sharpe, Drawdown%
            "current_price": round(current_price_val, 2)
        },
        "performance": performance,  # Jetzt in Prozent!
        "last_week_prices": last_week_data,
        "info": {}
    }

    # 5. Alles "andere Wichtige" aus dem Raw-JSON kopieren
    exclude_keys = ["history", "metrics_history", "metrics_summary"]
    for key, value in raw.items():
        if key not in exclude_keys:
            # z.B. recommendations, financials, profile
            processed_data["info"][key] = value

    # 6. Speichern
    _save_processed_json(ticker, processed_data)


def add_metrics_for_universe(universe: List[str]) -> None:
    for t in universe:
        try:
            add_metrics_to_json(t)
        except Exception as e:
            print(f"[METRICS] Failed for {t}: {e}")
