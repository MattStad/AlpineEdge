# src/data_loader.py
#
# End-to-End:
# - fetch_and_save_all_yf_data_for_ticker(ticker): lädt ALLE Daten von yfinance und speichert sie als JSON
# - download_universe(universe): macht das für mehrere Ticker
# - load_ticker_data(ticker): lädt JSON und gibt dir ein Dict[str, DataFrame] zurück

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import warnings

import pandas as pd
import yfinance as yf

import contextlib
import io

# --------------------------------------------------------------------
# Basis-Pfade
# --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../AlpineEdge
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)



# lästige Deprecation-Warnings von yfinance etwas dämpfen
warnings.filterwarnings(
    "ignore",
    message=".*earnings.*deprecated.*",
    category=DeprecationWarning,
)


# --------------------------------------------------------------------
# DOWNLOAD-TEIL: von yfinance holen und als JSON speichern
# --------------------------------------------------------------------

def get_ticker(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker)


def _safe_call(fn, description: str):
    """Ruhe-modus: fängt Exceptions ab und schluckt stdout/stderr-Ausgabe von yfinance."""
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            return fn()
    except Exception as e:
        print(f"[WARN] {description} failed: {e}")
        return None


def fetch_all_yf_raw(ticker: str) -> Dict[str, Any]:
    """
    Holt alles Relevante von yfinance für einen Ticker.
    Gibt ein Dict mit gemischten Objekten zurück (DataFrames, Series, dicts, lists, ...).
    """
    tk = get_ticker(ticker)

    data: Dict[str, Any] = {}

    # 1) Historische Preise (Daily)
    data["history"] = _safe_call(
        lambda: tk.history(period="max", interval="1d", auto_adjust=False),
        f"{ticker} history",
    )

    # 2) Unternehmensaktionen
    data["dividends"] = tk.dividends
    data["splits"] = tk.splits
    data["actions"] = tk.actions

    # 3) Info & fast_info
    data["info"] = _safe_call(tk.get_info, f"{ticker} get_info")

    fast_info = getattr(tk, "fast_info", None)
    if fast_info is not None:
        try:
            data["fast_info"] = dict(fast_info.__dict__)
        except Exception:
            try:
                data["fast_info"] = dict(fast_info)
            except Exception:
                data["fast_info"] = None

    # 4) Fundamentals: Jahres- und Quartalszahlen (leise, ohne 404-Output)
    data["income_stmt"] = _safe_call(lambda: tk.income_stmt, f"{ticker} income_stmt")
    data["quarterly_income_stmt"] = _safe_call(
        lambda: tk.quarterly_income_stmt, f"{ticker} quarterly_income_stmt"
    )
    data["balance_sheet"] = _safe_call(
        lambda: tk.balance_sheet, f"{ticker} balance_sheet"
    )
    data["quarterly_balance_sheet"] = _safe_call(
        lambda: tk.quarterly_balance_sheet, f"{ticker} quarterly_balance_sheet"
    )
    data["cashflow"] = _safe_call(lambda: tk.cashflow, f"{ticker} cashflow")
    data["quarterly_cashflow"] = _safe_call(
        lambda: tk.quarterly_cashflow, f"{ticker} quarterly_cashflow"
    )

    # 5) Earnings (deprecated, aber du wolltest "alles")
    data["earnings"] = _safe_call(lambda: tk.earnings, f"{ticker} earnings")
    data["quarterly_earnings"] = _safe_call(
        lambda: tk.quarterly_earnings, f"{ticker} quarterly_earnings"
    )


    # 6) Analysten & Empfehlungen
    data["recommendations"] = _safe_call(
        lambda: tk.recommendations, f"{ticker} recommendations"
    )
    data["analyst_price_targets"] = _safe_call(
        tk.get_analyst_price_targets, f"{ticker} analyst_price_targets"
    )

    # 7) news, Kalender, Nachhaltigkeit
    data["news"] = _safe_call(lambda: tk.news, f"{ticker} news")
    data["calendar"] = _safe_call(lambda: tk.calendar, f"{ticker} calendar")
    data["sustainability"] = _safe_call(
        lambda: tk.sustainability, f"{ticker} sustainability"
    )

    return data


def _to_jsonable(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Konvertiert pandas-Objekte in JSON-freundliche Strukturen.
    ALLES (Spaltennamen, Keys, Werte) wird zu Strings gemacht.
    Damit ist garantiert alles JSON-serialisierbar.
    """
    if obj is None:
        return None

    # DataFrame → Spaltennamen + Werte zu Strings
    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return None
        df = obj.reset_index()
        df.columns = [str(c) for c in df.columns]
        df = df.map(lambda x: str(x))
        return {
            "type": "dataframe",
            "data": df.to_dict(orient="records"),
        }

    # Series → wie kleines DataFrame
    if isinstance(obj, pd.Series):
        if obj.empty:
            return None
        df = obj.reset_index()
        df.columns = [str(c) for c in df.columns]
        df = df.map(lambda x: str(x))
        return {
            "type": "series",
            "data": df.to_dict(orient="records"),
        }

    # dict → Keys & Values zu Strings
    if isinstance(obj, dict):
        cleaned = {str(k): str(v) for k, v in obj.items()}
        return {
            "type": "dict",
            "data": cleaned,
        }

    # Liste von Dicts (z.B. news, recommendations)
    if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
        cleaned_rows = []
        for row in obj:
            cleaned_row = {str(k): str(v) for k, v in row.items()}
            cleaned_rows.append(cleaned_row)
        return {
            "type": "list_of_dicts",
            "data": cleaned_rows,
        }

    # Fallback: alles andere als String
    return {
        "type": "value",
        "data": str(obj),
    }


def save_to_json(ticker: str, raw_data: Dict[str, Any]) -> Path:
    """Speichert alle Daten zu einem Ticker in EINER .json Text-Datei."""
    path = DATA_DIR / f"{ticker}.json"
    out: Dict[str, Any] = {}

    for key, obj in raw_data.items():
        jsonable = _to_jsonable(obj)
        if jsonable is not None:
            out[key] = jsonable

    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved JSON for {ticker}: {path}")
    return path


def fetch_and_save_all_yf_data_for_ticker(ticker: str) -> Path:
    """High-Level: alles von yfinance holen und als JSON speichern."""
    print(f"\n=== {ticker} ===")
    raw = fetch_all_yf_raw(ticker)
    return save_to_json(ticker, raw)


def download_universe(universe: List[str]) -> None:
    """Für eine Liste von Tickern alle Daten ziehen & speichern."""
    for t in universe:
        fetch_and_save_all_yf_data_for_ticker(t)


# --------------------------------------------------------------------
# LOAD-TEIL: JSON -> pandas DataFrames
# --------------------------------------------------------------------

def _load_raw_json(ticker: str) -> Dict[str, Any]:
    path = DATA_DIR / f"{ticker}.json"
    if not path.exists():
        raise FileNotFoundError(f"JSON for {ticker} not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw


def _df_from_block(block: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """Nimmt einen unserer JSON-Blöcke und baut wieder einen DataFrame."""
    if not block:
        return None
    if block.get("type") not in ("dataframe", "series", "list_of_dicts"):
        return None
    data = block.get("")
    if not data:
        return None
    df = pd.DataFrame(data)
    return df


def _try_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Versucht, alle Spalten numerisch zu konvertieren, ohne abzustürzen."""
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    return df


def load_ticker_data(ticker: str) -> Dict[str, pd.DataFrame]:
    """
    Lädt alle Tabellen aus TICKER.json als DataFrames.
    Gibt ein Dict zurück, z.B.:
        data["history"], data["dividends"], data["income_stmt"], data["news"], ...
    """
    raw = _load_raw_json(ticker)

    out: Dict[str, pd.DataFrame] = {}

    # ---------------- History ----------------
    history_block = raw.get("history")
    history_df = _df_from_block(history_block)
    if history_df is not None:
        for date_col in ["Date", "date", "Datetime", "datetime"]:
            if date_col in history_df.columns:
                history_df[date_col] = pd.to_datetime(
                    history_df[date_col], errors="coerce", utc=True
                )
                history_df.set_index(date_col, inplace=True)
                break
        history_df = _try_numeric(history_df)
        out["history"] = history_df

    # ---------------- Dividenden ----------------
    dividends_df = _df_from_block(raw.get("dividends"))
    if dividends_df is not None:
        for date_col in ["Date", "date"]:
            if date_col in dividends_df.columns:
                dividends_df[date_col] = pd.to_datetime(
                    dividends_df[date_col], errors="coerce", utc=True
                )
                dividends_df.set_index(date_col, inplace=True)
                break
        dividends_df = _try_numeric(dividends_df)
        out["dividends"] = dividends_df

    # ---------------- Splits ----------------
    splits_df = _df_from_block(raw.get("splits"))
    if splits_df is not None:
        for date_col in ["Date", "date"]:
            if date_col in splits_df.columns:
                splits_df[date_col] = pd.to_datetime(
                    splits_df[date_col], errors="coerce", utc=True
                )
                splits_df.set_index(date_col, inplace=True)
                break
        splits_df = _try_numeric(splits_df)
        out["splits"] = splits_df

    # ---------------- Fundamentals ----------------
    for key in [
        "income_stmt",
        "quarterly_income_stmt",
        "balance_sheet",
        "quarterly_balance_sheet",
        "cashflow",
        "quarterly_cashflow",
        "earnings",
        "quarterly_earnings",
    ]:
        df = _df_from_block(raw.get(key))
        if df is not None:
            df = _try_numeric(df)
            out[key] = df

    # ---------------- Empfehlungen ----------------
    recos_df = _df_from_block(raw.get("recommendations"))
    if recos_df is not None:
        for col in recos_df.columns:
            if "date" in col.lower() or "time" in col.lower():
                recos_df[col] = pd.to_datetime(
                    recos_df[col], errors="coerce", utc=True
                )
        recos_df = _try_numeric(recos_df)
        out["recommendations"] = recos_df

    # ---------------- Analyst Price Targets ----------------
    targets_df = _df_from_block(raw.get("analyst_price_targets"))
    if targets_df is not None:
        targets_df = _try_numeric(targets_df)
        out["analyst_price_targets"] = targets_df

    # ---------------- news ----------------
    news_df = _df_from_block(raw.get("news"))
    if news_df is not None:
        for col in news_df.columns:
            if "time" in col.lower() or "date" in col.lower():
                news_df[col] = pd.to_datetime(
                    news_df[col], errors="coerce", utc=True
                )
        out["news"] = news_df

    # ---------------- Info & fast_info ----------------
    info_df = _df_from_block(raw.get("info"))
    if info_df is not None:
        out["info"] = info_df

    fast_info_df = _df_from_block(raw.get("fast_info"))
    if fast_info_df is not None:
        out["fast_info"] = fast_info_df

    return out


# --------------------------------------------------------------------
# kleiner Selbsttest, wenn direkt ausgeführt
# --------------------------------------------------------------------

if __name__ == "__main__":
    UNIVERSE = ["VOE.VI", "EBS.VI", "RBI.VI", "OMV.VI", "ANDR.VI"]

    # 1) Alle Daten ziehen & speichern
    download_universe(UNIVERSE)

    # 2) Einen Ticker laden & kurz anzeigen
    data = load_ticker_data("VOE.VI")

    print("Keys:", data.keys())
    print("\nHistory head:")
    print(data["history"].head())

    print("\nRecommendations:")
    print(data.get("recommendations"))

    print("\nnews head:")
    news_df = data.get("news")
    print(news_df.head() if news_df is not None else "Keine news")
