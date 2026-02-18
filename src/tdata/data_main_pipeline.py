# src/data_main_pipeline.py
#
# Orchestriert die komplette Daten-Pipeline:
# 1) Für alle Ticker aus config.TICKERS: Rohdaten von yfinance holen (data_loader)
# 2) Für alle Ticker: Metriken berechnen und ins JSON schreiben (metrics_builder)

from __future__ import annotations

from datetime import datetime
import traceback

from src.config import TICKERS
from src.tdata.data_loader import fetch_and_save_all_yf_data_for_ticker
from src.tdata.metrics_builder import add_metrics_to_json  # <--- dein echter Funktionsname


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PIPELINE {ts}] {msg}")


def run_data_loader() -> None:
    _log(f"Starte Data-Download für {len(TICKERS)} Ticker: {', '.join(TICKERS)}")
    for ticker in TICKERS:
        try:
            _log(f"[DATA] Lade Daten für {ticker} ...")
            # deine Funktion in data_loader.py hat aktuell nur (ticker) als Parameter
            fetch_and_save_all_yf_data_for_ticker(ticker)
            _log(f"[DATA] Fertig für {ticker}")
        except Exception as e:
            _log(f"[DATA] FEHLER für {ticker}: {e}")
            traceback.print_exc()


def run_metrics_builder() -> None:
    _log(f"Starte Metrics-Berechnung für {len(TICKERS)} Ticker: {', '.join(TICKERS)}")
    for ticker in TICKERS:
        try:
            _log(f"[METRICS] Berechne Metriken für {ticker} ...")
            add_metrics_to_json(ticker)
            _log(f"[METRICS] Fertig für {ticker}")
        except Exception as e:
            _log(f"[METRICS] FEHLER für {ticker}: {e}")
            traceback.print_exc()


def main() -> None:
    _log("=== AlpineEdge-Pipeline gestartet ===")

    # 1) Rohdaten bauen / aktualisieren
    run_data_loader()

    # 2) Metriken oben drauf rechnen
    run_metrics_builder()

    _log("=== AlpineEdge-Pipeline erfolgreich beendet ===")


if __name__ == "__main__":
    main()
