# 🏔️ AlpineEdge

**AI-Powered Trading Council für die Wiener Börse (ATX)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/AI-Ollama%20Local-orange.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**AlpineEdge** ist ein lokales, KI-gestütztes Handelssystem, das quantitative Finanzdaten mit qualitativer Nachrichtenanalyse kombiniert. Anstatt sich auf ein einzelnes Modell zu verlassen, nutzt AlpineEdge einen **Swarm von spezialisierten KI-Agenten** (Trading Council), die über Kauf-, Verkaufs- oder Halteentscheidungen abstimmen.

---

## 🚀 Features

* **🧠 Local AI Swarm:** Nutzt `Ollama`, um mehrere LLMs (Llama 3.1, Qwen 2.5, Mistral, etc.) lokal und privatsphärenfreundlich laufen zu lassen.
* **🏛️ The Trading Council:** Ein Multi-Agenten-System, in dem verschiedene KI-Persönlichkeiten (der skeptische Risiko-Manager, der aggressive Analyst, etc.) diskutieren und abstimmen.
* **🇦🇹 ATX Fokus:** Spezialisiert auf den österreichischen Aktienmarkt (Wiener Börse), kombiniert mit globalen Makro-Daten.
* **📰 Smart News Engine:** Aggregiert Nachrichten via RSS und klassifiziert sie automatisch nach Relevanz und Sentiment.
* **📊 Quantitative Metriken:** Berechnet automatisch Sharpe Ratio, Volatilität, Max Drawdown und Performance-Trends.
* **🛡️ 100% Lokal:** Keine API-Kosten für LLMs, volle Datenhoheit.

---

## 🛠️ Architektur

Das System besteht aus drei Hauptkomponenten:

1.  **Data Pipeline (`src/tdata`)**:
    * Lädt historische Kurse via `yfinance`.
    * Berechnet technische Indikatoren und speichert sie in `processed` JSON-Dateien.
2.  **News Engine (`src/news`)**:
    * Scraping von RSS-Feeds (Global & Lokal).
    * Speicherung in einer SQLite Datenbank (`news.db`).
    * Filterung nach Relevanz ("Importance").
3.  **Strategy Runner (`src/strategy_runner.py`)**:
    * Lädt die Marktlage (Global & Österreich).
    * Füttert den **Trade Brain Swarm** mit Daten.
    * Aggregiert die Votes der KI-Modelle zu einer finalen Entscheidung.

---

## 🆕 What's New (v2.0)

**Major Updates:**
- ✅ **Portfolio Management System** - Position sizing, stop loss/take profit, P&L tracking
- ✅ **Backtesting Framework** - Test strategies on historical data
- ✅ **Confidence-Weighted Voting** - Smarter AI Council decisions (fixes HOLD-trap)
- ✅ **Sector-Aware Strategies** - Customized analysis per sector (Banking, Energy, etc.)
- ✅ **News Sentiment Scoring** - Quantified news impact analysis
- ✅ **Risk Manager** - Portfolio limits, sector exposure, correlation checks
- ✅ **Momentum Filters** - Only buy in uptrends

See `CHANGELOG.md` for full details.

## ⚙️ Installation

### Voraussetzungen
* Python 3.10 oder höher
* [Ollama](https://ollama.com/) installiert und laufend
* Empfohlen: GPU mit min. 8GB VRAM (für flüssige Inferenz)

### 1. Repository klonen

```bash
git clone [https://github.com/MattStad/AlpineEdge.git](https://github.com/MattStad/AlpineEdge.git)
cd AlpineEdge
```

### 2. Dependencies installieren
Erstelle ein virtuelles Environment und installiere die Pakete:

```bash
# Virtuelles Environment erstellen
python -m venv .venv

# Aktivieren (Windows)
.venv\Scripts\activate

# Aktivieren (Mac/Linux)
# source .venv/bin/activate

# Pakete installieren
pip install -r requirements.txt
```

### 3. KI-Modelle laden
Lade die Modelle für den Swarm herunter (via Ollama):

```bash
ollama pull llama3.1
ollama pull qwen2.5
ollama pull mistral-nemo
ollama pull phi3.5
ollama pull gemma2
```

---

## ▶️ Nutzung

### 1. Daten aktualisieren & Metriken bauen
Lade die neuesten Finanzdaten und berechne die Indikatoren:

```bash
python src/tdata/data_main_pipeline.py
python src/tdata/metrics_builder.py
```

### 2. News fetchen (Optional)
Aktualisiere die lokale News-Datenbank:

```bash
python src/news/news_rss_fetcher.py
```

### 3. Den KI-Rat einberufen (Strategy Runner)
Lass den Swarm die Aktien analysieren:

```bash
python src/strategy_runner.py
```

**Beispiel Output:**
```text
[SWARM] OMV.VI: Agents analyzing...
  > llama3.1 thinking...
    -> BUY (0.65) : Strong momentum + positive sentiment
  > qwen2.5 thinking...
    -> BUY (0.75) : Sector tailwinds (oil prices up)
  > mistral-nemo thinking...
    -> BUY (0.70) : Technical breakout confirmed
  ...
  => RESULT: BUY (Conf: 0.72) | Score: B:3, S:0, H:2
```

### 4. Backtesting (Strategie validieren)
Teste deine Strategie auf historischen Daten:

```bash
# Simple Rule-Based Backtest (schnell)
python run_backtest.py

# Mit echtem AI Council (langsam, aber realistisch)
python run_backtest.py --ai-council
```

**Output:**
```text
[BACKTEST] Starting backtest from 2023-01-01 to 2024-12-31
[BACKTEST] Universe: 8 tickers
[BACKTEST] Initial Capital: $10,000.00

[BACKTEST] Progress: 50.0% | 2023-07-01 | Portfolio Value: $11,234.00
...
[BACKTEST] Completed!
============================================================
Final Portfolio Value: $12,456.78
Total Return: 24.57%
CAGR: 11.23%
Sharpe Ratio: 1.45
Max Drawdown: -8.34%
Win Rate: 62.5%
Total Trades: 48
============================================================
```

Ergebnisse werden gespeichert in `backtest_results/`:
- `backtest_metrics_*.json` - Performance Kennzahlen
- `backtest_trades_*.csv` - Alle Trades mit P&L
- `backtest_equity_*.csv` - Equity Curve (täglich)

---

## 🤖 Der "Trading Council"

AlpineEdge verlässt sich nicht auf Zufall. Jedes Modell hat eine Rolle (definiert im Prompt Engineering):

| Modell | Rolle | Charakteristik |
| :--- | :--- | :--- |
| **Llama 3.1** | *The Chairman* | Ausgewogen, strikt, JSON-Compliance-König. |
| **Qwen 2.5** | *The Quant* | Stark in Logik, Zahlen und strukturierten Daten. |
| **Mistral Nemo** | *The Skeptic* | Großes Kontextfenster, sucht nach Risiken. |
| **Gemma 2** | *The Strategist* | Nuancierte Analyse, erkennt subtile Signale. |
| **Phi 3.5** | *The Scout* | Schnell und effizient für erste Einschätzungen. |

---

## 📂 Projektstruktur

```plaintext
AlpineEdge/
├── data/                 # Lokale Datenbanken & JSONs
│   ├── news.db           # SQLite Datenbank für Nachrichten
│   ├── raw/              # Rohdaten (yfinance)
│   └── processed/        # Berechnete Metriken für die KI
├── src/
│   ├── news/             # News Fetcher & Classifier
│   ├── tdata/            # Technical Data Pipeline
│   ├── trade_brain.py    # Die Logik des KI-Swarms
│   ├── strategy_runner.py# Hauptskript zur Ausführung
│   └── config.py         # Ticker-Listen & RSS-Feeds
├── requirements.txt      # Python Abhängigkeiten
└── README.md
```

---

## ⚠️ Disclaimer

**Dies ist keine Finanzberatung.**
AlpineEdge ist ein experimentelles Softwareprojekt zur Forschung an KI-gestützter Datenanalyse. Die generierten Signale ("BUY", "SELL") dienen rein informativen Zwecken. Der Handel mit Aktien und Finanzinstrumenten birgt ein hohes Verlustrisiko.

---
