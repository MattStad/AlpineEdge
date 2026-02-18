# run_backtest.py
"""
Example Backtest Script für AlpineEdge Strategy.

Testet die AI Council Trading Strategy auf historischen Daten.
"""

import asyncio
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.backtest import Backtester
from src.trade_brain import HttpClient, decide_for_ticker
from src.config import TICKERS
from src.sector_strategies import get_strategy_for_ticker
from src.news.news_sentiment import calculate_news_sentiment

# Pfade
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "backtest_results"


def load_processed_data(ticker: str):
    """Lädt processed metrics für einen Ticker"""
    import json
    
    file_path = PROCESSED_DIR / f"{ticker}_processed.json"
    
    if not file_path.exists():
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def ai_council_strategy(date, prices, portfolio):
    """
    AlpineEdge AI Council Strategy für Backtesting.
    
    Nutzt die gleiche Logik wie strategy_runner.py aber synchron.
    """
    signals = {}
    
    # Für jeden Ticker: Generate Signal
    for ticker in prices.keys():
        try:
            # 1. Load processed data
            processed_data = load_processed_data(ticker)
            
            if not processed_data:
                continue
            
            # 2. Momentum Filter (Quick Win!)
            momentum_20d = processed_data.get('metrics', {}).get('momentum_20d', 0)
            
            # Nur BUY wenn Momentum positiv (in Uptrend)
            if momentum_20d <= 0:
                continue
            
            # 3. Simplified Signal Generation (ohne LLM für Speed)
            # In Production würde man hier den AI Council nutzen
            
            metrics = processed_data.get('metrics', {})
            rsi = metrics.get('rsi_14', 50)
            sharpe = metrics.get('sharpe_annual', 0)
            
            # Simple Rules:
            # BUY wenn: RSI < 40 (oversold) und Momentum > 0 und Sharpe > 0.5
            if rsi < 40 and momentum_20d > 2 and sharpe > 0.5:
                signals[ticker] = {
                    "action": "BUY",
                    "confidence": 0.6 + (abs(50 - rsi) / 100),  # Höher wenn oversold
                    "reason": f"Oversold (RSI {rsi}) + Uptrend (M {momentum_20d:.1f}%)"
                }
            
            # SELL wenn: RSI > 70 (overbought) oder Momentum negativ
            elif ticker in portfolio.positions:
                if rsi > 70 or momentum_20d < -3:
                    signals[ticker] = {
                        "action": "SELL",
                        "confidence": 0.7,
                        "reason": f"Overbought (RSI {rsi}) or Downtrend"
                    }
        
        except Exception as e:
            print(f"[STRATEGY] Error for {ticker}: {e}")
            continue
    
    return signals


async def async_ai_council_strategy_wrapper(date, prices, portfolio):
    """
    Async wrapper für echten AI Council (optional, langsam!).
    
    Nutze das nur wenn du wirklich den LLM Council testen willst.
    Warnung: Sehr langsam! (5 Models x 20 Tickers x 500 Tage = viele API Calls)
    """
    client = HttpClient()
    signals = {}
    
    macro_context = {
        "global": "Generic global context for backtest",
        "austria": "Generic Austrian context for backtest"
    }
    
    for ticker in prices.keys():
        try:
            processed_data = load_processed_data(ticker)
            if not processed_data:
                continue
            
            sector_strategy = get_strategy_for_ticker(ticker)
            sector_context = sector_strategy.build_sector_context()
            
            # Simplified news (würde in Reality aus DB kommen)
            company_news = ["Generic news for backtest"]
            
            final, _ = await decide_for_ticker(
                client,
                ticker,
                processed_data,
                macro_context,
                company_news,
                sector_context
            )
            
            if final and final['action'] in ['BUY', 'SELL']:
                signals[ticker] = {
                    "action": final['action'],
                    "confidence": final.get('confidence', 0.5),
                    "reason": "AI Council Decision"
                }
        
        except Exception as e:
            print(f"[STRATEGY] Error for {ticker}: {e}")
    
    return signals


def run_simple_backtest():
    """
    Führt einfachen Backtest mit Rule-Based Strategy aus.
    Schnell, gut für erste Tests.
    """
    print("\n" + "="*60)
    print("ALPINEEDGE BACKTEST - SIMPLE RULE-BASED STRATEGY")
    print("="*60 + "\n")
    
    # Universe: Die liquidesten ATX Aktien
    universe = [
        "VOE.VI", "EBS.VI", "OMV.VI", "VER.VI", 
        "ANDR.VI", "RBI.VI", "WIE.VI", "EVN.VI"
    ]
    
    # Backtest Setup
    backtester = Backtester(
        start_date="2023-01-01",
        end_date="2024-12-31",
        initial_capital=10000,
        universe=universe
    )
    
    # Load Data
    backtester.load_historical_data(DATA_DIR)
    
    # Run
    metrics = backtester.run(ai_council_strategy, verbose=True)
    
    # Export
    backtester.export_results(OUTPUT_DIR)
    
    return metrics


def run_ai_council_backtest():
    """
    Führt Backtest mit echtem AI Council aus.
    WARNUNG: Sehr langsam! Nur für finale Validierung nutzen.
    """
    print("\n" + "="*60)
    print("ALPINEEDGE BACKTEST - AI COUNCIL (SLOW!)")
    print("="*60 + "\n")
    print("⚠️  This will take HOURS with 5 LLMs per ticker!")
    print("⚠️  Consider using run_simple_backtest() first.\n")
    
    universe = ["VOE.VI", "OMV.VI"]  # Nur 2 Ticker für Test
    
    backtester = Backtester(
        start_date="2024-06-01",
        end_date="2024-12-31",  # Nur 6 Monate
        initial_capital=10000,
        universe=universe
    )
    
    backtester.load_historical_data(DATA_DIR)
    
    # Wrapper für async strategy
    def sync_wrapper(date, prices, portfolio):
        return asyncio.run(async_ai_council_strategy_wrapper(date, prices, portfolio))
    
    metrics = backtester.run(sync_wrapper, verbose=True)
    backtester.export_results(OUTPUT_DIR)
    
    return metrics


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--ai-council":
        # Echte AI Council Strategy (langsam!)
        metrics = run_ai_council_backtest()
    else:
        # Simple Rule-Based Strategy (schnell)
        metrics = run_simple_backtest()
    
    print("\n✅ Backtest completed!")
    print(f"📊 Results saved to: {OUTPUT_DIR}")
