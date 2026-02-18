# src/backtest.py
"""
Backtesting Framework für AlpineEdge Trading Strategies.

Simuliert die Strategie auf historischen Daten und berechnet Performance-Metriken.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

import pandas as pd
import numpy as np

from portfolio import Portfolio


class Backtester:
    """
    Backtesting Engine für Trading Strategies.
    
    Features:
    - Day-by-day simulation auf historischen Daten
    - Stop Loss / Take Profit execution
    - Performance metrics (Sharpe, Max DD, Win Rate)
    - Equity curve tracking
    - Trade log export
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 10000,
        universe: List[str] = None
    ):
        self.start = pd.Timestamp(start_date, tz='UTC')
        self.end = pd.Timestamp(end_date, tz='UTC')
        self.portfolio = Portfolio(initial_capital)
        self.universe = universe or []
        
        # Tracking
        self.trades_log: List[Dict] = []
        self.daily_log: List[Dict] = []
        
        # Historical Data Cache
        self.historical_data: Dict[str, pd.DataFrame] = {}
    
    def load_historical_data(self, data_dir: Path) -> None:
        """
        Lädt historische Preisdaten für Universe.
        
        Erwartet processed JSON files mit history DataFrame.
        """
        print(f"[BACKTEST] Loading historical data for {len(self.universe)} tickers...")
        
        for ticker in self.universe:
            try:
                json_path = data_dir / f"{ticker}.json"
                
                if not json_path.exists():
                    print(f"[BACKTEST] Warning: No data for {ticker}")
                    continue
                
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # History block extrahieren
                history_block = data.get('history', {}).get('data', [])
                
                if not history_block:
                    print(f"[BACKTEST] Warning: No history data for {ticker}")
                    continue
                
                # DataFrame bauen
                df = pd.DataFrame(history_block)
                
                # Datum parsen
                for date_col in ['Date', 'date', 'Datetime', 'datetime']:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
                        df.set_index(date_col, inplace=True)
                        break
                
                # Close price extrahieren
                for price_col in ['Adj Close', 'AdjClose', 'Close', 'close']:
                    if price_col in df.columns:
                        df['Close'] = pd.to_numeric(df[price_col], errors='coerce')
                        break
                
                # Nur Close behalten, nach Datum sortieren
                df = df[['Close']].sort_index()
                df = df[~df.index.duplicated(keep='first')]  # Remove duplicates
                
                # Filter auf Backtest-Zeitraum
                df = df[(df.index >= self.start) & (df.index <= self.end)]
                
                if len(df) > 0:
                    self.historical_data[ticker] = df
                    print(f"[BACKTEST] Loaded {ticker}: {len(df)} days")
                
            except Exception as e:
                print(f"[BACKTEST] Error loading {ticker}: {e}")
        
        print(f"[BACKTEST] Successfully loaded {len(self.historical_data)} tickers")
    
    def get_price(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        """Holt Preis für einen Ticker an einem bestimmten Datum"""
        if ticker not in self.historical_data:
            return None
        
        df = self.historical_data[ticker]
        
        # Use asof für nächsten verfügbaren Preis
        try:
            price = df['Close'].asof(date)
            if pd.isna(price):
                return None
            return float(price)
        except:
            return None
    
    def get_current_prices(self, date: pd.Timestamp) -> Dict[str, float]:
        """Holt Preise für alle Ticker an einem Datum"""
        prices = {}
        for ticker in self.universe:
            price = self.get_price(ticker, date)
            if price is not None:
                prices[ticker] = price
        return prices
    
    def run(
        self,
        strategy_func,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Führt Backtest aus.
        
        Args:
            strategy_func: Function(date, market_data, portfolio) -> Dict[ticker, signal]
                          Signal = {"action": "BUY"/"SELL", "confidence": float, "reason": str}
            verbose: Ob Fortschritt geloggt werden soll
        
        Returns:
            Performance metrics Dict
        """
        print(f"\n[BACKTEST] Starting backtest from {self.start.date()} to {self.end.date()}")
        print(f"[BACKTEST] Universe: {len(self.universe)} tickers")
        print(f"[BACKTEST] Initial Capital: ${self.portfolio.initial_capital:,.2f}\n")
        
        # Trading days generieren (nur Werktage)
        trading_days = pd.date_range(self.start, self.end, freq='B', tz='UTC')
        
        for i, date in enumerate(trading_days):
            # Progress
            if verbose and i % 30 == 0:
                progress = (i / len(trading_days)) * 100
                print(f"[BACKTEST] Progress: {progress:.1f}% | {date.date()} | Portfolio Value: ${self.portfolio.get_total_value(self.get_current_prices(date)):,.2f}")
            
            # 1. Get market data
            current_prices = self.get_current_prices(date)
            
            if not current_prices:
                continue
            
            # 2. Check stops on existing positions FIRST
            for ticker in list(self.portfolio.positions.keys()):
                if ticker not in current_prices:
                    continue
                
                price = current_prices[ticker]
                action = self.portfolio.check_stops(ticker, price)
                
                if action:
                    result = self.portfolio.sell(ticker, price, reason=action)
                    
                    if result['success']:
                        self.trades_log.append({
                            "date": date.isoformat(),
                            "ticker": ticker,
                            "action": action,
                            "price": price,
                            "pnl": result['pnl'],
                            "pnl_pct": result['pnl_pct']
                        })
                        
                        if verbose:
                            pnl_str = f"+${result['pnl']:.2f}" if result['pnl'] > 0 else f"-${abs(result['pnl']):.2f}"
                            print(f"  [{date.date()}] {action}: {ticker} @ ${price:.2f} | P&L: {pnl_str} ({result['pnl_pct']:+.1f}%)")
            
            # 3. Generate new signals from strategy
            try:
                signals = strategy_func(date, current_prices, self.portfolio)
            except Exception as e:
                print(f"[BACKTEST] Strategy error on {date.date()}: {e}")
                signals = {}
            
            # 4. Execute signals
            for ticker, signal in signals.items():
                if ticker not in current_prices:
                    continue
                
                action = signal.get('action')
                confidence = signal.get('confidence', 0.5)
                reason = signal.get('reason', '')
                price = current_prices[ticker]
                
                if action == 'BUY':
                    result = self.portfolio.buy(ticker, price, confidence, reason)
                    
                    if result['success']:
                        self.trades_log.append({
                            "date": date.isoformat(),
                            "ticker": ticker,
                            "action": "BUY",
                            "price": price,
                            "shares": result['shares'],
                            "position_value": result['position_value'],
                            "reason": reason
                        })
                        
                        if verbose:
                            print(f"  [{date.date()}] BUY: {ticker} @ ${price:.2f} | Size: ${result['position_value']:.0f} | Conf: {confidence:.2f}")
                
                elif action == 'SELL' and ticker in self.portfolio.positions:
                    result = self.portfolio.sell(ticker, price, reason="SIGNAL_SELL")
                    
                    if result['success']:
                        self.trades_log.append({
                            "date": date.isoformat(),
                            "ticker": ticker,
                            "action": "SELL",
                            "price": price,
                            "pnl": result['pnl'],
                            "pnl_pct": result['pnl_pct']
                        })
                        
                        if verbose:
                            pnl_str = f"+${result['pnl']:.2f}" if result['pnl'] > 0 else f"-${abs(result['pnl']):.2f}"
                            print(f"  [{date.date()}] SELL: {ticker} @ ${price:.2f} | P&L: {pnl_str} ({result['pnl_pct']:+.1f}%)")
            
            # 5. Track daily portfolio value
            self.portfolio.snapshot(date, current_prices)
            
            total_value = self.portfolio.get_total_value(current_prices)
            self.daily_log.append({
                "date": date.isoformat(),
                "value": total_value,
                "cash": self.portfolio.cash,
                "positions": len(self.portfolio.positions),
                "return_pct": (total_value / self.portfolio.initial_capital - 1) * 100
            })
        
        # Final Summary
        final_prices = self.get_current_prices(self.end)
        metrics = self.calculate_metrics()
        
        print(f"\n[BACKTEST] Completed!")
        print("="*60)
        print(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
        print(f"Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"CAGR: {metrics['cagr']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"Total Trades: {metrics['total_trades']}")
        print("="*60 + "\n")
        
        return metrics
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Berechnet Performance Metriken"""
        
        if not self.daily_log:
            return {}
        
        df = pd.DataFrame(self.daily_log)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Basic metrics
        initial_value = self.portfolio.initial_capital
        final_value = df['value'].iloc[-1]
        total_return_pct = (final_value / initial_value - 1) * 100
        
        # CAGR
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        cagr = (pow(final_value / initial_value, 1/years) - 1) * 100 if years > 0 else 0
        
        # Returns
        df['returns'] = df['value'].pct_change()
        
        # Sharpe Ratio (annualized)
        mean_return = df['returns'].mean()
        std_return = df['returns'].std()
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # Max Drawdown
        running_max = df['value'].cummax()
        drawdown = (df['value'] / running_max - 1)
        max_drawdown_pct = drawdown.min() * 100
        
        # Trade Stats
        closed_trades = [t for t in self.trades_log if 'pnl' in t]
        winning_trades = len([t for t in closed_trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in closed_trades if t.get('pnl', 0) < 0])
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Average Win/Loss
        wins = [t['pnl'] for t in closed_trades if t.get('pnl', 0) > 0]
        losses = [t['pnl'] for t in closed_trades if t.get('pnl', 0) < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        return {
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return_pct": total_return_pct,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown_pct,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            "days_tested": days,
            "years_tested": years
        }
    
    def export_results(self, output_dir: Path) -> None:
        """Exportiert Backtest-Ergebnisse als JSON und CSV"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Metrics
        metrics = self.calculate_metrics()
        metrics_file = output_dir / f"backtest_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[BACKTEST] Metrics saved to {metrics_file}")
        
        # Trades Log
        trades_df = pd.DataFrame(self.trades_log)
        trades_file = output_dir / f"backtest_trades_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"[BACKTEST] Trades saved to {trades_file}")
        
        # Equity Curve
        equity_df = pd.DataFrame(self.daily_log)
        equity_file = output_dir / f"backtest_equity_{timestamp}.csv"
        equity_df.to_csv(equity_file, index=False)
        print(f"[BACKTEST] Equity curve saved to {equity_file}")


if __name__ == "__main__":
    # Simple test strategy: Buy and Hold
    def buy_and_hold_strategy(date, prices, portfolio):
        """Simple Buy & Hold für Test"""
        signals = {}
        
        # Nur am ersten Tag kaufen
        if portfolio.total_trades == 0:
            for ticker in prices.keys():
                signals[ticker] = {
                    "action": "BUY",
                    "confidence": 0.5,
                    "reason": "Buy and Hold"
                }
        
        return signals
    
    # Test
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data" / "raw"
    
    universe = ["VOE.VI", "EBS.VI", "OMV.VI"]
    
    backtester = Backtester(
        start_date="2023-01-01",
        end_date="2024-12-31",
        initial_capital=10000,
        universe=universe
    )
    
    backtester.load_historical_data(DATA_DIR)
    metrics = backtester.run(buy_and_hold_strategy, verbose=True)
    
    # Export
    output_dir = PROJECT_ROOT / "backtest_results"
    backtester.export_results(output_dir)
