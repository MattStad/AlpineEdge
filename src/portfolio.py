# src/portfolio.py
"""
Portfolio Management System mit Position Sizing, Risk Management und P&L Tracking.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List, Any
import json
from pathlib import Path


@dataclass
class Position:
    """Repräsentiert eine offene Position"""
    ticker: str
    shares: float
    entry_price: float
    entry_date: datetime
    stop_loss: float
    take_profit: float
    entry_reason: str = ""


@dataclass
class Trade:
    """Repräsentiert einen abgeschlossenen Trade"""
    ticker: str
    action: str  # BUY, SELL, STOP_LOSS, TAKE_PROFIT
    shares: float
    price: float
    date: datetime
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    reason: str = ""


class Portfolio:
    """
    Portfolio Management mit Kelly Criterion Position Sizing und Risk Management.
    
    Features:
    - Dynamische Position Sizes basierend auf Confidence
    - Stop Loss / Take Profit Management
    - P&L Tracking
    - Portfolio Value Berechnung
    - Trade History
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        max_position_pct: float = 0.10,  # Max 10% pro Position
        stop_loss_pct: float = 0.05,     # 5% Stop Loss
        take_profit_pct: float = 0.15,    # 15% Take Profit
        max_open_positions: int = 10
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[Dict[str, Any]] = []
        
        # Risk Parameters
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_open_positions = max_open_positions
        
        # Tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
    
    def calculate_position_size(
        self,
        ticker: str,
        current_price: float,
        confidence: float
    ) -> float:
        """
        Berechnet Position Size mit Kelly Criterion-inspiriertem Ansatz.
        
        Args:
            ticker: Ticker Symbol
            current_price: Aktueller Preis
            confidence: Signal Confidence (0-1)
        
        Returns:
            Anzahl Shares zu kaufen
        """
        # Base Position Size: max 10% des Kapitals
        base_size = min(self.max_position_pct, confidence / 10)
        
        # Skaliere mit Confidence (höhere Conf = größere Position)
        scaled_size = base_size * (0.5 + confidence * 0.5)  # Min 50%, Max 100% of base
        
        # Berechne Position Value
        position_value = self.get_total_value() * scaled_size
        
        # Sicherstellen dass genug Cash da ist
        position_value = min(position_value, self.cash * 0.95)  # Max 95% des Cash
        
        shares = position_value / current_price
        return shares
    
    def buy(
        self,
        ticker: str,
        current_price: float,
        confidence: float,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Kauft eine Position wenn möglich.
        
        Returns:
            Dict mit success, message, shares
        """
        # Check 1: Schon eine Position?
        if ticker in self.positions:
            return {
                "success": False,
                "message": f"Already holding position in {ticker}",
                "shares": 0
            }
        
        # Check 2: Max Open Positions erreicht?
        if len(self.positions) >= self.max_open_positions:
            return {
                "success": False,
                "message": f"Max open positions ({self.max_open_positions}) reached",
                "shares": 0
            }
        
        # Check 3: Genug Cash?
        shares = self.calculate_position_size(ticker, current_price, confidence)
        position_value = shares * current_price
        
        if position_value > self.cash:
            return {
                "success": False,
                "message": f"Insufficient cash: need {position_value:.2f}, have {self.cash:.2f}",
                "shares": 0
            }
        
        # Execute Trade
        stop_loss = current_price * (1 - self.stop_loss_pct)
        take_profit = current_price * (1 + self.take_profit_pct)
        
        position = Position(
            ticker=ticker,
            shares=shares,
            entry_price=current_price,
            entry_date=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_reason=reason
        )
        
        self.positions[ticker] = position
        self.cash -= position_value
        self.total_trades += 1
        
        # Log Trade
        trade = Trade(
            ticker=ticker,
            action="BUY",
            shares=shares,
            price=current_price,
            date=datetime.now(),
            reason=reason
        )
        self.closed_trades.append(trade)
        
        return {
            "success": True,
            "message": f"Bought {shares:.2f} shares of {ticker} @ {current_price:.2f}",
            "shares": shares,
            "position_value": position_value,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }
    
    def sell(
        self,
        ticker: str,
        current_price: float,
        reason: str = "MANUAL_SELL"
    ) -> Dict[str, Any]:
        """
        Verkauft eine Position.
        
        Returns:
            Dict mit success, message, pnl, pnl_pct
        """
        if ticker not in self.positions:
            return {
                "success": False,
                "message": f"No position in {ticker} to sell",
                "pnl": 0,
                "pnl_pct": 0
            }
        
        position = self.positions[ticker]
        
        # Calculate P&L
        sell_value = position.shares * current_price
        buy_value = position.shares * position.entry_price
        pnl = sell_value - buy_value
        pnl_pct = (current_price / position.entry_price - 1) * 100
        
        # Execute
        self.cash += sell_value
        del self.positions[ticker]
        
        # Track
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Log Trade
        trade = Trade(
            ticker=ticker,
            action=reason,
            shares=position.shares,
            price=current_price,
            date=datetime.now(),
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason
        )
        self.closed_trades.append(trade)
        
        return {
            "success": True,
            "message": f"Sold {position.shares:.2f} shares of {ticker} @ {current_price:.2f}",
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "holding_days": (datetime.now() - position.entry_date).days
        }
    
    def check_stops(
        self,
        ticker: str,
        current_price: float
    ) -> Optional[str]:
        """
        Prüft ob Stop Loss oder Take Profit erreicht wurde.
        
        Returns:
            "STOP_LOSS", "TAKE_PROFIT" oder None
        """
        if ticker not in self.positions:
            return None
        
        position = self.positions[ticker]
        
        if current_price <= position.stop_loss:
            return "STOP_LOSS"
        
        if current_price >= position.take_profit:
            return "TAKE_PROFIT"
        
        return None
    
    def get_position_value(self, ticker: str, current_price: float) -> float:
        """Berechnet aktuellen Wert einer Position"""
        if ticker not in self.positions:
            return 0.0
        
        position = self.positions[ticker]
        return position.shares * current_price
    
    def get_total_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Berechnet Total Portfolio Value (Cash + Positions).
        
        Args:
            current_prices: Dict[ticker, price]. Wenn None, nutzt Entry Prices.
        """
        total = self.cash
        
        for ticker, position in self.positions.items():
            if current_prices and ticker in current_prices:
                price = current_prices[ticker]
            else:
                price = position.entry_price
            
            total += position.shares * price
        
        return total
    
    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """Berechnet unrealisierten P&L aller offenen Positionen"""
        unrealized = 0.0
        
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                current_value = position.shares * current_prices[ticker]
                entry_value = position.shares * position.entry_price
                unrealized += (current_value - entry_value)
        
        return unrealized
    
    def get_realized_pnl(self) -> float:
        """Berechnet realisierten P&L aller geschlossenen Trades"""
        return sum(t.pnl for t in self.closed_trades if t.pnl is not None)
    
    def get_win_rate(self) -> float:
        """Berechnet Win Rate"""
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return 0.0
        return self.winning_trades / total
    
    def get_metrics(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Berechnet Portfolio Performance Metriken.
        """
        total_value = self.get_total_value(current_prices)
        total_return = (total_value / self.initial_capital - 1) * 100
        
        realized_pnl = self.get_realized_pnl()
        unrealized_pnl = self.get_unrealized_pnl(current_prices or {})
        
        return {
            "total_value": total_value,
            "cash": self.cash,
            "invested": total_value - self.cash,
            "total_return_pct": total_return,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": realized_pnl + unrealized_pnl,
            "open_positions": len(self.positions),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.get_win_rate(),
        }
    
    def snapshot(self, date: datetime, current_prices: Dict[str, float]) -> None:
        """Speichert Equity Curve Snapshot"""
        total_value = self.get_total_value(current_prices)
        
        self.equity_curve.append({
            "date": date.isoformat(),
            "value": total_value,
            "cash": self.cash,
            "positions": len(self.positions),
            "pnl": total_value - self.initial_capital
        })
    
    def save_state(self, filepath: Path) -> None:
        """Speichert Portfolio State als JSON"""
        state = {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": {
                ticker: {
                    "ticker": p.ticker,
                    "shares": p.shares,
                    "entry_price": p.entry_price,
                    "entry_date": p.entry_date.isoformat(),
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "entry_reason": p.entry_reason
                }
                for ticker, p in self.positions.items()
            },
            "closed_trades": [
                {
                    "ticker": t.ticker,
                    "action": t.action,
                    "shares": t.shares,
                    "price": t.price,
                    "date": t.date.isoformat(),
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "reason": t.reason
                }
                for t in self.closed_trades
            ],
            "equity_curve": self.equity_curve,
            "metrics": self.get_metrics()
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        
        print(f"[PORTFOLIO] State saved to {filepath}")
    
    def print_summary(self, current_prices: Optional[Dict[str, float]] = None) -> None:
        """Druckt Portfolio Summary"""
        metrics = self.get_metrics(current_prices)
        
        print("\n" + "="*60)
        print("PORTFOLIO SUMMARY")
        print("="*60)
        print(f"Total Value:     ${metrics['total_value']:,.2f}")
        print(f"Cash:            ${metrics['cash']:,.2f}")
        print(f"Invested:        ${metrics['invested']:,.2f}")
        print(f"Total Return:    {metrics['total_return_pct']:.2f}%")
        print(f"Realized P&L:    ${metrics['realized_pnl']:,.2f}")
        print(f"Unrealized P&L:  ${metrics['unrealized_pnl']:,.2f}")
        print(f"Total P&L:       ${metrics['total_pnl']:,.2f}")
        print()
        print(f"Open Positions:  {metrics['open_positions']}")
        print(f"Total Trades:    {metrics['total_trades']}")
        print(f"Win Rate:        {metrics['win_rate']*100:.1f}%")
        print(f"  Wins:          {metrics['winning_trades']}")
        print(f"  Losses:        {metrics['losing_trades']}")
        print("="*60 + "\n")
        
        if self.positions:
            print("OPEN POSITIONS:")
            for ticker, pos in self.positions.items():
                current_price = current_prices.get(ticker, pos.entry_price) if current_prices else pos.entry_price
                pnl_pct = (current_price / pos.entry_price - 1) * 100
                days_held = (datetime.now() - pos.entry_date).days
                
                print(f"  {ticker}: {pos.shares:.2f} shares @ ${pos.entry_price:.2f}")
                print(f"    Current: ${current_price:.2f} | P&L: {pnl_pct:+.2f}% | Days: {days_held}")
                print(f"    Stop: ${pos.stop_loss:.2f} | Target: ${pos.take_profit:.2f}")
            print()


if __name__ == "__main__":
    # Test
    portfolio = Portfolio(initial_capital=10000)
    
    # Buy
    result = portfolio.buy("VOE.VI", 25.50, confidence=0.75, reason="Strong momentum")
    print(result)
    
    # Check value
    portfolio.print_summary({"VOE.VI": 26.00})
    
    # Sell
    result = portfolio.sell("VOE.VI", 27.00, reason="TAKE_PROFIT")
    print(result)
    
    portfolio.print_summary()
