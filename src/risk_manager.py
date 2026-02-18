# src/risk_manager.py
"""
Risk Management Layer für AlpineEdge.

Validiert Trade Signals basierend auf Portfolio Risk Limits.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from portfolio import Portfolio
from sector_strategies import get_sector_for_ticker, SECTOR_MAP

import numpy as np


class RiskManager:
    """
    Risk Management System mit mehreren Check-Layers.
    
    Rules:
    1. Portfolio Maximum Drawdown Limit
    2. Sector Exposure Limit
    3. Volatility vs Confidence Check
    4. Correlation Check (verhindert zu viele korrelierte Positionen)
    5. Position Size Limit
    """
    
    def __init__(
        self,
        max_portfolio_drawdown: float = 0.20,  # 20% Max DD Limit
        max_sector_exposure: float = 0.40,      # 40% max in einem Sektor
        max_correlated_positions: int = 2,      # Max 2 korrelierte Positionen
        high_volatility_threshold: float = 40.0,  # 40% annual vol = high
        min_confidence_for_high_vol: float = 0.7  # 70% confidence nötig
    ):
        self.max_portfolio_drawdown = max_portfolio_drawdown
        self.max_sector_exposure = max_sector_exposure
        self.max_correlated_positions = max_correlated_positions
        self.high_volatility_threshold = high_volatility_threshold
        self.min_confidence_for_high_vol = min_confidence_for_high_vol
    
    def check_signal(
        self,
        ticker: str,
        signal: Dict[str, Any],
        portfolio: Portfolio,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validiert ob ein Trade Signal ausgeführt werden darf.
        
        Args:
            ticker: Ticker Symbol
            signal: Dict mit action, confidence, reason
            portfolio: Portfolio instance
            market_data: Optional market data (für Volatilität, etc.)
        
        Returns:
            Dict mit approved: bool, reason: str
        """
        action = signal.get('action')
        confidence = signal.get('confidence', 0.5)
        
        # Nur BUY Signals checken (SELL immer erlaubt)
        if action != 'BUY':
            return {"approved": True, "reason": ""}
        
        # Rule 1: Portfolio Drawdown Limit
        check = self._check_portfolio_drawdown(portfolio)
        if not check['approved']:
            return check
        
        # Rule 2: Sector Exposure Limit
        check = self._check_sector_exposure(ticker, portfolio, signal, market_data)
        if not check['approved']:
            return check
        
        # Rule 3: Volatility Check
        if market_data:
            check = self._check_volatility(ticker, confidence, market_data)
            if not check['approved']:
                return check
        
        # Rule 4: Correlation Check
        check = self._check_correlation(ticker, portfolio)
        if not check['approved']:
            return check
        
        # Rule 5: Position Size Limit (max 15% in eine Position)
        check = self._check_position_size(ticker, portfolio, signal)
        if not check['approved']:
            return check
        
        return {"approved": True, "reason": "All risk checks passed"}
    
    def _check_portfolio_drawdown(self, portfolio: Portfolio) -> Dict[str, Any]:
        """
        Prüft ob Portfolio Drawdown Limit erreicht wurde.
        Wenn ja: keine neuen Positionen mehr.
        """
        current_value = portfolio.get_total_value()
        initial_value = portfolio.initial_capital
        
        # Berechne Drawdown vom Peak
        if hasattr(portfolio, 'equity_curve') and portfolio.equity_curve:
            values = [snapshot['value'] for snapshot in portfolio.equity_curve]
            peak = max(values)
            current_dd = (current_value / peak - 1)
        else:
            current_dd = (current_value / initial_value - 1)
        
        if current_dd < -self.max_portfolio_drawdown:
            return {
                "approved": False,
                "reason": f"Portfolio drawdown limit reached ({current_dd*100:.1f}% < {-self.max_portfolio_drawdown*100:.0f}%)"
            }
        
        return {"approved": True, "reason": ""}
    
    def _check_sector_exposure(
        self,
        ticker: str,
        portfolio: Portfolio,
        signal: Dict,
        market_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Prüft ob Sector Exposure Limit erreicht würde.
        """
        sector = get_sector_for_ticker(ticker)
        
        if sector is None:
            # Unbekannter Sektor = OK
            return {"approved": True, "reason": ""}
        
        # Berechne aktuellen Sector Exposure
        total_value = portfolio.get_total_value()
        sector_value = 0.0
        
        for pos_ticker, position in portfolio.positions.items():
            pos_sector = get_sector_for_ticker(pos_ticker)
            if pos_sector == sector:
                # Nutze Entry Price als Fallback wenn keine Market Data
                price = position.entry_price
                if market_data and pos_ticker in market_data:
                    price = market_data[pos_ticker].get('current_price', price)
                
                sector_value += position.shares * price
        
        current_exposure = sector_value / total_value if total_value > 0 else 0
        
        # Simuliere neue Position
        new_position_pct = min(0.10, signal.get('confidence', 0.5) / 10)
        new_sector_exposure = current_exposure + new_position_pct
        
        if new_sector_exposure > self.max_sector_exposure:
            return {
                "approved": False,
                "reason": f"Sector {sector} exposure limit would be exceeded ({new_sector_exposure*100:.1f}% > {self.max_sector_exposure*100:.0f}%)"
            }
        
        return {"approved": True, "reason": ""}
    
    def _check_volatility(
        self,
        ticker: str,
        confidence: float,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        High Volatility Assets brauchen höhere Confidence.
        """
        vol = market_data.get(ticker, {}).get('vol_annual')
        
        if vol is None:
            # Keine Volatility Data = OK
            return {"approved": True, "reason": ""}
        
        if vol > self.high_volatility_threshold:
            if confidence < self.min_confidence_for_high_vol:
                return {
                    "approved": False,
                    "reason": f"High volatility ({vol:.1f}%) requires confidence >= {self.min_confidence_for_high_vol:.0%} (got {confidence:.0%})"
                }
        
        return {"approved": True, "reason": ""}
    
    def _check_correlation(
        self,
        ticker: str,
        portfolio: Portfolio
    ) -> Dict[str, Any]:
        """
        Verhindert zu viele Positionen im selben Sektor.
        Vereinfachte Korrelations-Check via Sector.
        """
        sector = get_sector_for_ticker(ticker)
        
        if sector is None:
            return {"approved": True, "reason": ""}
        
        # Zähle Positionen im gleichen Sektor
        sector_positions = 0
        for pos_ticker in portfolio.positions.keys():
            pos_sector = get_sector_for_ticker(pos_ticker)
            if pos_sector == sector:
                sector_positions += 1
        
        if sector_positions >= self.max_correlated_positions:
            return {
                "approved": False,
                "reason": f"Already holding {sector_positions} positions in {sector} sector (max {self.max_correlated_positions})"
            }
        
        return {"approved": True, "reason": ""}
    
    def _check_position_size(
        self,
        ticker: str,
        portfolio: Portfolio,
        signal: Dict
    ) -> Dict[str, Any]:
        """
        Verhindert zu große Einzelpositionen (max 15% Portfolio).
        """
        confidence = signal.get('confidence', 0.5)
        max_size = min(0.15, confidence / 10)  # Max 15%
        
        # Standard Portfolio Position Sizing ist 10%, das ist OK
        # Nur wenn Confidence extrem hoch, könnte es größer werden
        if confidence > 1.0:  # Safety check
            return {
                "approved": False,
                "reason": f"Invalid confidence value: {confidence}"
            }
        
        return {"approved": True, "reason": ""}
    
    def get_risk_summary(self, portfolio: Portfolio) -> Dict[str, Any]:
        """
        Gibt Risk Summary für Portfolio zurück.
        """
        total_value = portfolio.get_total_value()
        
        # Sector Exposure berechnen
        sector_exposures = {}
        for pos_ticker, position in portfolio.positions.items():
            sector = get_sector_for_ticker(pos_ticker)
            if sector:
                value = position.shares * position.entry_price
                sector_exposures[sector] = sector_exposures.get(sector, 0) + value
        
        # Als Prozent
        sector_exposures_pct = {
            sector: (value / total_value * 100) if total_value > 0 else 0
            for sector, value in sector_exposures.items()
        }
        
        # Current Drawdown
        if hasattr(portfolio, 'equity_curve') and portfolio.equity_curve:
            values = [s['value'] for s in portfolio.equity_curve]
            peak = max(values)
            current_dd_pct = (total_value / peak - 1) * 100
        else:
            current_dd_pct = (total_value / portfolio.initial_capital - 1) * 100
        
        return {
            "portfolio_value": total_value,
            "current_drawdown_pct": current_dd_pct,
            "max_drawdown_limit_pct": self.max_portfolio_drawdown * 100,
            "open_positions": len(portfolio.positions),
            "sector_exposures_pct": sector_exposures_pct,
            "max_sector_exposure_pct": self.max_sector_exposure * 100
        }
    
    def print_summary(self, portfolio: Portfolio) -> None:
        """Druckt Risk Summary"""
        summary = self.get_risk_summary(portfolio)
        
        print("\n" + "="*60)
        print("RISK MANAGEMENT SUMMARY")
        print("="*60)
        print(f"Portfolio Value: ${summary['portfolio_value']:,.2f}")
        print(f"Current Drawdown: {summary['current_drawdown_pct']:.2f}%")
        print(f"DD Limit: {summary['max_drawdown_limit_pct']:.0f}%")
        print()
        print(f"Open Positions: {summary['open_positions']}")
        print()
        print("Sector Exposures:")
        for sector, exposure in sorted(summary['sector_exposures_pct'].items(), key=lambda x: -x[1]):
            print(f"  {sector}: {exposure:.1f}%")
        print(f"\nMax Sector Exposure: {summary['max_sector_exposure_pct']:.0f}%")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test
    from portfolio import Portfolio
    
    portfolio = Portfolio(initial_capital=10000)
    risk_manager = RiskManager()
    
    # Buy some positions
    portfolio.buy("EBS.VI", 40.0, confidence=0.7, reason="Test")
    portfolio.buy("RBI.VI", 18.0, confidence=0.6, reason="Test")
    
    # Check new signal
    signal = {"action": "BUY", "confidence": 0.65, "reason": "New signal"}
    result = risk_manager.check_signal("BG.VI", signal, portfolio)
    
    print(f"Signal Check Result: {result}")
    
    # Print summary
    risk_manager.print_summary(portfolio)
