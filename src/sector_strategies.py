# src/sector_strategies.py
"""
Sector-specific trading strategies for ATX stocks.

Different sectors need different analysis focus:
- Banking: Interest rates, NPL ratios, ECB policy
- Energy: Oil/Gas prices, EU regulations, Green Deal
- Industrial: Order books, Capex, global trade
- Real Estate: Interest rates, occupancy, construction
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


# Sector Mapping
SECTOR_MAP = {
    "Banking": ["EBS.VI", "RBI.VI", "BG.VI"],
    "Energy": ["OMV.VI", "VER.VI", "EVN.VI"],
    "Industrial": ["ANDR.VI", "VOE.VI", "STR.VI", "POS.VI", "SBO.VI"],
    "Real_Estate": ["CAI.VI", "WIE.VI", "CPI.VI"],
    "Consumer": ["DOC.VI"],
    "Insurance": ["UQA.VI", "VIG.VI"],
    "Technology": ["ATS.VI"],
    "Materials": ["LNZ.VI"],
    "Postal": ["POST.VI"],
}

# Reverse Mapping: Ticker -> Sector
TICKER_TO_SECTOR: Dict[str, str] = {}
for sector, tickers in SECTOR_MAP.items():
    for ticker in tickers:
        TICKER_TO_SECTOR[ticker] = sector


@dataclass
class SectorWeights:
    """Gewichtung verschiedener Faktoren pro Sektor"""
    rsi: float
    sharpe: float
    news_sentiment: float
    macro_global: float
    macro_eu: float
    macro_local: float
    fundamentals: float


class SectorStrategy:
    """
    Sektor-spezifische Analyse-Strategie.
    
    Jeder Sektor hat eigene Gewichtungen und Fokus-Bereiche.
    """
    
    def __init__(self, sector: str):
        self.sector = sector
        self.weights = self.get_sector_weights()
        self.focus_areas = self.get_focus_areas()
    
    def get_sector_weights(self) -> SectorWeights:
        """
        Definiert Gewichtungen für verschiedene Analyse-Faktoren pro Sektor.
        """
        if self.sector == "Banking":
            return SectorWeights(
                rsi=0.15,
                sharpe=0.20,
                news_sentiment=0.25,
                macro_global=0.15,
                macro_eu=0.15,
                macro_local=0.05,
                fundamentals=0.05  # NPL Ratio, Capital Ratio
            )
        
        elif self.sector == "Energy":
            return SectorWeights(
                rsi=0.15,
                sharpe=0.20,
                news_sentiment=0.15,
                macro_global=0.35,  # Oil prices, USD
                macro_eu=0.10,      # EU Green Deal
                macro_local=0.00,
                fundamentals=0.05
            )
        
        elif self.sector == "Industrial":
            return SectorWeights(
                rsi=0.20,
                sharpe=0.25,
                news_sentiment=0.20,
                macro_global=0.20,  # Global trade
                macro_eu=0.10,
                macro_local=0.00,
                fundamentals=0.05   # Order books
            )
        
        elif self.sector == "Real_Estate":
            return SectorWeights(
                rsi=0.15,
                sharpe=0.25,
                news_sentiment=0.15,
                macro_global=0.05,
                macro_eu=0.25,      # ECB interest rates
                macro_local=0.10,   # Vienna market
                fundamentals=0.05   # Occupancy rates
            )
        
        elif self.sector == "Insurance":
            return SectorWeights(
                rsi=0.15,
                sharpe=0.30,        # Stable returns wichtig
                news_sentiment=0.20,
                macro_global=0.10,
                macro_eu=0.20,      # Regulation
                macro_local=0.00,
                fundamentals=0.05
            )
        
        elif self.sector == "Technology":
            return SectorWeights(
                rsi=0.25,           # Tech ist volatiler
                sharpe=0.15,
                news_sentiment=0.30,  # Earnings wichtig
                macro_global=0.20,
                macro_eu=0.05,
                macro_local=0.00,
                fundamentals=0.05
            )
        
        # Default für unbekannte Sektoren
        return SectorWeights(
            rsi=0.20,
            sharpe=0.20,
            news_sentiment=0.20,
            macro_global=0.15,
            macro_eu=0.15,
            macro_local=0.05,
            fundamentals=0.05
        )
    
    def get_focus_areas(self) -> List[str]:
        """
        Definiert die wichtigsten Analyse-Bereiche für einen Sektor.
        Wird in Prompts genutzt um LLM zu fokussieren.
        """
        if self.sector == "Banking":
            return [
                "ECB interest rate policy and impact on margins",
                "NPL (Non-Performing Loans) ratio trends",
                "Capital adequacy and regulatory compliance",
                "Deposit growth and loan demand",
                "Digital banking initiatives"
            ]
        
        elif self.sector == "Energy":
            return [
                "Oil and gas price trends",
                "USD exchange rate impact",
                "EU Green Deal and carbon pricing",
                "Renewable energy transition plans",
                "Capex allocation (growth vs dividends)"
            ]
        
        elif self.sector == "Industrial":
            return [
                "Order book development",
                "Global trade volumes and tariffs",
                "Raw material costs (steel, copper)",
                "Manufacturing PMI indicators",
                "Infrastructure investment plans"
            ]
        
        elif self.sector == "Real_Estate":
            return [
                "Interest rate environment and financing costs",
                "Occupancy rates and rental income",
                "Construction costs and supply",
                "Vienna/CEE property market trends",
                "Regulatory changes (rent control)"
            ]
        
        elif self.sector == "Insurance":
            return [
                "Claims ratio development",
                "Investment portfolio performance",
                "Solvency II capital requirements",
                "CEE market expansion",
                "Natural disaster exposure"
            ]
        
        elif self.sector == "Technology":
            return [
                "Semiconductor demand and chip shortages",
                "R&D spending and innovation pipeline",
                "Customer concentration risks",
                "Competition from Asian suppliers",
                "Supply chain resilience"
            ]
        
        elif self.sector == "Consumer":
            return [
                "Consumer spending trends",
                "Tourism recovery (especially aviation catering)",
                "Margin pressure from inflation",
                "Contract renewals and pricing power"
            ]
        
        elif self.sector == "Materials":
            return [
                "Raw material prices (pulp, chemicals)",
                "Sustainability and circular economy trends",
                "Customer industry health (textiles)",
                "Capacity utilization rates"
            ]
        
        return ["General market trends", "Company-specific news", "Valuation metrics"]
    
    def build_sector_context(self, macro_data: Optional[Dict] = None) -> str:
        """
        Baut sektor-spezifischen Kontext-String für LLM Prompts.
        """
        focus_str = "\n".join([f"  - {area}" for area in self.focus_areas])
        
        context = f"""
=== SECTOR-SPECIFIC ANALYSIS: {self.sector.upper()} ===

Key Focus Areas for {self.sector}:
{focus_str}

Analysis Weight Distribution:
  - Technical (RSI): {self.weights.rsi*100:.0f}%
  - Risk-Adjusted Return (Sharpe): {self.weights.sharpe*100:.0f}%
  - News Sentiment: {self.weights.news_sentiment*100:.0f}%
  - Global Macro: {self.weights.macro_global*100:.0f}%
  - EU/ECB Macro: {self.weights.macro_eu*100:.0f}%
  - Local Austrian: {self.weights.macro_local*100:.0f}%
  - Fundamentals: {self.weights.fundamentals*100:.0f}%
"""
        
        # Add sector-specific macro indicators if available
        if macro_data:
            if self.sector == "Banking" and "ecb_rate" in macro_data:
                context += f"\nECB Main Rate: {macro_data['ecb_rate']}%"
            
            if self.sector == "Energy" and "oil_price" in macro_data:
                context += f"\nBrent Crude: ${macro_data['oil_price']}/bbl"
            
            if self.sector == "Real_Estate" and "mortgage_rates" in macro_data:
                context += f"\nMortgage Rates: {macro_data['mortgage_rates']}%"
        
        return context
    
    def adjust_signal_confidence(
        self,
        base_confidence: float,
        ticker: str,
        metrics: Dict
    ) -> float:
        """
        Passt Confidence basierend auf sektor-spezifischen Faktoren an.
        
        Beispiel: Bei Banking-Aktien höhere Confidence wenn Zinsen steigen.
        """
        adjusted = base_confidence
        
        # Banking: Belohne hohe Sharpe Ratio (stabile Banken bevorzugt)
        if self.sector == "Banking":
            sharpe = metrics.get("sharpe_annual", 0)
            if sharpe > 1.0:
                adjusted += 0.05
            elif sharpe < 0.5:
                adjusted -= 0.05
        
        # Energy: Penalty bei hoher Volatilität
        elif self.sector == "Energy":
            vol = metrics.get("vol_annual", 0)
            if vol > 40:  # > 40% annual vol
                adjusted -= 0.10
        
        # Technology: Belohne Momentum
        elif self.sector == "Technology":
            perf_1m = metrics.get("performance", {}).get("1m", 0)
            if perf_1m and perf_1m > 5:  # +5% last month
                adjusted += 0.05
        
        # Real Estate: Penalty wenn Interest Rates steigen (würde man aus Macro holen)
        # elif self.sector == "Real_Estate":
        #     if rising_interest_rates:
        #         adjusted -= 0.10
        
        # Clamp to 0-1
        return max(0.0, min(1.0, adjusted))


def get_sector_for_ticker(ticker: str) -> Optional[str]:
    """Gibt Sektor für einen Ticker zurück"""
    return TICKER_TO_SECTOR.get(ticker)


def get_strategy_for_ticker(ticker: str) -> SectorStrategy:
    """Gibt passende SectorStrategy für einen Ticker"""
    sector = get_sector_for_ticker(ticker)
    
    if sector is None:
        # Fallback: "Unknown" Sektor mit default weights
        sector = "Unknown"
    
    return SectorStrategy(sector)


if __name__ == "__main__":
    # Test
    print("Testing Sector Strategies...\n")
    
    test_tickers = ["EBS.VI", "OMV.VI", "VOE.VI", "CAI.VI", "ATS.VI"]
    
    for ticker in test_tickers:
        sector = get_sector_for_ticker(ticker)
        strategy = get_strategy_for_ticker(ticker)
        
        print(f"{ticker} -> {sector}")
        print(f"  Focus Areas: {strategy.focus_areas[:2]}")
        print(f"  News Weight: {strategy.weights.news_sentiment*100:.0f}%")
        print()
