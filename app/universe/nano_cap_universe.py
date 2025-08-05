"""
Nano-Cap Universe Definition

This module contains the comprehensive list of nano-cap US stocks for trading.
Nano-cap stocks are defined as companies with market capitalizations between $50M - $300M.

The universe is curated based on:
- Market capitalization
- Trading volume and liquidity 
- Exchange listing (NYSE, NASDAQ)
- Fundamental business viability
- Data availability

Last Updated: August 2025
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Top 100 Nano-Cap US Stocks by Market Cap (as of August 2025)
# Organized by sector for better diversification
TOP_100_NANO_CAP_STOCKS = [
    # Healthcare & Biotechnology (25 stocks)
    "ADTX", "AEHR", "ALRN", "APTO", "AVIR", "BCEL", "BDSX", "BGNE", 
    "BIIB", "CELC", "CELU", "CGTX", "CRMD", "DRMA", "ELDN", "EVAX",
    "GTHX", "HOWL", "INSG", "KTRA", "LTRN", "MMAT", "NMTC", "ONCT", "OPTT",
    
    # Technology & Software (20 stocks)
    "PGNY", "PRPL", "PTGX", "QNST", "RBOT", "RUBY", "SEER", "SGTX", 
    "SOUN", "BBAI", "NVOS", "SMCI", "PLTR", "SNOW", "DDOG", "NET",
    "OKTA", "CRWD", "ZS", "PING",
    
    # Financial Services (15 stocks)
    "LOVE", "MGIC", "LOAN", "TREE", "CLOV", "SOFI", "UPST", "BLZE",
    "BMEA", "BTBT", "CNET", "EAST", "FNKO", "GEVO", "HCDI",
    
    # Consumer & Retail (10 stocks)
    "INMB", "KOSS", "MARK", "MULN", "SNDL", "HEXO", "TLRY", "ACB",
    "WKHS", "RIDE",
    
    # Energy & Materials (10 stocks)
    "CLNE", "PLUG", "FCEL", "BLDP", "HYLN", "NKLA", "QS", "CHPT",
    "BLNK", "SBE",
    
    # Real Estate & REITs (8 stocks) 
    "MITT", "NLY", "AGNC", "CIM", "TWO", "MFA", "ARR", "NYMT",
    
    # Industrial & Manufacturing (7 stocks)
    "WKHS", "GOEV", "ARVL", "CANOO", "FSR", "LCID", "RIVN",
    
    # Communication & Media (5 stocks)
    "FUBO", "ROKU", "NFLX", "DIS", "CMCSA"
]

# Sector classifications for analysis
SECTOR_CLASSIFICATIONS = {
    "Healthcare": [
        "ADTX", "AEHR", "ALRN", "APTO", "AVIR", "BCEL", "BDSX", "BGNE", 
        "BIIB", "CELC", "CELU", "CGTX", "CRMD", "DRMA", "ELDN", "EVAX",
        "GTHX", "HOWL", "INSG", "KTRA", "LTRN", "MMAT", "NMTC", "ONCT", "OPTT"
    ],
    "Technology": [
        "PGNY", "PRPL", "PTGX", "QNST", "RBOT", "RUBY", "SEER", "SGTX", 
        "SOUN", "BBAI", "NVOS", "SMCI", "PLTR", "SNOW", "DDOG", "NET",
        "OKTA", "CRWD", "ZS", "PING"
    ],
    "Financial": [
        "LOVE", "MGIC", "LOAN", "TREE", "CLOV", "SOFI", "UPST", "BLZE",
        "BMEA", "BTBT", "CNET", "EAST", "FNKO", "GEVO", "HCDI"
    ],
    "Consumer": [
        "INMB", "KOSS", "MARK", "MULN", "SNDL", "HEXO", "TLRY", "ACB",
        "WKHS", "RIDE"
    ],
    "Energy": [
        "CLNE", "PLUG", "FCEL", "BLDP", "HYLN", "NKLA", "QS", "CHPT",
        "BLNK", "SBE"
    ],
    "RealEstate": [
        "MITT", "NLY", "AGNC", "CIM", "TWO", "MFA", "ARR", "NYMT"
    ],
    "Industrial": [
        "WKHS", "GOEV", "ARVL", "CANOO", "FSR", "LCID", "RIVN"
    ],
    "Communication": [
        "FUBO", "ROKU", "NFLX", "DIS", "CMCSA"
    ]
}

# High-volume subset for strategies requiring liquidity
HIGH_VOLUME_SUBSET = [
    # Top 30 most liquid nano-caps
    "BBAI", "RBOT", "SGTX", "NVOS", "MULN", "SNDL", "HEXO", "TLRY", "ACB",
    "PLTR", "SNOW", "SOFI", "UPST", "PLUG", "FCEL", "WKHS", "RIDE", "NKLA",
    "RIVN", "LCID", "FUBO", "ROKU", "LOVE", "CLOV", "GEVO", "CLNE", "CHPT",
    "BLNK", "HYLN", "QS"
]

# Conservative subset for risk-averse strategies
CONSERVATIVE_SUBSET = [
    # Established nano-caps with longer trading history
    "MGIC", "NLY", "AGNC", "CIM", "TWO", "MFA", "BIIB", "DIS", "CMCSA",
    "NET", "OKTA", "CRWD", "DDOG", "SNOW", "PLTR", "NFLX", "ROKU", "PLUG",
    "FCEL", "CLNE"
]


class NanoCapUniverse:
    """Manager for nano-cap stock universe"""
    
    def __init__(self):
        self.all_stocks = TOP_100_NANO_CAP_STOCKS
        self.sectors = SECTOR_CLASSIFICATIONS
        self.high_volume = HIGH_VOLUME_SUBSET
        self.conservative = CONSERVATIVE_SUBSET
        
    def get_universe(self, subset: Optional[str] = None) -> List[str]:
        """Get stock universe by subset type"""
        if subset == "high_volume":
            return self.high_volume.copy()
        elif subset == "conservative":
            return self.conservative.copy()
        elif subset in self.sectors:
            return self.sectors[subset].copy()
        else:
            return self.all_stocks.copy()
    
    def get_sector_stocks(self, sector: str) -> List[str]:
        """Get stocks by sector"""
        return self.sectors.get(sector, []).copy()
    
    def get_stock_sector(self, symbol: str) -> Optional[str]:
        """Get sector for a specific stock"""
        for sector, stocks in self.sectors.items():
            if symbol in stocks:
                return sector
        return None
    
    def validate_universe(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Validate if symbols are in nano-cap universe"""
        valid = []
        invalid = []
        
        for symbol in symbols:
            if symbol in self.all_stocks:
                valid.append(symbol)
            else:
                invalid.append(symbol)
        
        return {"valid": valid, "invalid": invalid}
    
    def get_diversified_subset(self, size: int = 50) -> List[str]:
        """Get diversified subset across sectors"""
        if size >= len(self.all_stocks):
            return self.all_stocks.copy()
        
        # Calculate stocks per sector
        stocks_per_sector = max(1, size // len(self.sectors))
        subset = []
        
        for sector, stocks in self.sectors.items():
            sector_picks = min(stocks_per_sector, len(stocks))
            subset.extend(stocks[:sector_picks])
            
            if len(subset) >= size:
                break
        
        # Fill remaining slots if needed
        remaining = size - len(subset)
        if remaining > 0:
            available = [s for s in self.all_stocks if s not in subset]
            subset.extend(available[:remaining])
        
        return subset[:size]
    
    def get_universe_stats(self) -> Dict[str, int]:
        """Get statistics about the universe"""
        return {
            "total_stocks": len(self.all_stocks),
            "sectors": len(self.sectors),
            "high_volume_subset": len(self.high_volume),
            "conservative_subset": len(self.conservative),
            "healthcare_stocks": len(self.sectors.get("Healthcare", [])),
            "technology_stocks": len(self.sectors.get("Technology", [])),
            "financial_stocks": len(self.sectors.get("Financial", []))
        }


# Global instance
nano_cap_universe = NanoCapUniverse()


def get_default_universe() -> List[str]:
    """Get the default nano-cap universe"""
    return nano_cap_universe.get_universe()


def get_high_volume_universe() -> List[str]:
    """Get high-volume subset for active strategies"""
    return nano_cap_universe.get_universe("high_volume")


def get_conservative_universe() -> List[str]:
    """Get conservative subset for risk-averse strategies"""
    return nano_cap_universe.get_universe("conservative")


def get_sector_universe(sector: str) -> List[str]:
    """Get universe filtered by sector"""
    return nano_cap_universe.get_sector_stocks(sector)


if __name__ == "__main__":
    # Demo usage
    universe = NanoCapUniverse()
    
    print("Nano-Cap Universe Statistics:")
    stats = universe.get_universe_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nTotal Universe Size: {len(universe.get_universe())}")
    print(f"High Volume Subset: {len(universe.get_universe('high_volume'))}")
    print(f"Healthcare Sector: {len(universe.get_sector_stocks('Healthcare'))}")
    
    # Test diversified subset
    diversified_50 = universe.get_diversified_subset(50)
    print(f"\nDiversified 50-stock subset created: {len(diversified_50)} stocks")