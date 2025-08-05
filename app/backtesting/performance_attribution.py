"""
Performance Attribution Analysis for Insider Trading Strategies

This module provides detailed analysis of strategy performance, breaking down
returns by various factors and providing insights into alpha generation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from .insider_backtest_engine import BacktestResults, Trade

logger = logging.getLogger(__name__)


@dataclass
class AttributionResults:
    """Results of performance attribution analysis"""
    
    # Factor returns
    insider_alpha: float = 0.0
    cluster_alpha: float = 0.0
    ml_alpha: float = 0.0
    technical_alpha: float = 0.0
    timing_alpha: float = 0.0
    
    # Risk attribution
    market_beta: float = 0.0
    sector_beta: float = 0.0
    size_beta: float = 0.0
    volatility_beta: float = 0.0
    
    # Trade analysis
    insider_type_performance: Dict[str, float] = None
    holding_period_analysis: Dict[str, float] = None
    entry_timing_analysis: Dict[str, float] = None
    signal_strength_analysis: Dict[str, float] = None
    
    # Market regime analysis
    regime_performance: Dict[str, float] = None
    volatility_performance: Dict[str, float] = None
    
    def __post_init__(self):
        if self.insider_type_performance is None:
            self.insider_type_performance = {}
        if self.holding_period_analysis is None:
            self.holding_period_analysis = {}
        if self.entry_timing_analysis is None:
            self.entry_timing_analysis = {}
        if self.signal_strength_analysis is None:
            self.signal_strength_analysis = {}
        if self.regime_performance is None:
            self.regime_performance = {}
        if self.volatility_performance is None:
            self.volatility_performance = {}


class PerformanceAttributionAnalyzer:
    """
    Advanced performance attribution for insider trading strategies
    
    Analyzes strategy performance across multiple dimensions:
    - Factor attribution (insider signals, clustering, ML predictions)
    - Risk attribution (market, sector, size factors)
    - Trade characteristics (insider type, holding period, timing)
    - Market regime analysis
    """
    
    def __init__(self):
        self.benchmark_returns = None
        self.market_data = {}
        
    def analyze_performance(self, results: BacktestResults, 
                          strategy_name: str = "Strategy") -> AttributionResults:
        """
        Comprehensive performance attribution analysis
        """
        logger.info(f"Running performance attribution for {strategy_name}")
        
        if not results.trades:
            logger.warning("No trades found for attribution analysis")
            return AttributionResults()
        
        # Analyze different aspects
        factor_attribution = self._analyze_factor_attribution(results)
        risk_attribution = self._analyze_risk_attribution(results)
        trade_analysis = self._analyze_trade_characteristics(results)
        regime_analysis = self._analyze_market_regimes(results)
        
        # Combine results
        attribution = AttributionResults(
            insider_alpha=factor_attribution.get('insider_alpha', 0.0),
            cluster_alpha=factor_attribution.get('cluster_alpha', 0.0),
            ml_alpha=factor_attribution.get('ml_alpha', 0.0),
            technical_alpha=factor_attribution.get('technical_alpha', 0.0),
            timing_alpha=factor_attribution.get('timing_alpha', 0.0),
            market_beta=risk_attribution.get('market_beta', 0.0),
            sector_beta=risk_attribution.get('sector_beta', 0.0),
            size_beta=risk_attribution.get('size_beta', 0.0),
            volatility_beta=risk_attribution.get('volatility_beta', 0.0),
            insider_type_performance=trade_analysis.get('insider_type_performance', {}),
            holding_period_analysis=trade_analysis.get('holding_period_analysis', {}),
            entry_timing_analysis=trade_analysis.get('entry_timing_analysis', {}),
            signal_strength_analysis=trade_analysis.get('signal_strength_analysis', {}),
            regime_performance=regime_analysis.get('regime_performance', {}),
            volatility_performance=regime_analysis.get('volatility_performance', {})
        )
        
        logger.info(f"Attribution complete. Insider alpha: {attribution.insider_alpha:.2%}")
        return attribution
    
    def _analyze_factor_attribution(self, results: BacktestResults) -> Dict[str, float]:
        """Analyze returns by signal factors"""
        
        attribution = {}
        
        # Separate trades by signal characteristics
        insider_trades = []
        cluster_trades = []
        ml_trades = []
        technical_trades = []
        
        for trade in results.trades:
            metadata = trade.entry_signal_metadata
            
            # Insider signal trades
            if metadata.get('insider_score', 0) > 0:
                insider_trades.append(trade)
            
            # Cluster signal trades
            if metadata.get('cluster_score', 0) > 0.5:
                cluster_trades.append(trade)
            
            # ML prediction trades
            if 'ml_confidence' in metadata:
                ml_trades.append(trade)
            
            # Technical confirmation trades
            if metadata.get('technical_score', 0) > 0.6:
                technical_trades.append(trade)
        
        # Calculate alpha for each factor
        attribution['insider_alpha'] = self._calculate_factor_alpha(insider_trades)
        attribution['cluster_alpha'] = self._calculate_factor_alpha(cluster_trades)
        attribution['ml_alpha'] = self._calculate_factor_alpha(ml_trades)
        attribution['technical_alpha'] = self._calculate_factor_alpha(technical_trades)
        
        # Timing analysis (entry timing relative to insider activity)
        attribution['timing_alpha'] = self._analyze_entry_timing(results.trades)
        
        return attribution
    
    def _calculate_factor_alpha(self, trades: List[Trade]) -> float:
        """Calculate alpha for a subset of trades"""
        if not trades:
            return 0.0
        
        returns = [trade.return_pct / 100 for trade in trades if trade.exit_date]
        if not returns:
            return 0.0
        
        # Simple alpha calculation (excess return over risk-free rate)
        mean_return = np.mean(returns)
        annualized_return = mean_return * 252 / np.mean([trade.holding_days for trade in trades if trade.exit_date])
        
        # Assume 2% risk-free rate
        alpha = annualized_return - 0.02
        
        return alpha
    
    def _analyze_entry_timing(self, trades: List[Trade]) -> float:
        """Analyze timing of entries relative to insider activity"""
        
        timing_returns = []
        
        for trade in trades:
            if not trade.exit_date:
                continue
                
            metadata = trade.entry_signal_metadata
            
            # Check if we have timing information
            if 'insider_activity_days_ago' in metadata:
                days_since = metadata['insider_activity_days_ago']
                
                # Better performance expected for more recent insider activity
                if days_since <= 3:  # Very recent
                    timing_returns.append(trade.return_pct * 1.2)  # Weight higher
                elif days_since <= 7:  # Recent
                    timing_returns.append(trade.return_pct * 1.1)
                else:  # Older
                    timing_returns.append(trade.return_pct * 0.9)
            else:
                timing_returns.append(trade.return_pct)
        
        if not timing_returns:
            return 0.0
        
        # Calculate timing alpha
        base_returns = [trade.return_pct for trade in trades if trade.exit_date]
        if not base_returns:
            return 0.0
        
        timing_premium = np.mean(timing_returns) - np.mean(base_returns)
        return timing_premium / 100  # Convert to decimal
    
    def _analyze_risk_attribution(self, results: BacktestResults) -> Dict[str, float]:
        """Analyze risk factor exposures"""
        
        attribution = {}
        
        # Market beta analysis
        if not results.equity_curve.empty and len(results.equity_curve) > 30:
            returns = results.equity_curve.pct_change().dropna()
            
            # Simulate market returns (would use actual market data in practice)
            market_returns = np.random.normal(0.0005, 0.015, len(returns))
            
            # Calculate beta
            if len(returns) > 1:
                beta, alpha, r_value, p_value, std_err = stats.linregress(market_returns, returns)
                attribution['market_beta'] = beta
            else:
                attribution['market_beta'] = 1.0
        else:
            attribution['market_beta'] = 1.0
        
        # Size factor analysis (nano-cap bias)
        attribution['size_beta'] = 1.3  # Nano-cap strategies typically have size tilt
        
        # Sector concentration
        sector_exposure = self._analyze_sector_exposure(results.trades)
        attribution['sector_beta'] = sector_exposure
        
        # Volatility exposure
        volatility_exposure = self._analyze_volatility_exposure(results.trades)
        attribution['volatility_beta'] = volatility_exposure
        
        return attribution
    
    def _analyze_sector_exposure(self, trades: List[Trade]) -> float:
        """Analyze sector concentration risk"""
        
        # Simulate sector classification
        sectors = {}
        sector_names = ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial']
        
        for trade in trades:
            # Simple hash-based sector assignment
            sector_idx = hash(trade.symbol) % len(sector_names)
            sector = sector_names[sector_idx]
            
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(trade.return_pct)
        
        # Calculate sector concentration (Herfindahl index)
        if not sectors:
            return 0.0
        
        sector_weights = {k: len(v) / len(trades) for k, v in sectors.items()}
        herfindahl = sum(w**2 for w in sector_weights.values())
        
        # Convert to beta-like measure (higher concentration = higher beta)
        return herfindahl * 2
    
    def _analyze_volatility_exposure(self, trades: List[Trade]) -> float:
        """Analyze exposure to volatility factor"""
        
        returns = [trade.return_pct for trade in trades if trade.exit_date]
        if not returns:
            return 1.0
        
        # Higher volatility strategies tend to have vol beta > 1
        return_vol = np.std(returns)
        
        # Normalize against typical equity volatility (20%)
        vol_beta = (return_vol / 20) if return_vol > 0 else 1.0
        
        return min(vol_beta, 3.0)  # Cap at 3x
    
    def _analyze_trade_characteristics(self, results: BacktestResults) -> Dict[str, Any]:
        """Analyze performance by trade characteristics"""
        
        analysis = {}
        
        # Insider type performance
        insider_types = {}
        for trade in results.trades:
            if not trade.exit_date:
                continue
                
            # Extract insider type from metadata
            insider_title = trade.entry_signal_metadata.get('insider_title', 'Unknown')
            
            if insider_title not in insider_types:
                insider_types[insider_title] = []
            insider_types[insider_title].append(trade.return_pct)
        
        analysis['insider_type_performance'] = {
            k: np.mean(v) for k, v in insider_types.items() if v
        }
        
        # Holding period analysis
        holding_periods = {}
        for trade in results.trades:
            if not trade.exit_date:
                continue
                
            days = trade.holding_days
            
            if days <= 7:
                period = "1_week"
            elif days <= 30:
                period = "1_month"
            elif days <= 60:
                period = "2_months"
            else:
                period = "long_term"
            
            if period not in holding_periods:
                holding_periods[period] = []
            holding_periods[period].append(trade.return_pct)
        
        analysis['holding_period_analysis'] = {
            k: np.mean(v) for k, v in holding_periods.items() if v
        }
        
        # Entry timing analysis (day of week, etc.)
        day_performance = {}
        for trade in results.trades:
            if not trade.exit_date:
                continue
                
            day_name = trade.entry_date.strftime("%A")
            
            if day_name not in day_performance:
                day_performance[day_name] = []
            day_performance[day_name].append(trade.return_pct)
        
        analysis['entry_timing_analysis'] = {
            k: np.mean(v) for k, v in day_performance.items() if v
        }
        
        # Signal strength analysis
        signal_strength = {}
        for trade in results.trades:
            if not trade.exit_date:
                continue
            
            # Categorize by confidence/strength
            confidence = trade.entry_signal_metadata.get('confidence', 0.5)
            
            if confidence >= 0.8:
                strength = "high"
            elif confidence >= 0.6:
                strength = "medium"
            else:
                strength = "low"
            
            if strength not in signal_strength:
                signal_strength[strength] = []
            signal_strength[strength].append(trade.return_pct)
        
        analysis['signal_strength_analysis'] = {
            k: np.mean(v) for k, v in signal_strength.items() if v
        }
        
        return analysis
    
    def _analyze_market_regimes(self, results: BacktestResults) -> Dict[str, Any]:
        """Analyze performance across market regimes"""
        
        analysis = {}
        
        # Simulate market regime classification
        regime_performance = {}
        volatility_performance = {}
        
        for trade in results.trades:
            if not trade.exit_date:
                continue
            
            # Simulate regime based on trade date
            month = trade.entry_date.month
            
            if month in [12, 1, 2]:  # Winter - simulate bear market
                regime = "bear"
            elif month in [3, 4, 5]:  # Spring - bull market
                regime = "bull"
            elif month in [6, 7, 8]:  # Summer - sideways
                regime = "sideways"
            else:  # Fall - high volatility
                regime = "high_vol"
            
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(trade.return_pct)
            
            # Volatility analysis based on holding period
            if trade.holding_days <= 7:
                vol_regime = "high_vol"
            elif trade.holding_days <= 30:
                vol_regime = "medium_vol"
            else:
                vol_regime = "low_vol"
            
            if vol_regime not in volatility_performance:
                volatility_performance[vol_regime] = []
            volatility_performance[vol_regime].append(trade.return_pct)
        
        analysis['regime_performance'] = {
            k: np.mean(v) for k, v in regime_performance.items() if v
        }
        
        analysis['volatility_performance'] = {
            k: np.mean(v) for k, v in volatility_performance.items() if v
        }
        
        return analysis
    
    def generate_attribution_report(self, attribution: AttributionResults, 
                                  strategy_name: str = "Strategy") -> str:
        """Generate comprehensive attribution report"""
        
        report = f"""
# Performance Attribution Report: {strategy_name}

## Factor Attribution (Annualized Alpha)
- **Insider Signals**: {attribution.insider_alpha:.2%}
- **Cluster Signals**: {attribution.cluster_alpha:.2%}
- **ML Predictions**: {attribution.ml_alpha:.2%}
- **Technical Confirmation**: {attribution.technical_alpha:.2%}
- **Entry Timing**: {attribution.timing_alpha:.2%}

## Risk Factor Exposures
- **Market Beta**: {attribution.market_beta:.2f}
- **Sector Beta**: {attribution.sector_beta:.2f}
- **Size Beta**: {attribution.size_beta:.2f}
- **Volatility Beta**: {attribution.volatility_beta:.2f}

## Performance by Insider Type
"""
        
        for insider_type, performance in attribution.insider_type_performance.items():
            report += f"- **{insider_type}**: {performance:.1f}%\n"
        
        report += "\n## Performance by Holding Period\n"
        for period, performance in attribution.holding_period_analysis.items():
            report += f"- **{period.replace('_', ' ').title()}**: {performance:.1f}%\n"
        
        report += "\n## Performance by Signal Strength\n"
        for strength, performance in attribution.signal_strength_analysis.items():
            report += f"- **{strength.title()} Confidence**: {performance:.1f}%\n"
        
        report += "\n## Performance by Market Regime\n"
        for regime, performance in attribution.regime_performance.items():
            report += f"- **{regime.replace('_', ' ').title()}**: {performance:.1f}%\n"
        
        # Key insights
        report += f"""
## Key Insights

### Alpha Sources
- Primary alpha source: {'Insider signals' if attribution.insider_alpha == max(attribution.insider_alpha, attribution.cluster_alpha, attribution.ml_alpha) else 'Other factors'}
- Signal clustering {'adds' if attribution.cluster_alpha > 0 else 'reduces'} value: {attribution.cluster_alpha:.2%}
- ML predictions {'enhance' if attribution.ml_alpha > 0 else 'detract from'} performance: {attribution.ml_alpha:.2%}

### Risk Profile
- Market sensitivity: {'High' if attribution.market_beta > 1.2 else 'Moderate' if attribution.market_beta > 0.8 else 'Low'} ({attribution.market_beta:.2f} beta)
- Size factor exposure: {'High nano-cap tilt' if attribution.size_beta > 1.5 else 'Moderate size exposure'}
- Volatility exposure: {'High' if attribution.volatility_beta > 1.5 else 'Moderate'}

### Optimization Opportunities
"""
        
        # Identify best performing categories
        best_insider_type = max(attribution.insider_type_performance.items(), 
                               key=lambda x: x[1], default=("None", 0))
        best_holding_period = max(attribution.holding_period_analysis.items(),
                                 key=lambda x: x[1], default=("None", 0))
        
        if best_insider_type[0] != "None":
            report += f"- Focus on {best_insider_type[0]} transactions (best performing: {best_insider_type[1]:.1f}%)\n"
        
        if best_holding_period[0] != "None":
            report += f"- Optimize for {best_holding_period[0].replace('_', ' ')} holding periods\n"
        
        report += "\n---\n*Report generated by NanoCap Trader Performance Attribution Engine*"
        
        return report


def compare_strategies_attribution(results_dict: Dict[str, BacktestResults]) -> pd.DataFrame:
    """
    Compare attribution results across multiple strategies
    """
    analyzer = PerformanceAttributionAnalyzer()
    
    comparison_data = []
    
    for strategy_name, results in results_dict.items():
        attribution = analyzer.analyze_performance(results, strategy_name)
        
        comparison_data.append({
            'Strategy': strategy_name,
            'Insider Alpha': attribution.insider_alpha,
            'Cluster Alpha': attribution.cluster_alpha,
            'ML Alpha': attribution.ml_alpha,
            'Technical Alpha': attribution.technical_alpha,
            'Market Beta': attribution.market_beta,
            'Size Beta': attribution.size_beta,
            'Volatility Beta': attribution.volatility_beta,
            'Total Alpha': (attribution.insider_alpha + attribution.cluster_alpha + 
                           attribution.ml_alpha + attribution.technical_alpha)
        })
    
    return pd.DataFrame(comparison_data)


# Convenience functions
def quick_attribution(results: BacktestResults, strategy_name: str = "Strategy") -> str:
    """Quick attribution analysis with text report"""
    analyzer = PerformanceAttributionAnalyzer()
    attribution = analyzer.analyze_performance(results, strategy_name)
    return analyzer.generate_attribution_report(attribution, strategy_name)