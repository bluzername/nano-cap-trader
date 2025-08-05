"""
Advanced Strategy Comparison and Analysis Tools

This module provides comprehensive tools for comparing multiple trading strategies,
including statistical significance testing, risk-adjusted performance analysis,
and strategy ranking methodologies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
import warnings

from .insider_backtest_engine import BacktestResults, BacktestConfig, run_strategy_comparison
from .performance_attribution import PerformanceAttributionAnalyzer, compare_strategies_attribution

logger = logging.getLogger(__name__)


@dataclass
class ComparisonMetrics:
    """Comprehensive comparison metrics between strategies"""
    
    # Basic comparison
    return_difference: float = 0.0
    sharpe_difference: float = 0.0
    drawdown_difference: float = 0.0
    
    # Statistical significance
    return_pvalue: float = 1.0
    sharpe_pvalue: float = 1.0
    is_significant: bool = False
    confidence_level: float = 0.95
    
    # Risk-adjusted metrics
    information_ratio_diff: float = 0.0
    calmar_ratio_diff: float = 0.0
    sortino_ratio_diff: float = 0.0
    
    # Trading characteristics
    trade_frequency_diff: float = 0.0
    win_rate_diff: float = 0.0
    profit_factor_diff: float = 0.0
    
    # Robustness metrics
    consistency_score: float = 0.0
    regime_adaptability: float = 0.0
    downside_protection: float = 0.0


@dataclass
class StrategyRanking:
    """Strategy ranking results"""
    
    strategy_name: str
    overall_score: float
    rank: int
    
    # Component scores
    return_score: float = 0.0
    risk_score: float = 0.0
    consistency_score: float = 0.0
    drawdown_score: float = 0.0
    
    # Rationale
    strengths: List[str] = None
    weaknesses: List[str] = None
    recommendation: str = ""
    
    def __post_init__(self):
        if self.strengths is None:
            self.strengths = []
        if self.weaknesses is None:
            self.weaknesses = []


class StrategyComparisonEngine:
    """
    Advanced strategy comparison and ranking engine
    
    Features:
    - Statistical significance testing
    - Multi-dimensional performance analysis
    - Risk-adjusted comparisons
    - Monte Carlo robustness testing
    - Strategy ranking and recommendation
    """
    
    def __init__(self):
        self.attribution_analyzer = PerformanceAttributionAnalyzer()
        
    def compare_strategies(self, 
                          results1: BacktestResults, 
                          results2: BacktestResults,
                          strategy1_name: str = "Strategy 1",
                          strategy2_name: str = "Strategy 2") -> ComparisonMetrics:
        """
        Comprehensive comparison between two strategies
        """
        logger.info(f"Comparing {strategy1_name} vs {strategy2_name}")
        
        # Basic performance differences
        return_diff = results1.annual_return - results2.annual_return
        sharpe_diff = results1.sharpe_ratio - results2.sharpe_ratio
        drawdown_diff = results2.max_drawdown - results1.max_drawdown  # Less negative is better
        
        # Statistical significance testing
        return_pvalue, sharpe_pvalue = self._test_statistical_significance(results1, results2)
        is_significant = return_pvalue < 0.05 or sharpe_pvalue < 0.05
        
        # Risk-adjusted differences
        ir_diff = results1.information_ratio - results2.information_ratio
        calmar_diff = results1.calmar_ratio - results2.calmar_ratio
        sortino_diff = results1.sortino_ratio - results2.sortino_ratio
        
        # Trading characteristics
        freq_diff = (results1.total_trades - results2.total_trades) / max(len(results1.equity_curve), 1)
        win_rate_diff = results1.win_rate - results2.win_rate
        pf_diff = results1.profit_factor - results2.profit_factor
        
        # Robustness metrics
        consistency = self._calculate_consistency_score(results1, results2)
        regime_adapt = self._calculate_regime_adaptability(results1, results2)
        downside_prot = self._calculate_downside_protection(results1, results2)
        
        return ComparisonMetrics(
            return_difference=return_diff,
            sharpe_difference=sharpe_diff,
            drawdown_difference=drawdown_diff,
            return_pvalue=return_pvalue,
            sharpe_pvalue=sharpe_pvalue,
            is_significant=is_significant,
            information_ratio_diff=ir_diff,
            calmar_ratio_diff=calmar_diff,
            sortino_ratio_diff=sortino_diff,
            trade_frequency_diff=freq_diff,
            win_rate_diff=win_rate_diff,
            profit_factor_diff=pf_diff,
            consistency_score=consistency,
            regime_adaptability=regime_adapt,
            downside_protection=downside_prot
        )
    
    def _test_statistical_significance(self, 
                                     results1: BacktestResults, 
                                     results2: BacktestResults) -> Tuple[float, float]:
        """Test statistical significance of performance differences"""
        
        # Extract daily returns if available
        returns1 = self._extract_daily_returns(results1)
        returns2 = self._extract_daily_returns(results2)
        
        if len(returns1) < 30 or len(returns2) < 30:
            logger.warning("Insufficient data for significance testing")
            return 1.0, 1.0
        
        # Test return differences
        try:
            # Use Mann-Whitney U test (non-parametric) for robustness
            return_stat, return_pvalue = mannwhitneyu(returns1, returns2, alternative='two-sided')
        except Exception as e:
            logger.warning(f"Return significance test failed: {e}")
            return_pvalue = 1.0
        
        # Test Sharpe ratio differences using bootstrap
        try:
            sharpe_pvalue = self._bootstrap_sharpe_test(returns1, returns2)
        except Exception as e:
            logger.warning(f"Sharpe significance test failed: {e}")
            sharpe_pvalue = 1.0
        
        return return_pvalue, sharpe_pvalue
    
    def _extract_daily_returns(self, results: BacktestResults) -> np.ndarray:
        """Extract daily returns from backtest results"""
        
        if not results.equity_curve.empty:
            returns = results.equity_curve.pct_change().dropna()
            return returns.values
        
        # Fallback: simulate returns from trades
        if results.trades:
            trade_returns = [t.return_pct / 100 for t in results.trades if t.exit_date]
            if trade_returns:
                # Distribute over holding periods
                daily_returns = []
                for trade in results.trades:
                    if trade.exit_date and trade.holding_days > 0:
                        daily_return = (trade.return_pct / 100) / trade.holding_days
                        daily_returns.extend([daily_return] * trade.holding_days)
                
                return np.array(daily_returns) if daily_returns else np.array([0.0])
        
        return np.array([0.0])
    
    def _bootstrap_sharpe_test(self, returns1: np.ndarray, returns2: np.ndarray, 
                              n_bootstrap: int = 1000) -> float:
        """Bootstrap test for Sharpe ratio differences"""
        
        def sharpe_ratio(returns):
            if len(returns) == 0 or np.std(returns) == 0:
                return 0.0
            return np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Observed difference
        sharpe1 = sharpe_ratio(returns1)
        sharpe2 = sharpe_ratio(returns2)
        observed_diff = sharpe1 - sharpe2
        
        # Bootstrap distribution under null hypothesis
        combined = np.concatenate([returns1, returns2])
        n1, n2 = len(returns1), len(returns2)
        
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            # Resample under null
            resampled = np.random.choice(combined, size=len(combined), replace=True)
            boot_returns1 = resampled[:n1]
            boot_returns2 = resampled[n1:n1+n2]
            
            boot_sharpe1 = sharpe_ratio(boot_returns1)
            boot_sharpe2 = sharpe_ratio(boot_returns2)
            bootstrap_diffs.append(boot_sharpe1 - boot_sharpe2)
        
        # Calculate p-value
        bootstrap_diffs = np.array(bootstrap_diffs)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        return p_value
    
    def _calculate_consistency_score(self, results1: BacktestResults, results2: BacktestResults) -> float:
        """Calculate relative consistency score"""
        
        returns1 = self._extract_daily_returns(results1)
        returns2 = self._extract_daily_returns(results2)
        
        if len(returns1) < 10 or len(returns2) < 10:
            return 0.0
        
        # Consistency measured by coefficient of variation of monthly returns
        monthly_returns1 = self._aggregate_to_monthly(returns1)
        monthly_returns2 = self._aggregate_to_monthly(returns2)
        
        if len(monthly_returns1) < 3 or len(monthly_returns2) < 3:
            return 0.0
        
        cv1 = np.std(monthly_returns1) / (np.mean(monthly_returns1) + 1e-6)
        cv2 = np.std(monthly_returns2) / (np.mean(monthly_returns2) + 1e-6)
        
        # Lower CV is better (more consistent)
        consistency_score = (cv2 - cv1) / max(cv1, cv2, 0.01)
        
        return np.clip(consistency_score, -1.0, 1.0)
    
    def _aggregate_to_monthly(self, daily_returns: np.ndarray) -> np.ndarray:
        """Aggregate daily returns to monthly"""
        
        if len(daily_returns) < 20:
            return daily_returns
        
        # Simple aggregation: group every ~21 trading days
        n_months = len(daily_returns) // 21
        monthly_returns = []
        
        for i in range(n_months):
            start_idx = i * 21
            end_idx = min((i + 1) * 21, len(daily_returns))
            month_return = np.sum(daily_returns[start_idx:end_idx])
            monthly_returns.append(month_return)
        
        return np.array(monthly_returns)
    
    def _calculate_regime_adaptability(self, results1: BacktestResults, results2: BacktestResults) -> float:
        """Calculate relative regime adaptability"""
        
        # Simplified: compare performance stability across time
        returns1 = self._extract_daily_returns(results1)
        returns2 = self._extract_daily_returns(results2)
        
        if len(returns1) < 60 or len(returns2) < 60:
            return 0.0
        
        # Split into quarters and compare consistency
        n_periods = 4
        period_length = min(len(returns1), len(returns2)) // n_periods
        
        period_returns1 = []
        period_returns2 = []
        
        for i in range(n_periods):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length
            
            period_ret1 = np.mean(returns1[start_idx:end_idx])
            period_ret2 = np.mean(returns2[start_idx:end_idx])
            
            period_returns1.append(period_ret1)
            period_returns2.append(period_ret2)
        
        # Calculate regime adaptability as consistency across periods
        var1 = np.var(period_returns1)
        var2 = np.var(period_returns2)
        
        # Lower variance across regimes is better
        adaptability_score = (var2 - var1) / max(var1, var2, 1e-6)
        
        return np.clip(adaptability_score, -1.0, 1.0)
    
    def _calculate_downside_protection(self, results1: BacktestResults, results2: BacktestResults) -> float:
        """Calculate relative downside protection"""
        
        # Compare maximum drawdowns and downside deviation
        dd_protection = (results2.max_drawdown - results1.max_drawdown) / max(abs(results1.max_drawdown), abs(results2.max_drawdown), 0.01)
        
        # Compare Sortino ratios
        sortino_protection = (results1.sortino_ratio - results2.sortino_ratio) / max(results1.sortino_ratio, results2.sortino_ratio, 0.01)
        
        # Combine measures
        downside_score = (dd_protection + sortino_protection) / 2
        
        return np.clip(downside_score, -1.0, 1.0)
    
    def rank_strategies(self, results_dict: Dict[str, BacktestResults], 
                       weights: Optional[Dict[str, float]] = None) -> List[StrategyRanking]:
        """
        Rank strategies using multi-criteria analysis
        """
        logger.info(f"Ranking {len(results_dict)} strategies")
        
        if weights is None:
            weights = {
                'return': 0.25,
                'risk': 0.25,
                'consistency': 0.25,
                'drawdown': 0.25
            }
        
        rankings = []
        
        # Calculate normalized scores for each strategy
        return_scores = self._normalize_scores([r.annual_return for r in results_dict.values()])
        risk_scores = self._normalize_scores([r.sharpe_ratio for r in results_dict.values()])
        consistency_scores = self._calculate_consistency_scores(results_dict)
        drawdown_scores = self._normalize_scores([-r.max_drawdown for r in results_dict.values()])  # Invert drawdown
        
        for i, (strategy_name, results) in enumerate(results_dict.items()):
            # Calculate component scores
            return_score = return_scores[i]
            risk_score = risk_scores[i]
            consistency_score = consistency_scores[i]
            drawdown_score = drawdown_scores[i]
            
            # Calculate overall score
            overall_score = (
                return_score * weights['return'] +
                risk_score * weights['risk'] +
                consistency_score * weights['consistency'] +
                drawdown_score * weights['drawdown']
            )
            
            # Generate insights
            strengths, weaknesses = self._analyze_strategy_characteristics(results, strategy_name)
            recommendation = self._generate_recommendation(results, overall_score)
            
            rankings.append(StrategyRanking(
                strategy_name=strategy_name,
                overall_score=overall_score,
                rank=0,  # Will be set after sorting
                return_score=return_score,
                risk_score=risk_score,
                consistency_score=consistency_score,
                drawdown_score=drawdown_score,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendation=recommendation
            ))
        
        # Sort and assign ranks
        rankings.sort(key=lambda x: x.overall_score, reverse=True)
        for i, ranking in enumerate(rankings):
            ranking.rank = i + 1
        
        logger.info(f"Top strategy: {rankings[0].strategy_name} (score: {rankings[0].overall_score:.3f})")
        
        return rankings
    
    def _normalize_scores(self, values: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        
        if not values or len(set(values)) <= 1:
            return [0.5] * len(values)
        
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return [0.5] * len(values)
        
        return [(v - min_val) / (max_val - min_val) for v in values]
    
    def _calculate_consistency_scores(self, results_dict: Dict[str, BacktestResults]) -> List[float]:
        """Calculate consistency scores for all strategies"""
        
        consistency_values = []
        
        for results in results_dict.values():
            returns = self._extract_daily_returns(results)
            
            if len(returns) < 30:
                consistency_values.append(0.0)
                continue
            
            # Calculate consistency as negative coefficient of variation
            monthly_returns = self._aggregate_to_monthly(returns)
            
            if len(monthly_returns) < 3:
                consistency_values.append(0.0)
                continue
            
            cv = np.std(monthly_returns) / (abs(np.mean(monthly_returns)) + 1e-6)
            consistency = 1.0 / (1.0 + cv)  # Higher is better
            consistency_values.append(consistency)
        
        return self._normalize_scores(consistency_values)
    
    def _analyze_strategy_characteristics(self, results: BacktestResults, 
                                        strategy_name: str) -> Tuple[List[str], List[str]]:
        """Analyze strategy strengths and weaknesses"""
        
        strengths = []
        weaknesses = []
        
        # Return analysis
        if results.annual_return > 0.15:
            strengths.append("High absolute returns")
        elif results.annual_return < 0.05:
            weaknesses.append("Low absolute returns")
        
        # Risk-adjusted returns
        if results.sharpe_ratio > 1.0:
            strengths.append("Excellent risk-adjusted returns")
        elif results.sharpe_ratio < 0.5:
            weaknesses.append("Poor risk-adjusted returns")
        
        # Drawdown control
        if results.max_drawdown > -0.05:
            strengths.append("Strong drawdown control")
        elif results.max_drawdown < -0.20:
            weaknesses.append("High maximum drawdown")
        
        # Win rate
        if results.win_rate > 0.65:
            strengths.append("High win rate")
        elif results.win_rate < 0.45:
            weaknesses.append("Low win rate")
        
        # Trade frequency
        if results.total_trades > 50:
            strengths.append("Active trading strategy")
        elif results.total_trades < 10:
            weaknesses.append("Low trade frequency")
        
        # Strategy-specific analysis
        if 'insider' in strategy_name.lower():
            if results.insider_hit_rate > 0.6:
                strengths.append("Strong insider signal accuracy")
            elif results.insider_hit_rate < 0.4:
                weaknesses.append("Weak insider signal accuracy")
        
        return strengths, weaknesses
    
    def _generate_recommendation(self, results: BacktestResults, overall_score: float) -> str:
        """Generate strategy recommendation"""
        
        if overall_score > 0.8:
            return "STRONG BUY - Excellent performance across all metrics"
        elif overall_score > 0.6:
            return "BUY - Good performance with manageable risks"
        elif overall_score > 0.4:
            return "HOLD - Average performance, consider alternatives"
        else:
            return "AVOID - Poor performance, significant weaknesses"
    
    def generate_comparison_report(self, 
                                 rankings: List[StrategyRanking],
                                 results_dict: Dict[str, BacktestResults]) -> str:
        """Generate comprehensive comparison report"""
        
        report = """
# Strategy Comparison Report

## Executive Summary

"""
        
        # Top strategy summary
        top_strategy = rankings[0]
        report += f"""
**Recommended Strategy**: {top_strategy.strategy_name}
- **Overall Score**: {top_strategy.overall_score:.3f}/1.000
- **Recommendation**: {top_strategy.recommendation}
- **Key Strengths**: {', '.join(top_strategy.strengths[:3])}
"""
        
        # Strategy rankings table
        report += "\n## Strategy Rankings\n\n"
        report += "| Rank | Strategy | Score | Return | Risk | Consistency | Drawdown | Recommendation |\n"
        report += "|------|----------|-------|--------|------|-------------|----------|----------------|\n"
        
        for ranking in rankings:
            report += f"| {ranking.rank} | {ranking.strategy_name} | {ranking.overall_score:.3f} | "
            report += f"{ranking.return_score:.3f} | {ranking.risk_score:.3f} | "
            report += f"{ranking.consistency_score:.3f} | {ranking.drawdown_score:.3f} | "
            report += f"{ranking.recommendation.split(' - ')[0]} |\n"
        
        # Detailed performance metrics
        report += "\n## Detailed Performance Metrics\n\n"
        report += "| Strategy | Annual Return | Sharpe Ratio | Max Drawdown | Win Rate | Total Trades |\n"
        report += "|----------|---------------|--------------|--------------|----------|-------------|\n"
        
        for ranking in rankings:
            results = results_dict[ranking.strategy_name]
            report += f"| {ranking.strategy_name} | {results.annual_return:.1%} | "
            report += f"{results.sharpe_ratio:.2f} | {results.max_drawdown:.1%} | "
            report += f"{results.win_rate:.1%} | {results.total_trades} |\n"
        
        # Strategy analysis
        report += "\n## Individual Strategy Analysis\n"
        
        for ranking in rankings:
            report += f"\n### {ranking.rank}. {ranking.strategy_name}\n"
            report += f"**Overall Score**: {ranking.overall_score:.3f} | **Recommendation**: {ranking.recommendation}\n\n"
            
            if ranking.strengths:
                report += f"**Strengths**:\n"
                for strength in ranking.strengths:
                    report += f"- {strength}\n"
            
            if ranking.weaknesses:
                report += f"\n**Areas for Improvement**:\n"
                for weakness in ranking.weaknesses:
                    report += f"- {weakness}\n"
            
            report += "\n"
        
        # Implementation recommendations
        report += "\n## Implementation Recommendations\n"
        
        top_3 = rankings[:3]
        if len(top_3) > 1:
            report += f"""
### Portfolio Approach
Consider combining the top strategies for diversification:
- **Primary allocation (60%)**: {top_3[0].strategy_name}
- **Secondary allocation (30%)**: {top_3[1].strategy_name}
"""
            if len(top_3) > 2:
                report += f"- **Tactical allocation (10%)**: {top_3[2].strategy_name}\n"
        
        report += f"""
### Risk Management
- Monitor drawdowns closely for all strategies
- Consider reducing allocation during high-volatility periods
- Rebalance quarterly based on performance attribution

### Next Steps
1. **Paper trade** the top-ranked strategy for 30 days
2. **Backtest** with updated data monthly
3. **Monitor** key performance indicators daily
4. **Review** strategy rankings quarterly
"""
        
        report += "\n---\n*Report generated by NanoCap Trader Strategy Comparison Engine*"
        
        return report


# High-level convenience functions
async def compare_all_strategies(universe: List[str] = None, 
                               start_date: str = "2023-01-01",
                               end_date: str = "2024-01-01") -> str:
    """
    Compare all available strategies and return comprehensive report
    """
    if universe is None:
        universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    config = BacktestConfig(
        start_date=datetime.strptime(start_date, "%Y-%m-%d"),
        end_date=datetime.strptime(end_date, "%Y-%m-%d")
    )
    
    strategies = [
        'statistical_arbitrage',
        'momentum', 
        'mean_reversion',
        'multi_strategy',
        'insider_momentum_advanced',
        'insider_options_flow',
        'insider_ml_predictor'
    ]
    
    # Run backtests
    results_dict = await run_strategy_comparison(strategies, universe, config)
    
    # Analyze and rank
    comparison_engine = StrategyComparisonEngine()
    rankings = comparison_engine.rank_strategies(results_dict)
    
    # Generate report
    return comparison_engine.generate_comparison_report(rankings, results_dict)


def quick_compare(strategy1: str, strategy2: str, universe: List[str] = None) -> str:
    """Quick comparison between two strategies"""
    # This would be implemented with actual backtesting
    # For now, return placeholder
    return f"Comparison between {strategy1} and {strategy2} would be performed here."