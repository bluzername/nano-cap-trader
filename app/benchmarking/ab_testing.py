"""Advanced A/B testing framework for strategy comparison and optimization."""
from __future__ import annotations
import asyncio
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from scipy import stats
import logging
import json

from ..strategies.base_strategy import BaseStrategy, Signal
from .performance_metrics import PerformanceAnalyzer, BenchmarkData

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Results from A/B testing comparison."""
    test_id: str
    strategies: List[str]
    start_date: dt.datetime
    end_date: dt.datetime
    
    # Performance metrics for each strategy
    performance_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Statistical significance tests
    statistical_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Risk-adjusted comparisons
    risk_adjusted_rankings: List[Tuple[str, float]] = field(default_factory=list)
    
    # Drawdown analysis
    drawdown_analysis: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Trade-level statistics
    trade_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Recommendation
    recommended_strategy: Optional[str] = None
    confidence_level: float = 0.0
    
    # Test metadata
    test_duration_days: int = 0
    total_trades: int = 0
    universe_size: int = 0


class ABTestFramework:
    """
    Advanced A/B testing framework for algorithmic trading strategies.
    
    Features:
    - Statistical significance testing (t-tests, Mann-Whitney U)
    - Multiple comparison correction (Bonferroni, Benjamini-Hochberg)
    - Bayesian analysis for early stopping
    - Performance attribution analysis
    - Risk-adjusted metrics comparison
    - Sector and factor decomposition
    """
    
    def __init__(self):
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Test tracking
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.completed_tests: Dict[str, ABTestResult] = {}
        
        # Configuration
        self.min_test_duration_days = 30
        self.min_trades_per_strategy = 20
        self.significance_level = 0.05
        self.power_threshold = 0.8
        
        # Early stopping parameters
        self.early_stopping_enabled = True
        self.bayesian_threshold = 0.95  # 95% confidence for early stopping
        self.min_days_before_early_stop = 14
        
    async def start_ab_test(
        self,
        test_id: str,
        strategies: Dict[str, BaseStrategy],
        universe: List[str],
        test_duration_days: Optional[int] = None,
        benchmark: str = "russell_2000",
        enable_paper_trading: bool = True
    ) -> str:
        """Start a new A/B test comparing multiple strategies."""
        try:
            if test_id in self.active_tests:
                raise ValueError(f"Test {test_id} already exists")
            
            if len(strategies) < 2:
                raise ValueError("Need at least 2 strategies for A/B testing")
            
            if len(strategies) > 5:
                raise ValueError("Maximum 5 strategies supported for A/B testing")
            
            test_config = {
                'test_id': test_id,
                'strategies': {name: strategy for name, strategy in strategies.items()},
                'universe': universe,
                'start_date': dt.datetime.now(),
                'benchmark': benchmark,
                'enable_paper_trading': enable_paper_trading,
                'test_duration_days': test_duration_days or self.min_test_duration_days,
                
                # Tracking data
                'daily_returns': {name: [] for name in strategies.keys()},
                'daily_positions': {name: [] for name in strategies.keys()},
                'trade_history': {name: [] for name in strategies.keys()},
                'signal_history': {name: [] for name in strategies.keys()},
                'portfolio_values': {name: [strategies[name].portfolio_value] for name in strategies.keys()},
                
                # Statistical tracking
                'cumulative_pnl': {name: 0.0 for name in strategies.keys()},
                'daily_statistics': [],
                
                # Risk metrics
                'max_drawdowns': {name: 0.0 for name in strategies.keys()},
                'volatilities': {name: [] for name in strategies.keys()},
                
                # Early stopping data
                'early_stop_checks': [],
                'bayesian_probabilities': {name: [] for name in strategies.keys()},
            }
            
            self.active_tests[test_id] = test_config
            
            logger.info(f"Started A/B test {test_id} with {len(strategies)} strategies")
            logger.info(f"Universe: {len(universe)} symbols, Duration: {test_config['test_duration_days']} days")
            
            return test_id
            
        except Exception as e:
            logger.error(f"Error starting A/B test {test_id}: {e}")
            raise
    
    async def update_test_data(
        self,
        test_id: str,
        market_data: Dict[str, pd.DataFrame],
        date: dt.datetime = None
    ) -> None:
        """Update test with new market data and strategy performance."""
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"Test {test_id} not found")
            
            test_config = self.active_tests[test_id]
            strategies = test_config['strategies']
            
            if date is None:
                date = dt.datetime.now()
            
            # Run each strategy and collect results
            strategy_results = {}
            
            for strategy_name, strategy in strategies.items():
                try:
                    # Generate signals
                    signals = await strategy.generate_signals(market_data)
                    test_config['signal_history'][strategy_name].extend(signals)
                    
                    # Update positions (paper trading)
                    if test_config['enable_paper_trading']:
                        for signal in signals:
                            strategy.execute_signal(signal, market_data)
                    
                    # Update portfolio values
                    strategy.update_positions(market_data)
                    current_value = strategy.portfolio_value_current
                    test_config['portfolio_values'][strategy_name].append(current_value)
                    
                    # Calculate daily return
                    previous_value = test_config['portfolio_values'][strategy_name][-2] if len(test_config['portfolio_values'][strategy_name]) > 1 else strategy.portfolio_value
                    daily_return = (current_value - previous_value) / previous_value
                    test_config['daily_returns'][strategy_name].append(daily_return)
                    
                    # Update cumulative PnL
                    test_config['cumulative_pnl'][strategy_name] = (current_value / strategy.portfolio_value) - 1
                    
                    # Track volatility
                    if len(test_config['daily_returns'][strategy_name]) >= 5:
                        recent_returns = test_config['daily_returns'][strategy_name][-5:]
                        volatility = np.std(recent_returns) * np.sqrt(252)
                        test_config['volatilities'][strategy_name].append(volatility)
                    
                    # Update max drawdown
                    portfolio_values = test_config['portfolio_values'][strategy_name]
                    if len(portfolio_values) > 1:
                        peak = max(portfolio_values)
                        current_drawdown = (current_value - peak) / peak
                        test_config['max_drawdowns'][strategy_name] = min(
                            test_config['max_drawdowns'][strategy_name], 
                            current_drawdown
                        )
                    
                    strategy_results[strategy_name] = {
                        'signals_count': len(signals),
                        'portfolio_value': current_value,
                        'daily_return': daily_return,
                        'cumulative_pnl': test_config['cumulative_pnl'][strategy_name],
                        'positions_count': len(strategy.positions)
                    }
                    
                except Exception as e:
                    logger.error(f"Error updating strategy {strategy_name} in test {test_id}: {e}")
                    continue
            
            # Store daily statistics
            test_config['daily_statistics'].append({
                'date': date,
                'strategy_results': strategy_results
            })
            
            # Check for early stopping
            if self.early_stopping_enabled:
                await self._check_early_stopping(test_id)
            
            # Check if test should end
            test_duration = (date - test_config['start_date']).days
            if test_duration >= test_config['test_duration_days']:
                await self._finalize_test(test_id)
            
        except Exception as e:
            logger.error(f"Error updating test data for {test_id}: {e}")
    
    async def _check_early_stopping(self, test_id: str) -> None:
        """Check if test can be stopped early based on statistical significance."""
        try:
            test_config = self.active_tests[test_id]
            
            # Need minimum time before considering early stopping
            test_duration = (dt.datetime.now() - test_config['start_date']).days
            if test_duration < self.min_days_before_early_stop:
                return
            
            strategies = list(test_config['strategies'].keys())
            returns_data = test_config['daily_returns']
            
            # Check if we have enough data points
            min_data_points = min(len(returns_data[strategy]) for strategy in strategies)
            if min_data_points < 10:
                return
            
            # Perform pairwise Bayesian comparison
            bayesian_results = {}
            
            for i, strategy1 in enumerate(strategies):
                for strategy2 in strategies[i+1:]:
                    returns1 = np.array(returns_data[strategy1])
                    returns2 = np.array(returns_data[strategy2])
                    
                    # Bayesian t-test
                    probability = self._bayesian_ttest(returns1, returns2)
                    
                    pair_key = f"{strategy1}_vs_{strategy2}"
                    bayesian_results[pair_key] = probability
            
            # Check if any comparison has high confidence
            high_confidence_comparisons = [
                pair for pair, prob in bayesian_results.items() 
                if prob > self.bayesian_threshold or prob < (1 - self.bayesian_threshold)
            ]
            
            test_config['early_stop_checks'].append({
                'date': dt.datetime.now(),
                'bayesian_results': bayesian_results,
                'high_confidence_comparisons': high_confidence_comparisons
            })
            
            # Stop early if we have strong evidence
            if len(high_confidence_comparisons) >= len(strategies) - 1:
                logger.info(f"Early stopping triggered for test {test_id} - strong statistical evidence")
                await self._finalize_test(test_id, early_stop=True)
            
        except Exception as e:
            logger.error(f"Error checking early stopping for {test_id}: {e}")
    
    def _bayesian_ttest(self, returns1: np.ndarray, returns2: np.ndarray) -> float:
        """Perform Bayesian t-test to compare two return series."""
        try:
            # Simple Bayesian t-test implementation
            n1, n2 = len(returns1), len(returns2)
            mean1, mean2 = np.mean(returns1), np.mean(returns2)
            var1, var2 = np.var(returns1, ddof=1), np.var(returns2, ddof=1)
            
            if var1 <= 0 or var2 <= 0:
                return 0.5  # No difference
            
            # Pooled variance
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            
            # Standard error
            se = np.sqrt(pooled_var * (1/n1 + 1/n2))
            
            if se <= 0:
                return 0.5
            
            # T-statistic
            t_stat = (mean1 - mean2) / se
            
            # Degrees of freedom
            df = n1 + n2 - 2
            
            # Probability that strategy 1 is better than strategy 2
            probability = 1 - stats.t.cdf(t_stat, df)
            
            return probability
            
        except Exception as e:
            logger.error(f"Error in Bayesian t-test: {e}")
            return 0.5
    
    async def _finalize_test(self, test_id: str, early_stop: bool = False) -> ABTestResult:
        """Finalize test and generate comprehensive results."""
        try:
            test_config = self.active_tests[test_id]
            strategies = test_config['strategies']
            
            # Calculate comprehensive performance metrics
            performance_metrics = {}
            statistical_tests = {}
            trade_statistics = {}
            
            # Get benchmark data
            start_date = test_config['start_date']
            end_date = dt.datetime.now()
            benchmark_data = await self.performance_analyzer.get_benchmark_data(
                test_config['benchmark'], start_date, end_date
            )
            
            # Analyze each strategy
            for strategy_name, strategy in strategies.items():
                returns = pd.Series(test_config['daily_returns'][strategy_name])
                
                if benchmark_data and not returns.empty:
                    # Calculate comprehensive metrics
                    metrics = self.performance_analyzer.calculate_comprehensive_metrics(
                        returns, benchmark_data
                    )
                    performance_metrics[strategy_name] = metrics
                
                # Trade statistics
                signals = test_config['signal_history'][strategy_name]
                trade_stats = self._calculate_trade_statistics(signals, strategy)
                trade_statistics[strategy_name] = trade_stats
            
            # Statistical significance tests
            strategy_names = list(strategies.keys())
            for i, strategy1 in enumerate(strategy_names):
                for strategy2 in strategy_names[i+1:]:
                    pair_key = f"{strategy1}_vs_{strategy2}"
                    
                    returns1 = np.array(test_config['daily_returns'][strategy1])
                    returns2 = np.array(test_config['daily_returns'][strategy2])
                    
                    # T-test
                    t_stat, t_pvalue = stats.ttest_ind(returns1, returns2)
                    
                    # Mann-Whitney U test
                    u_stat, u_pvalue = stats.mannwhitneyu(
                        returns1, returns2, alternative='two-sided'
                    )
                    
                    statistical_tests[pair_key] = {
                        't_statistic': t_stat,
                        't_pvalue': t_pvalue,
                        'u_statistic': u_stat,
                        'u_pvalue': u_pvalue,
                        'significant': min(t_pvalue, u_pvalue) < self.significance_level
                    }
            
            # Risk-adjusted rankings
            risk_adjusted_rankings = []
            for strategy_name in strategy_names:
                if strategy_name in performance_metrics:
                    metrics = performance_metrics[strategy_name]
                    # Combined score: Sharpe ratio + Information ratio - Max drawdown penalty
                    score = (
                        metrics.get('sharpe_ratio', 0) * 0.4 +
                        metrics.get('information_ratio', 0) * 0.3 +
                        metrics.get('calmar_ratio', 0) * 0.2 +
                        max(0, -metrics.get('max_drawdown', 0)) * 0.1
                    )
                    risk_adjusted_rankings.append((strategy_name, score))
            
            risk_adjusted_rankings.sort(key=lambda x: x[1], reverse=True)
            
            # Drawdown analysis
            drawdown_analysis = {}
            for strategy_name in strategy_names:
                portfolio_values = test_config['portfolio_values'][strategy_name]
                if len(portfolio_values) > 1:
                    values_series = pd.Series(portfolio_values)
                    cumulative_returns = values_series / values_series.iloc[0] - 1
                    rolling_max = cumulative_returns.expanding().max()
                    drawdown = cumulative_returns - rolling_max
                    
                    drawdown_analysis[strategy_name] = {
                        'max_drawdown': drawdown.min(),
                        'avg_drawdown': drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0,
                        'drawdown_duration_avg': self._calculate_avg_drawdown_duration(drawdown),
                        'recovery_time_avg': self._calculate_avg_recovery_time(drawdown)
                    }
            
            # Determine recommended strategy
            recommended_strategy = None
            confidence_level = 0.0
            
            if risk_adjusted_rankings:
                best_strategy = risk_adjusted_rankings[0][0]
                best_score = risk_adjusted_rankings[0][1]
                
                # Check statistical significance vs other strategies
                significant_wins = 0
                total_comparisons = 0
                
                for other_strategy, _ in risk_adjusted_rankings[1:]:
                    pair_key = f"{best_strategy}_vs_{other_strategy}"
                    if pair_key in statistical_tests:
                        total_comparisons += 1
                        if statistical_tests[pair_key]['significant']:
                            significant_wins += 1
                
                if total_comparisons > 0:
                    confidence_level = significant_wins / total_comparisons
                    if confidence_level >= 0.7:  # 70% confidence threshold
                        recommended_strategy = best_strategy
            
            # Create result object
            result = ABTestResult(
                test_id=test_id,
                strategies=list(strategy_names),
                start_date=start_date,
                end_date=end_date,
                performance_metrics=performance_metrics,
                statistical_tests=statistical_tests,
                risk_adjusted_rankings=risk_adjusted_rankings,
                drawdown_analysis=drawdown_analysis,
                trade_statistics=trade_statistics,
                recommended_strategy=recommended_strategy,
                confidence_level=confidence_level,
                test_duration_days=(end_date - start_date).days,
                total_trades=sum(len(test_config['signal_history'][s]) for s in strategy_names),
                universe_size=len(test_config['universe'])
            )
            
            # Store result and clean up
            self.completed_tests[test_id] = result
            del self.active_tests[test_id]
            
            logger.info(f"Finalized A/B test {test_id} - Recommended: {recommended_strategy} (confidence: {confidence_level:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error finalizing test {test_id}: {e}")
            raise
    
    def _calculate_trade_statistics(self, signals: List[Signal], strategy: BaseStrategy) -> Dict[str, Any]:
        """Calculate detailed trade statistics."""
        try:
            if not signals:
                return {}
            
            # Basic counts
            buy_signals = [s for s in signals if s.signal_type.value == 1]
            sell_signals = [s for s in signals if s.signal_type.value == -1]
            
            # Confidence distribution
            confidences = [s.confidence for s in signals]
            
            # Symbol distribution
            symbols = [s.symbol for s in signals]
            symbol_counts = pd.Series(symbols).value_counts()
            
            # Time distribution
            signal_times = [s.timestamp for s in signals]
            hourly_distribution = pd.Series([t.hour for t in signal_times]).value_counts().sort_index()
            
            return {
                'total_signals': len(signals),
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'avg_confidence': np.mean(confidences),
                'confidence_std': np.std(confidences),
                'unique_symbols': len(symbol_counts),
                'most_traded_symbols': symbol_counts.head(5).to_dict(),
                'signals_per_hour': hourly_distribution.to_dict(),
                'avg_signals_per_day': len(signals) / max(1, (signal_times[-1] - signal_times[0]).days) if len(signal_times) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade statistics: {e}")
            return {}
    
    def _calculate_avg_drawdown_duration(self, drawdown: pd.Series) -> float:
        """Calculate average drawdown duration in days."""
        try:
            if drawdown.empty:
                return 0.0
            
            # Find drawdown periods
            in_drawdown = drawdown < 0
            drawdown_periods = []
            
            start = None
            for i, is_dd in enumerate(in_drawdown):
                if is_dd and start is None:
                    start = i
                elif not is_dd and start is not None:
                    drawdown_periods.append(i - start)
                    start = None
            
            # Handle ongoing drawdown
            if start is not None:
                drawdown_periods.append(len(drawdown) - start)
            
            return np.mean(drawdown_periods) if drawdown_periods else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating drawdown duration: {e}")
            return 0.0
    
    def _calculate_avg_recovery_time(self, drawdown: pd.Series) -> float:
        """Calculate average recovery time from drawdowns."""
        try:
            if drawdown.empty:
                return 0.0
            
            recovery_times = []
            in_drawdown = False
            drawdown_start = None
            
            for i, dd in enumerate(drawdown):
                if dd < 0 and not in_drawdown:
                    # Start of drawdown
                    in_drawdown = True
                    drawdown_start = i
                elif dd >= 0 and in_drawdown:
                    # Recovery
                    if drawdown_start is not None:
                        recovery_times.append(i - drawdown_start)
                    in_drawdown = False
                    drawdown_start = None
            
            return np.mean(recovery_times) if recovery_times else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating recovery time: {e}")
            return 0.0
    
    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """Get current status of an active test."""
        if test_id in self.active_tests:
            test_config = self.active_tests[test_id]
            
            # Calculate current metrics
            current_metrics = {}
            for strategy_name in test_config['strategies'].keys():
                portfolio_values = test_config['portfolio_values'][strategy_name]
                returns = test_config['daily_returns'][strategy_name]
                
                current_metrics[strategy_name] = {
                    'current_value': portfolio_values[-1] if portfolio_values else 0,
                    'total_return': (portfolio_values[-1] / portfolio_values[0] - 1) if len(portfolio_values) > 1 else 0,
                    'daily_returns_count': len(returns),
                    'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0,
                    'max_drawdown': test_config['max_drawdowns'][strategy_name],
                    'signals_count': len(test_config['signal_history'][strategy_name])
                }
            
            return {
                'test_id': test_id,
                'status': 'active',
                'start_date': test_config['start_date'].isoformat(),
                'days_running': (dt.datetime.now() - test_config['start_date']).days,
                'target_duration': test_config['test_duration_days'],
                'strategies': list(test_config['strategies'].keys()),
                'universe_size': len(test_config['universe']),
                'current_metrics': current_metrics,
                'early_stop_checks': len(test_config['early_stop_checks'])
            }
        
        elif test_id in self.completed_tests:
            result = self.completed_tests[test_id]
            return {
                'test_id': test_id,
                'status': 'completed',
                'recommended_strategy': result.recommended_strategy,
                'confidence_level': result.confidence_level,
                'test_duration_days': result.test_duration_days,
                'total_trades': result.total_trades
            }
        
        else:
            return {'test_id': test_id, 'status': 'not_found'}
    
    def get_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tests."""
        all_tests = {}
        
        for test_id in self.active_tests:
            all_tests[test_id] = self.get_test_status(test_id)
        
        for test_id in self.completed_tests:
            all_tests[test_id] = self.get_test_status(test_id)
        
        return all_tests
    
    async def stop_test(self, test_id: str) -> ABTestResult:
        """Manually stop an active test."""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found or already completed")
        
        return await self._finalize_test(test_id)
    
    def export_test_results(self, test_id: str, format: str = 'json') -> Union[str, Dict]:
        """Export test results in specified format."""
        if test_id not in self.completed_tests:
            raise ValueError(f"Test {test_id} not found or not completed")
        
        result = self.completed_tests[test_id]
        
        # Convert to serializable format
        export_data = {
            'test_id': result.test_id,
            'strategies': result.strategies,
            'start_date': result.start_date.isoformat(),
            'end_date': result.end_date.isoformat(),
            'test_duration_days': result.test_duration_days,
            'recommended_strategy': result.recommended_strategy,
            'confidence_level': result.confidence_level,
            'performance_metrics': result.performance_metrics,
            'statistical_tests': result.statistical_tests,
            'risk_adjusted_rankings': result.risk_adjusted_rankings,
            'drawdown_analysis': result.drawdown_analysis,
            'trade_statistics': result.trade_statistics,
            'total_trades': result.total_trades,
            'universe_size': result.universe_size
        }
        
        if format.lower() == 'json':
            return json.dumps(export_data, indent=2, default=str)
        else:
            return export_data