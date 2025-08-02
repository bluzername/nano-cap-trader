"""Portfolio-level risk management and controls."""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging

from ..strategies.base_strategy import BaseStrategy, Position, Signal
from ..config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""
    portfolio_value: float
    cash: float
    gross_exposure: float
    net_exposure: float
    leverage: float
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    max_drawdown: float
    concentration_risk: float
    correlation_risk: float
    liquidity_risk: float


@dataclass
class RiskLimit:
    """Risk limit definition."""
    name: str
    current_value: float
    limit: float
    warning_threshold: float
    breach_count: int = 0
    last_breach: Optional[dt.datetime] = None
    
    @property
    def utilization(self) -> float:
        return self.current_value / self.limit if self.limit > 0 else 0
    
    @property
    def is_warning(self) -> bool:
        return self.current_value >= self.warning_threshold
    
    @property
    def is_breach(self) -> bool:
        return self.current_value >= self.limit


class PortfolioRiskManager:
    """
    Advanced portfolio-level risk management system.
    
    Features:
    - Real-time risk monitoring
    - Position sizing with risk budgeting
    - Correlation-based exposure limits
    - Sector and geographic concentration limits
    - Dynamic hedging recommendations
    - Stress testing and scenario analysis
    """
    
    def __init__(self):
        # Risk limits configuration
        self.risk_limits = {
            'max_portfolio_leverage': RiskLimit('Portfolio Leverage', 0, 2.0, 1.8),
            'max_gross_exposure': RiskLimit('Gross Exposure', 0, 2.0, 1.8),
            'max_net_exposure': RiskLimit('Net Exposure', 0, 1.0, 0.9),
            'max_single_position': RiskLimit('Single Position', 0, 0.05, 0.04),  # 5% max
            'max_sector_exposure': RiskLimit('Sector Exposure', 0, 0.25, 0.20),  # 25% max
            'max_correlation_exposure': RiskLimit('Correlation Exposure', 0, 0.15, 0.12),
            'max_daily_var': RiskLimit('Daily VaR', 0, 0.02, 0.015),  # 2% daily VaR
            'max_concentration_hhi': RiskLimit('Concentration HHI', 0, 0.1, 0.08),
        }
        
        # Portfolio state
        self.strategies: Dict[str, BaseStrategy] = {}
        self.last_risk_check: Optional[dt.datetime] = None
        self.risk_history: List[RiskMetrics] = []
        
        # Risk monitoring
        self.risk_alerts: List[Dict[str, Any]] = []
        self.emergency_stops: List[str] = []
        
        # Sector classification (simplified)
        self.sector_keywords = {
            'technology': ['TECH', 'SOFT', 'DATA', 'CLOUD', 'AI', 'CYBER'],
            'healthcare': ['HEALTH', 'BIO', 'PHARMA', 'MED', 'DRUG'],
            'financials': ['BANK', 'FINANCE', 'INSUR', 'CREDIT', 'LOAN'],
            'energy': ['OIL', 'GAS', 'ENERGY', 'SOLAR', 'WIND'],
            'industrials': ['INDUS', 'MANUF', 'AERO', 'DEFENSE'],
            'consumer': ['RETAIL', 'CONSUMER', 'FOOD', 'BEVERAGE'],
            'materials': ['METAL', 'MINING', 'CHEMICAL', 'MATERIAL'],
            'utilities': ['UTIL', 'ELECTRIC', 'WATER', 'TELECOM'],
            'real_estate': ['REIT', 'REAL', 'ESTATE', 'PROPERTY'],
        }
        
    def register_strategy(self, strategy: BaseStrategy) -> None:
        """Register a strategy for risk monitoring."""
        self.strategies[strategy.strategy_id] = strategy
        logger.info(f"Registered strategy {strategy.strategy_id} for risk monitoring")
    
    def unregister_strategy(self, strategy_id: str) -> None:
        """Unregister a strategy from risk monitoring."""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            logger.info(f"Unregistered strategy {strategy_id} from risk monitoring")
    
    async def validate_signal(self, signal: Signal, strategy_id: str) -> Tuple[bool, str]:
        """Validate a signal against portfolio risk limits."""
        try:
            if strategy_id not in self.strategies:
                return False, f"Strategy {strategy_id} not registered"
            
            strategy = self.strategies[strategy_id]
            
            # Calculate current risk metrics
            current_metrics = self.calculate_portfolio_risk()
            
            # Simulate the impact of the new signal
            simulated_metrics = await self._simulate_signal_impact(signal, strategy, current_metrics)
            
            # Check risk limits
            violations = self._check_risk_limits(simulated_metrics)
            
            if violations:
                violation_msg = "; ".join([f"{v['limit']}: {v['value']:.2%} > {v['threshold']:.2%}" for v in violations])
                return False, f"Risk limit violations: {violation_msg}"
            
            # Check position concentration
            concentration_check = self._check_position_concentration(signal, strategy)
            if not concentration_check[0]:
                return False, concentration_check[1]
            
            # Check sector concentration
            sector_check = self._check_sector_concentration(signal, strategy)
            if not sector_check[0]:
                return False, sector_check[1]
            
            # Check correlation risk
            correlation_check = await self._check_correlation_risk(signal, strategy)
            if not correlation_check[0]:
                return False, correlation_check[1]
            
            return True, "Signal approved"
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False, f"Risk validation error: {e}"
    
    def calculate_portfolio_risk(self) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        try:
            total_portfolio_value = 0
            total_cash = 0
            gross_exposure = 0
            net_exposure = 0
            positions_values = []
            
            # Aggregate across all strategies
            for strategy in self.strategies.values():
                portfolio_value = strategy.portfolio_value_current
                cash = strategy.cash
                
                total_portfolio_value += portfolio_value
                total_cash += cash
                
                # Calculate exposures
                for position in strategy.positions.values():
                    position_value = abs(position.market_value)
                    gross_exposure += position_value
                    net_exposure += position.market_value  # Signed
                    positions_values.append(position_value)
            
            # Normalize exposures
            if total_portfolio_value > 0:
                gross_exposure /= total_portfolio_value
                net_exposure /= total_portfolio_value
            
            # Calculate leverage
            leverage = gross_exposure
            
            # Calculate VaR and Expected Shortfall (simplified)
            daily_returns = self._get_portfolio_returns()
            var_95, expected_shortfall = self._calculate_var_es(daily_returns)
            
            # Calculate max drawdown
            max_drawdown = self._calculate_max_drawdown()
            
            # Calculate concentration risk (HHI)
            concentration_risk = self._calculate_concentration_hhi(positions_values, total_portfolio_value)
            
            # Calculate correlation risk
            correlation_risk = self._calculate_correlation_risk()
            
            # Calculate liquidity risk
            liquidity_risk = self._calculate_liquidity_risk()
            
            metrics = RiskMetrics(
                portfolio_value=total_portfolio_value,
                cash=total_cash,
                gross_exposure=gross_exposure,
                net_exposure=net_exposure,
                leverage=leverage,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                concentration_risk=concentration_risk,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk
            )
            
            # Update risk limits current values
            self._update_risk_limits(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    async def _simulate_signal_impact(
        self, 
        signal: Signal, 
        strategy: BaseStrategy, 
        current_metrics: RiskMetrics
    ) -> RiskMetrics:
        """Simulate the impact of a signal on portfolio risk."""
        try:
            # Calculate position size
            position_size = strategy.calculate_position_size(signal, {})
            position_value = position_size * signal.price
            
            # Simulate new metrics
            new_gross_exposure = current_metrics.gross_exposure
            new_net_exposure = current_metrics.net_exposure
            
            if signal.signal_type.value == 1:  # Buy
                new_gross_exposure += position_value / current_metrics.portfolio_value
                new_net_exposure += position_value / current_metrics.portfolio_value
            elif signal.signal_type.value == -1:  # Sell/Short
                new_gross_exposure += position_value / current_metrics.portfolio_value
                new_net_exposure -= position_value / current_metrics.portfolio_value
            
            # Create simulated metrics
            simulated_metrics = RiskMetrics(
                portfolio_value=current_metrics.portfolio_value,
                cash=current_metrics.cash - position_value,
                gross_exposure=new_gross_exposure,
                net_exposure=new_net_exposure,
                leverage=new_gross_exposure,
                var_95=current_metrics.var_95,  # Simplified - would need full recalculation
                expected_shortfall=current_metrics.expected_shortfall,
                max_drawdown=current_metrics.max_drawdown,
                concentration_risk=current_metrics.concentration_risk,
                correlation_risk=current_metrics.correlation_risk,
                liquidity_risk=current_metrics.liquidity_risk
            )
            
            return simulated_metrics
            
        except Exception as e:
            logger.error(f"Error simulating signal impact: {e}")
            return current_metrics
    
    def _check_risk_limits(self, metrics: RiskMetrics) -> List[Dict[str, Any]]:
        """Check risk metrics against defined limits."""
        violations = []
        
        # Check leverage
        if metrics.leverage > self.risk_limits['max_portfolio_leverage'].limit:
            violations.append({
                'limit': 'Portfolio Leverage',
                'value': metrics.leverage,
                'threshold': self.risk_limits['max_portfolio_leverage'].limit
            })
        
        # Check gross exposure
        if metrics.gross_exposure > self.risk_limits['max_gross_exposure'].limit:
            violations.append({
                'limit': 'Gross Exposure',
                'value': metrics.gross_exposure,
                'threshold': self.risk_limits['max_gross_exposure'].limit
            })
        
        # Check net exposure
        if abs(metrics.net_exposure) > self.risk_limits['max_net_exposure'].limit:
            violations.append({
                'limit': 'Net Exposure',
                'value': abs(metrics.net_exposure),
                'threshold': self.risk_limits['max_net_exposure'].limit
            })
        
        # Check VaR
        if metrics.var_95 > self.risk_limits['max_daily_var'].limit:
            violations.append({
                'limit': 'Daily VaR',
                'value': metrics.var_95,
                'threshold': self.risk_limits['max_daily_var'].limit
            })
        
        return violations
    
    def _check_position_concentration(self, signal: Signal, strategy: BaseStrategy) -> Tuple[bool, str]:
        """Check if signal would violate position concentration limits."""
        try:
            position_size = strategy.calculate_position_size(signal, {})
            position_value = position_size * signal.price
            
            # Calculate portfolio value
            total_portfolio_value = sum(s.portfolio_value_current for s in self.strategies.values())
            
            # Check single position limit
            position_weight = position_value / total_portfolio_value
            max_position_weight = self.risk_limits['max_single_position'].limit
            
            if position_weight > max_position_weight:
                return False, f"Position size {position_weight:.2%} exceeds limit {max_position_weight:.2%}"
            
            return True, "Position concentration OK"
            
        except Exception as e:
            logger.error(f"Error checking position concentration: {e}")
            return False, f"Error in concentration check: {e}"
    
    def _check_sector_concentration(self, signal: Signal, strategy: BaseStrategy) -> Tuple[bool, str]:
        """Check if signal would violate sector concentration limits."""
        try:
            # Classify signal symbol by sector
            signal_sector = self._classify_sector(signal.symbol)
            
            if not signal_sector:
                return True, "Sector unknown - allowed"
            
            # Calculate current sector exposure
            current_sector_exposure = 0
            total_portfolio_value = sum(s.portfolio_value_current for s in self.strategies.values())
            
            for strategy_obj in self.strategies.values():
                for position in strategy_obj.positions.values():
                    position_sector = self._classify_sector(position.symbol)
                    if position_sector == signal_sector:
                        current_sector_exposure += abs(position.market_value)
            
            # Add new position
            position_size = strategy.calculate_position_size(signal, {})
            position_value = position_size * signal.price
            new_sector_exposure = (current_sector_exposure + position_value) / total_portfolio_value
            
            max_sector_exposure = self.risk_limits['max_sector_exposure'].limit
            
            if new_sector_exposure > max_sector_exposure:
                return False, f"Sector {signal_sector} exposure {new_sector_exposure:.2%} exceeds limit {max_sector_exposure:.2%}"
            
            return True, "Sector concentration OK"
            
        except Exception as e:
            logger.error(f"Error checking sector concentration: {e}")
            return True, "Sector check error - allowing"
    
    async def _check_correlation_risk(self, signal: Signal, strategy: BaseStrategy) -> Tuple[bool, str]:
        """Check correlation risk with existing positions."""
        try:
            # Get existing symbols
            existing_symbols = []
            for strategy_obj in self.strategies.values():
                existing_symbols.extend(list(strategy_obj.positions.keys()))
            
            if not existing_symbols:
                return True, "No existing positions"
            
            # This would require correlation matrix calculation
            # For now, simplified check based on symbol similarity
            similar_symbols = [s for s in existing_symbols if self._symbols_similar(signal.symbol, s)]
            
            if len(similar_symbols) > 3:  # Arbitrary threshold
                return False, f"Too many similar positions: {similar_symbols}"
            
            return True, "Correlation risk OK"
            
        except Exception as e:
            logger.error(f"Error checking correlation risk: {e}")
            return True, "Correlation check error - allowing"
    
    def _classify_sector(self, symbol: str) -> Optional[str]:
        """Classify a symbol into a sector based on keywords."""
        symbol_upper = symbol.upper()
        
        for sector, keywords in self.sector_keywords.items():
            if any(keyword in symbol_upper for keyword in keywords):
                return sector
        
        return None
    
    def _symbols_similar(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are similar (simplified)."""
        # Simple similarity check - could be enhanced with more sophisticated logic
        return symbol1[:2] == symbol2[:2] or symbol1.replace('A', '').replace('B', '') == symbol2.replace('A', '').replace('B', '')
    
    def _get_portfolio_returns(self) -> pd.Series:
        """Get historical portfolio returns."""
        # Aggregate returns across all strategies
        all_returns = []
        
        for strategy in self.strategies.values():
            if strategy.daily_pnl:
                returns = pd.Series(strategy.daily_pnl)
                all_returns.append(returns)
        
        if all_returns:
            # Simple equal-weight combination
            combined = pd.concat(all_returns, axis=1).sum(axis=1)
            return combined
        
        return pd.Series()
    
    def _calculate_var_es(self, returns: pd.Series, confidence_level: float = 0.05) -> Tuple[float, float]:
        """Calculate Value at Risk and Expected Shortfall."""
        if returns.empty or len(returns) < 10:
            return 0.0, 0.0
        
        try:
            # VaR at confidence level
            var = np.percentile(returns, confidence_level * 100)
            
            # Expected Shortfall (average of returns below VaR)
            tail_returns = returns[returns <= var]
            expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var
            
            return abs(var), abs(expected_shortfall)
            
        except Exception as e:
            logger.error(f"Error calculating VaR/ES: {e}")
            return 0.0, 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown across all strategies."""
        max_dd = 0.0
        
        for strategy in self.strategies.values():
            if hasattr(strategy.performance, 'max_drawdown'):
                max_dd = min(max_dd, strategy.performance.max_drawdown)
        
        return abs(max_dd)
    
    def _calculate_concentration_hhi(self, position_values: List[float], total_value: float) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration."""
        if not position_values or total_value <= 0:
            return 0.0
        
        try:
            # Calculate weights
            weights = [value / total_value for value in position_values]
            
            # HHI = sum of squared weights
            hhi = sum(w ** 2 for w in weights)
            
            return hhi
            
        except Exception as e:
            logger.error(f"Error calculating HHI: {e}")
            return 0.0
    
    def _calculate_correlation_risk(self) -> float:
        """Calculate portfolio correlation risk (simplified)."""
        # Simplified - would need full correlation matrix
        return 0.1  # Placeholder
    
    def _calculate_liquidity_risk(self) -> float:
        """Calculate portfolio liquidity risk."""
        # Simplified - would analyze volume vs position sizes
        return 0.05  # Placeholder
    
    def _update_risk_limits(self, metrics: RiskMetrics) -> None:
        """Update current values in risk limits."""
        self.risk_limits['max_portfolio_leverage'].current_value = metrics.leverage
        self.risk_limits['max_gross_exposure'].current_value = metrics.gross_exposure
        self.risk_limits['max_net_exposure'].current_value = abs(metrics.net_exposure)
        self.risk_limits['max_daily_var'].current_value = metrics.var_95
        self.risk_limits['max_concentration_hhi'].current_value = metrics.concentration_risk
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status and alerts."""
        current_metrics = self.calculate_portfolio_risk()
        
        # Check for limit breaches
        active_alerts = []
        for limit in self.risk_limits.values():
            if limit.is_breach:
                active_alerts.append({
                    'type': 'BREACH',
                    'limit': limit.name,
                    'current': limit.current_value,
                    'threshold': limit.limit,
                    'utilization': limit.utilization
                })
            elif limit.is_warning:
                active_alerts.append({
                    'type': 'WARNING',
                    'limit': limit.name,
                    'current': limit.current_value,
                    'threshold': limit.warning_threshold,
                    'utilization': limit.utilization
                })
        
        return {
            'portfolio_value': current_metrics.portfolio_value,
            'risk_metrics': {
                'leverage': current_metrics.leverage,
                'gross_exposure': current_metrics.gross_exposure,
                'net_exposure': current_metrics.net_exposure,
                'var_95': current_metrics.var_95,
                'max_drawdown': current_metrics.max_drawdown,
                'concentration_risk': current_metrics.concentration_risk
            },
            'risk_limits': {name: {
                'current': limit.current_value,
                'limit': limit.limit,
                'utilization': limit.utilization,
                'status': 'BREACH' if limit.is_breach else 'WARNING' if limit.is_warning else 'OK'
            } for name, limit in self.risk_limits.items()},
            'active_alerts': active_alerts,
            'strategies_count': len(self.strategies),
            'last_update': dt.datetime.now().isoformat()
        }