"""Base strategy class providing common framework for all trading strategies."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from enum import Enum
import logging

from ..config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


class StrategyType(Enum):
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    MULTI_STRATEGY = "multi_strategy"


class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class Signal:
    """Trading signal with metadata."""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    price: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class Position:
    """Trading position with risk metrics."""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    strategy_id: str = ""
    
    @property
    def market_value(self) -> float:
        return self.quantity * (self.current_price or self.entry_price)
    
    @property
    def pnl_percent(self) -> float:
        if self.current_price:
            return (self.current_price - self.entry_price) / self.entry_price
        return 0.0


@dataclass
class PerformanceMetrics:
    """Strategy performance metrics."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    trades_count: int = 0
    avg_trade_duration: float = 0.0


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(
        self,
        strategy_id: str,
        strategy_type: StrategyType,
        universe: List[str],
        max_positions: int = 50,
        position_size_pct: float = 0.02,  # 2% of portfolio per position
        enable_stop_loss: bool = True,
        stop_loss_pct: float = 0.02,  # 2% stop loss
        enable_position_sizing: bool = True,
        max_volume_pct: float = 0.03,  # 3% of daily volume max
    ):
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.universe = universe
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.enable_stop_loss = enable_stop_loss
        self.stop_loss_pct = stop_loss_pct
        self.enable_position_sizing = enable_position_sizing
        self.max_volume_pct = max_volume_pct
        
        # State tracking
        self.positions: Dict[str, Position] = {}
        self.signals_history: List[Signal] = []
        self.performance: PerformanceMetrics = PerformanceMetrics()
        self.is_active: bool = False
        self.last_update: Optional[datetime] = None
        
        # Risk metrics
        self.portfolio_value: float = _settings.max_portfolio_value
        self.cash: float = _settings.max_portfolio_value
        self.daily_pnl: List[float] = []
        self.benchmark_returns: List[float] = []
        
    @abstractmethod
    async def generate_signals(
        self, 
        market_data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> List[Signal]:
        """Generate trading signals based on strategy logic."""
        pass
    
    @abstractmethod
    def calculate_position_size(
        self, 
        signal: Signal,
        market_data: Dict[str, Any]
    ) -> int:
        """Calculate position size based on risk management rules."""
        pass
    
    def validate_signal(self, signal: Signal, market_data: Dict[str, Any]) -> bool:
        """Validate signal against risk management rules."""
        try:
            # Check universe membership
            if signal.symbol not in self.universe:
                logger.warning(f"Symbol {signal.symbol} not in strategy universe")
                return False
            
            # Check position limits
            if len(self.positions) >= self.max_positions and signal.symbol not in self.positions:
                logger.warning(f"Maximum positions ({self.max_positions}) reached")
                return False
            
            # Check volume constraints if enabled
            if self.enable_position_sizing:
                volume_data = market_data.get(signal.symbol, {})
                avg_volume = volume_data.get('avg_volume', 0)
                if avg_volume == 0:
                    logger.warning(f"No volume data for {signal.symbol}")
                    return False
                
                position_size = self.calculate_position_size(signal, market_data)
                max_shares = avg_volume * self.max_volume_pct
                
                if position_size > max_shares:
                    logger.warning(
                        f"Position size {position_size} exceeds volume limit {max_shares} for {signal.symbol}"
                    )
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal for {signal.symbol}: {e}")
            return False
    
    def execute_signal(self, signal: Signal, market_data: Dict[str, Any]) -> bool:
        """Execute a validated signal."""
        try:
            if not self.validate_signal(signal, market_data):
                return False
            
            position_size = self.calculate_position_size(signal, market_data)
            
            if signal.signal_type == SignalType.BUY:
                return self._open_position(signal, position_size, market_data)
            elif signal.signal_type == SignalType.SELL:
                return self._close_position(signal, market_data)
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
            return False
    
    def _open_position(self, signal: Signal, quantity: int, market_data: Dict[str, Any]) -> bool:
        """Open a new position."""
        try:
            cost = quantity * signal.price
            transaction_cost = cost * 0.001  # 0.1% transaction cost
            total_cost = cost + transaction_cost
            
            if total_cost > self.cash:
                logger.warning(f"Insufficient cash for {signal.symbol}: need {total_cost}, have {self.cash}")
                return False
            
            # Calculate stop loss if enabled
            stop_loss = None
            if self.enable_stop_loss:
                if signal.signal_type == SignalType.BUY:
                    stop_loss = signal.price * (1 - self.stop_loss_pct)
                else:
                    stop_loss = signal.price * (1 + self.stop_loss_pct)
            
            position = Position(
                symbol=signal.symbol,
                quantity=quantity,
                entry_price=signal.price,
                entry_time=signal.timestamp,
                stop_loss=stop_loss,
                strategy_id=self.strategy_id
            )
            
            self.positions[signal.symbol] = position
            self.cash -= total_cost
            self.signals_history.append(signal)
            
            logger.info(f"Opened position: {quantity} shares of {signal.symbol} at ${signal.price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening position for {signal.symbol}: {e}")
            return False
    
    def _close_position(self, signal: Signal, market_data: Dict[str, Any]) -> bool:
        """Close an existing position."""
        try:
            if signal.symbol not in self.positions:
                logger.warning(f"No position to close for {signal.symbol}")
                return False
            
            position = self.positions[signal.symbol]
            proceeds = position.quantity * signal.price
            transaction_cost = proceeds * 0.001  # 0.1% transaction cost
            net_proceeds = proceeds - transaction_cost
            
            # Calculate realized PnL
            cost_basis = position.quantity * position.entry_price
            realized_pnl = net_proceeds - cost_basis
            
            self.cash += net_proceeds
            del self.positions[signal.symbol]
            self.signals_history.append(signal)
            
            logger.info(
                f"Closed position: {position.quantity} shares of {signal.symbol} "
                f"at ${signal.price:.2f}, PnL: ${realized_pnl:.2f}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error closing position for {signal.symbol}: {e}")
            return False
    
    def update_positions(self, market_data: Dict[str, Dict[str, float]]) -> None:
        """Update position values and check stop losses."""
        try:
            for symbol, position in list(self.positions.items()):
                if symbol in market_data:
                    current_price = market_data[symbol].get('close', position.entry_price)
                    position.current_price = current_price
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    
                    # Check stop loss
                    if self.enable_stop_loss and position.stop_loss:
                        if (position.quantity > 0 and current_price <= position.stop_loss) or \
                           (position.quantity < 0 and current_price >= position.stop_loss):
                            # Generate stop loss signal
                            stop_signal = Signal(
                                symbol=symbol,
                                signal_type=SignalType.SELL,
                                confidence=1.0,
                                price=current_price,
                                timestamp=datetime.now(),
                                metadata={"reason": "stop_loss"}
                            )
                            self.execute_signal(stop_signal, {symbol: market_data[symbol]})
                            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def calculate_performance_metrics(self, benchmark_returns: pd.Series) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        try:
            if not self.daily_pnl:
                return self.performance
            
            returns = pd.Series(self.daily_pnl)
            
            # Basic metrics
            total_return = (self.portfolio_value / _settings.max_portfolio_value) - 1
            
            # Risk-adjusted metrics
            if len(returns) > 1:
                sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
                sortino_ratio = np.sqrt(252) * returns.mean() / returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0
            else:
                sharpe_ratio = sortino_ratio = 0
            
            # Drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Alpha and Beta (if benchmark provided)
            alpha = beta = 0
            if len(benchmark_returns) == len(returns) and len(returns) > 1:
                covariance = np.cov(returns, benchmark_returns)[0][1]
                benchmark_variance = np.var(benchmark_returns)
                if benchmark_variance > 0:
                    beta = covariance / benchmark_variance
                    alpha = returns.mean() - beta * benchmark_returns.mean()
            
            # Update performance metrics
            self.performance = PerformanceMetrics(
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                alpha=alpha * 252,  # Annualized
                beta=beta,
                trades_count=len(self.signals_history),
            )
            
            return self.performance
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self.performance
    
    @property
    def portfolio_value_current(self) -> float:
        """Calculate current portfolio value."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_status(self) -> Dict[str, Any]:
        """Get current strategy status."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_type": self.strategy_type.value,
            "is_active": self.is_active,
            "positions_count": len(self.positions),
            "cash": self.cash,
            "portfolio_value": self.portfolio_value_current,
            "performance": {
                "total_return": self.performance.total_return,
                "sharpe_ratio": self.performance.sharpe_ratio,
                "max_drawdown": self.performance.max_drawdown,
                "trades_count": self.performance.trades_count,
            },
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }