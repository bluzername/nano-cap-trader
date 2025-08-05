"""
Advanced Backtesting Engine for Insider Trading Strategies

This module provides comprehensive backtesting capabilities specifically designed
for insider trading strategies with realistic market conditions and constraints.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from enum import Enum
import asyncio

from ..strategies.base_strategy import BaseStrategy, Signal
from ..strategies.strategy_factory import StrategyFactory

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    min_commission: float = 1.0
    slippage_bp: float = 5.0  # 5 basis points
    max_position_pct: float = 0.05  # 5% max per position
    max_daily_volume_pct: float = 0.01  # 1% of daily volume
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02  # 2% annually
    
    # Insider-specific parameters
    form4_lag_days: int = 2  # Form 4 filing lag
    insider_data_quality: float = 0.85  # 85% data completeness
    options_data_available: bool = False  # Options data availability


@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    direction: int  # 1 for long, -1 for short
    entry_signal_metadata: Dict[str, Any]
    exit_reason: str = ""
    commission_paid: float = 0.0
    slippage_cost: float = 0.0
    
    @property
    def pnl(self) -> float:
        """Calculate P&L for the trade"""
        if self.exit_price is None:
            return 0.0
        
        gross_pnl = (self.exit_price - self.entry_price) * self.quantity * self.direction
        net_pnl = gross_pnl - self.commission_paid - self.slippage_cost
        return net_pnl
    
    @property
    def return_pct(self) -> float:
        """Calculate percentage return"""
        if self.exit_price is None:
            return 0.0
        
        return ((self.exit_price - self.entry_price) / self.entry_price) * self.direction * 100
    
    @property
    def holding_days(self) -> int:
        """Calculate holding period in days"""
        if self.exit_date is None:
            return 0
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResults:
    """Comprehensive backtesting results"""
    # Basic metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Insider-specific metrics
    insider_signal_count: int = 0
    insider_hit_rate: float = 0.0
    avg_insider_return: float = 0.0
    cluster_signal_performance: float = 0.0
    ml_prediction_accuracy: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0
    cvar_95: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    
    # Attribution
    strategy_attribution: Dict[str, float] = field(default_factory=dict)
    monthly_returns: pd.Series = field(default_factory=pd.Series)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    
    # Trade details
    trades: List[Trade] = field(default_factory=list)


class InsiderBacktestEngine:
    """
    Advanced backtesting engine for insider trading strategies
    
    Features:
    - Realistic transaction costs and slippage
    - Form 4 filing delays simulation
    - Market regime analysis
    - Portfolio-level risk management
    - Performance attribution
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.market_data = {}
        self.insider_data = pd.DataFrame()
        self.benchmark_data = pd.DataFrame()
        
        # State tracking
        self.current_date = config.start_date
        self.portfolio_value = config.initial_capital
        self.cash = config.initial_capital
        self.positions = {}
        self.equity_curve = []
        self.trades = []
        
        # Performance tracking
        self.daily_returns = []
        self.benchmark_returns = []
        
    async def run_backtest(self, strategy: BaseStrategy, 
                          universe: List[str]) -> BacktestResults:
        """
        Run comprehensive backtest for insider strategy
        """
        logger.info(f"Starting backtest for {strategy.strategy_id}")
        
        try:
            # Initialize data
            await self._load_historical_data(universe)
            await self._load_insider_data(universe)
            await self._load_benchmark_data()
            
            # Run simulation
            await self._simulate_trading(strategy, universe)
            
            # Calculate results
            results = self._calculate_performance_metrics()
            results.trades = self.trades.copy()
            
            logger.info(f"Backtest completed: {results.total_return:.2%} return, "
                       f"{results.sharpe_ratio:.2f} Sharpe")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    async def _load_historical_data(self, universe: List[str]):
        """Load historical price and volume data"""
        logger.info("Loading historical market data...")
        
        # This would load actual historical data
        # For now, simulate realistic data
        for symbol in universe:
            dates = pd.date_range(
                start=self.config.start_date - timedelta(days=365),
                end=self.config.end_date,
                freq='D'
            )
            
            # Simulate realistic price data with trends and volatility
            np.random.seed(hash(symbol) % 2**32)
            
            # Base parameters for nano-cap stocks
            initial_price = np.random.uniform(2, 50)  # $2-50 range
            annual_vol = np.random.uniform(0.3, 0.8)  # 30-80% volatility
            drift = np.random.uniform(-0.1, 0.3)  # -10% to 30% annual drift
            
            # Generate price series with realistic characteristics
            daily_vol = annual_vol / np.sqrt(252)
            daily_drift = drift / 252
            
            returns = np.random.normal(daily_drift, daily_vol, len(dates))
            prices = initial_price * np.exp(np.cumsum(returns))
            
            # Add some realistic market structure
            # - Higher volatility during market stress
            # - Occasional gaps and illiquidity
            stress_periods = np.random.random(len(dates)) < 0.05
            returns[stress_periods] *= 2.0
            
            # Volume simulation (higher volume on price moves)
            base_volume = np.random.uniform(10000, 100000)
            volume_multiplier = 1 + np.abs(returns) * 5
            volumes = base_volume * volume_multiplier * np.random.uniform(0.5, 2.0, len(dates))
            
            # Create OHLC data
            high_mult = np.random.uniform(1.0, 1.05, len(dates))
            low_mult = np.random.uniform(0.95, 1.0, len(dates))
            
            self.market_data[symbol] = pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': prices * high_mult,
                'low': prices * low_mult,
                'close': prices,
                'volume': volumes.astype(int),
                'adj_close': prices
            }).set_index('date')
    
    async def _load_insider_data(self, universe: List[str]):
        """Load and process insider trading data"""
        logger.info("Loading insider trading data...")
        
        # Simulate realistic Form 4 data
        insider_records = []
        
        for symbol in universe:
            # Generate realistic insider activity
            num_insiders = np.random.poisson(3) + 1  # 1-6 insiders per stock
            
            for insider_id in range(num_insiders):
                # Insider characteristics
                insider_types = ['CEO', 'CFO', 'Director', '10% Owner', 'Officer']
                insider_type = np.random.choice(insider_types)
                insider_name = f"{symbol}_Insider_{insider_id}_{insider_type}"
                
                # Generate transactions over backtest period
                num_transactions = np.random.poisson(2) + 1  # 1-5 transactions
                
                for _ in range(num_transactions):
                    # Random transaction date
                    days_offset = np.random.randint(
                        0, (self.config.end_date - self.config.start_date).days
                    )
                    transaction_date = self.config.start_date + timedelta(days=days_offset)
                    
                    # Filing delay (realistic 1-3 days)
                    filing_delay = np.random.randint(1, 4)
                    filing_date = transaction_date + timedelta(days=filing_delay)
                    
                    # Transaction details
                    is_purchase = np.random.random() > 0.3  # 70% purchases
                    shares = np.random.randint(1000, 50000)
                    
                    # Get price at transaction date
                    if transaction_date in self.market_data[symbol].index:
                        price = self.market_data[symbol].loc[transaction_date, 'close']
                    else:
                        continue
                    
                    insider_records.append({
                        'ticker': symbol,
                        'reportingOwner': insider_name,
                        'insiderTitle': insider_type,
                        'transactionDate': transaction_date,
                        'filingDate': filing_date,
                        'transactionType': 'P' if is_purchase else 'S',
                        'shares': shares,
                        'pricePerShare': price,
                        'netTransactionValue': shares * price,
                        'is10PercentOwner': insider_type == '10% Owner'
                    })
        
        self.insider_data = pd.DataFrame(insider_records)
        logger.info(f"Generated {len(self.insider_data)} insider transactions")
    
    async def _load_benchmark_data(self):
        """Load benchmark data (SPY or similar)"""
        # Simulate benchmark data
        dates = pd.date_range(
            start=self.config.start_date - timedelta(days=365),
            end=self.config.end_date,
            freq='D'
        )
        
        # SPY-like returns (lower volatility, positive drift)
        np.random.seed(42)  # Consistent benchmark
        daily_returns = np.random.normal(0.0008, 0.015, len(dates))  # ~20% annual, 15% vol
        prices = 100 * np.exp(np.cumsum(daily_returns))
        
        self.benchmark_data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'returns': daily_returns
        }).set_index('date')
    
    async def _simulate_trading(self, strategy: BaseStrategy, universe: List[str]):
        """Simulate the trading process day by day"""
        logger.info("Running trading simulation...")
        
        current_date = self.config.start_date
        
        while current_date <= self.config.end_date:
            # Skip weekends (basic calendar)
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            
            # Update portfolio valuation
            self._update_portfolio_value(current_date)
            
            # Get available insider data (with filing delay)
            available_insider_data = self.insider_data[
                (self.insider_data['filingDate'] <= current_date) &
                (self.insider_data['transactionDate'] >= current_date - timedelta(days=90))
            ]
            
            if not available_insider_data.empty:
                # Generate signals using strategy
                signals = await self._generate_signals(
                    strategy, current_date, available_insider_data, universe
                )
                
                # Execute trades
                for signal in signals:
                    await self._execute_trade(signal, current_date)
            
            # Check for exits
            await self._check_exits(current_date)
            
            # Record daily performance
            self._record_daily_performance(current_date)
            
            current_date += timedelta(days=1)
    
    async def _generate_signals(self, strategy: BaseStrategy, current_date: datetime,
                               insider_data: pd.DataFrame, universe: List[str]) -> List[Signal]:
        """Generate trading signals for current date"""
        
        # Prepare data for strategy
        market_data_dict = {}
        for symbol in universe:
            if symbol in self.market_data:
                # Get data up to current date
                symbol_data = self.market_data[symbol][
                    self.market_data[symbol].index <= current_date
                ]
                if not symbol_data.empty:
                    market_data_dict[symbol] = symbol_data
        
        # Set strategy kwargs
        strategy.kwargs = {
            'form4_data': insider_data,
            'market_data': market_data_dict,
            'current_date': current_date
        }
        
        # Generate signals
        signals = strategy.generate_signals()
        
        # Filter signals for universe and existing positions
        filtered_signals = []
        for signal in signals:
            if (signal.symbol in universe and 
                signal.symbol not in self.positions and
                signal.symbol in market_data_dict):
                filtered_signals.append(signal)
        
        return filtered_signals
    
    async def _execute_trade(self, signal: Signal, current_date: datetime):
        """Execute a trade based on signal"""
        
        symbol = signal.symbol
        
        # Get current market data
        if symbol not in self.market_data:
            return
        
        market_data = self.market_data[symbol]
        current_data = market_data[market_data.index <= current_date]
        
        if current_data.empty:
            return
        
        current_price = current_data.iloc[-1]['close']
        current_volume = current_data.iloc[-1]['volume']
        
        # Calculate position size
        max_position_value = self.portfolio_value * self.config.max_position_pct
        max_shares_by_volume = int(current_volume * self.config.max_daily_volume_pct)
        max_shares_by_capital = int(max_position_value / current_price)
        
        shares = min(max_shares_by_volume, max_shares_by_capital)
        
        if shares <= 0 or shares * current_price > self.cash:
            return
        
        # Apply slippage
        slippage_factor = self.config.slippage_bp / 10000
        execution_price = current_price * (1 + slippage_factor)
        
        # Calculate costs
        gross_amount = shares * execution_price
        commission = max(gross_amount * self.config.commission_rate, self.config.min_commission)
        slippage_cost = shares * current_price * slippage_factor
        total_cost = gross_amount + commission + slippage_cost
        
        if total_cost > self.cash:
            return
        
        # Execute trade
        trade = Trade(
            symbol=symbol,
            entry_date=current_date,
            exit_date=None,
            entry_price=execution_price,
            exit_price=None,
            quantity=shares,
            direction=1,  # Long only for now
            entry_signal_metadata=signal.metadata,
            commission_paid=commission,
            slippage_cost=slippage_cost
        )
        
        # Update portfolio
        self.positions[symbol] = trade
        self.cash -= total_cost
        
        logger.debug(f"Executed trade: {symbol} x{shares} @ ${execution_price:.2f}")
    
    async def _check_exits(self, current_date: datetime):
        """Check for position exits"""
        
        positions_to_close = []
        
        for symbol, trade in self.positions.items():
            if symbol not in self.market_data:
                continue
            
            market_data = self.market_data[symbol]
            current_data = market_data[market_data.index <= current_date]
            
            if current_data.empty:
                continue
            
            current_price = current_data.iloc[-1]['close']
            holding_days = (current_date - trade.entry_date).days
            
            # Exit conditions
            should_exit = False
            exit_reason = ""
            
            # Time-based exit (60 days max holding)
            if holding_days >= 60:
                should_exit = True
                exit_reason = "time_limit"
            
            # Stop loss (20% loss)
            elif current_price < trade.entry_price * 0.8:
                should_exit = True
                exit_reason = "stop_loss"
            
            # Take profit (50% gain)
            elif current_price > trade.entry_price * 1.5:
                should_exit = True
                exit_reason = "take_profit"
            
            if should_exit:
                positions_to_close.append((symbol, current_price, exit_reason))
        
        # Close positions
        for symbol, exit_price, exit_reason in positions_to_close:
            await self._close_position(symbol, exit_price, current_date, exit_reason)
    
    async def _close_position(self, symbol: str, exit_price: float, 
                             exit_date: datetime, exit_reason: str):
        """Close a position"""
        
        if symbol not in self.positions:
            return
        
        trade = self.positions[symbol]
        
        # Apply slippage
        slippage_factor = self.config.slippage_bp / 10000
        execution_price = exit_price * (1 - slippage_factor)
        
        # Calculate costs
        gross_amount = trade.quantity * execution_price
        commission = max(gross_amount * self.config.commission_rate, self.config.min_commission)
        slippage_cost = trade.quantity * exit_price * slippage_factor
        net_proceeds = gross_amount - commission - slippage_cost
        
        # Update trade record
        trade.exit_date = exit_date
        trade.exit_price = execution_price
        trade.exit_reason = exit_reason
        trade.commission_paid += commission
        trade.slippage_cost += slippage_cost
        
        # Update portfolio
        self.cash += net_proceeds
        self.trades.append(trade)
        del self.positions[symbol]
        
        logger.debug(f"Closed position: {symbol} @ ${execution_price:.2f}, "
                    f"P&L: ${trade.pnl:.2f}")
    
    def _update_portfolio_value(self, current_date: datetime):
        """Update portfolio valuation"""
        
        position_value = 0.0
        
        for symbol, trade in self.positions.items():
            if symbol in self.market_data:
                market_data = self.market_data[symbol]
                current_data = market_data[market_data.index <= current_date]
                
                if not current_data.empty:
                    current_price = current_data.iloc[-1]['close']
                    position_value += trade.quantity * current_price
        
        self.portfolio_value = self.cash + position_value
    
    def _record_daily_performance(self, current_date: datetime):
        """Record daily performance metrics"""
        
        self.equity_curve.append({
            'date': current_date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions_value': self.portfolio_value - self.cash,
            'num_positions': len(self.positions)
        })
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2]['portfolio_value']
            daily_return = (self.portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
            
            # Benchmark return
            if current_date in self.benchmark_data.index:
                benchmark_return = self.benchmark_data.loc[current_date, 'returns']
                self.benchmark_returns.append(benchmark_return)
            else:
                self.benchmark_returns.append(0.0)
    
    def _calculate_performance_metrics(self) -> BacktestResults:
        """Calculate comprehensive performance metrics"""
        
        if not self.trades and not self.equity_curve:
            return BacktestResults()
        
        # Convert to pandas for easier calculation
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        returns_series = pd.Series(self.daily_returns)
        benchmark_series = pd.Series(self.benchmark_returns)
        
        # Basic metrics
        total_return = (self.portfolio_value - self.config.initial_capital) / self.config.initial_capital
        trading_days = len(self.daily_returns)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        
        volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 1 else 0
        sharpe_ratio = (annual_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + returns_series).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_series[returns_series < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else volatility
        sortino_ratio = (annual_return - self.config.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trading metrics
        completed_trades = [t for t in self.trades if t.exit_date is not None]
        total_trades = len(completed_trades)
        winning_trades = len([t for t in completed_trades if t.pnl > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        wins = [t.pnl for t in completed_trades if t.pnl > 0]
        losses = [t.pnl for t in completed_trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        
        # Risk metrics
        if len(returns_series) > 1:
            var_95 = np.percentile(returns_series, 5)
            cvar_95 = returns_series[returns_series <= var_95].mean()
        else:
            var_95 = cvar_95 = 0
        
        # Market metrics (beta, alpha)
        if len(returns_series) > 1 and len(benchmark_series) > 1:
            covariance = np.cov(returns_series, benchmark_series)[0, 1]
            benchmark_var = np.var(benchmark_series)
            beta = covariance / benchmark_var if benchmark_var > 0 else 0
            
            benchmark_annual = benchmark_series.mean() * 252
            alpha = annual_return - (self.config.risk_free_rate + beta * (benchmark_annual - self.config.risk_free_rate))
            
            # Information ratio
            excess_returns = returns_series - benchmark_series
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        else:
            beta = alpha = information_ratio = 0
        
        # Insider-specific metrics
        insider_trades = [t for t in completed_trades if 'insider_score' in t.entry_signal_metadata]
        insider_signal_count = len(insider_trades)
        insider_hit_rate = len([t for t in insider_trades if t.pnl > 0]) / len(insider_trades) if insider_trades else 0
        avg_insider_return = np.mean([t.return_pct for t in insider_trades]) if insider_trades else 0
        
        # Cluster signals performance
        cluster_trades = [t for t in completed_trades if t.entry_signal_metadata.get('cluster_score', 0) > 0.5]
        cluster_signal_performance = np.mean([t.return_pct for t in cluster_trades]) if cluster_trades else 0
        
        # ML prediction accuracy (if applicable)
        ml_trades = [t for t in completed_trades if 'ml_confidence' in t.entry_signal_metadata]
        ml_prediction_accuracy = len([t for t in ml_trades if t.pnl > 0]) / len(ml_trades) if ml_trades else 0
        
        return BacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            insider_signal_count=insider_signal_count,
            insider_hit_rate=insider_hit_rate,
            avg_insider_return=avg_insider_return,
            cluster_signal_performance=cluster_signal_performance,
            ml_prediction_accuracy=ml_prediction_accuracy,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            equity_curve=equity_df['portfolio_value'],
            drawdown_series=drawdown,
            trades=completed_trades
        )


async def run_strategy_comparison(strategies: List[str], universe: List[str],
                                config: BacktestConfig) -> Dict[str, BacktestResults]:
    """
    Run backtests for multiple strategies and compare results
    """
    results = {}
    
    for strategy_name in strategies:
        logger.info(f"Backtesting strategy: {strategy_name}")
        
        # Create strategy instance
        strategy = StrategyFactory.create_strategy(strategy_name, universe)
        if not strategy:
            logger.error(f"Failed to create strategy: {strategy_name}")
            continue
        
        # Run backtest
        engine = InsiderBacktestEngine(config)
        result = await engine.run_backtest(strategy, universe)
        results[strategy_name] = result
    
    return results


# Convenience function for quick backtesting
async def quick_backtest(strategy_name: str, universe: List[str] = None,
                        start_date: str = "2023-01-01", end_date: str = "2024-01-01") -> BacktestResults:
    """Quick backtest with default parameters"""
    
    if universe is None:
        universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # Default universe
    
    config = BacktestConfig(
        start_date=datetime.strptime(start_date, "%Y-%m-%d"),
        end_date=datetime.strptime(end_date, "%Y-%m-%d"),
        initial_capital=100000.0
    )
    
    strategy = StrategyFactory.create_strategy(strategy_name, universe)
    if not strategy:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    engine = InsiderBacktestEngine(config)
    return await engine.run_backtest(strategy, universe)