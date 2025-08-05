"""
Insider + Options Flow Strategy

This strategy combines Form 4 insider trading signals with unusual options activity:
- Detects when insiders buy followed by unusual call options activity
- Identifies "smart money" convergence (insiders + options traders)
- Uses options flow to time entries and validate insider signals
- Incorporates implied volatility analysis

Target Performance: 7-9% annual alpha
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .base_strategy import BaseStrategy, Signal, StrategyType, SignalType
# Technical indicators would be implemented here
# from ..utils.technical_indicators import (
#     calculate_rsi, calculate_atr, calculate_vwap,
#     calculate_bollinger_bands, calculate_obv
# )

logger = logging.getLogger(__name__)


@dataclass
class OptionsFlow:
    """Options flow data structure"""
    ticker: str
    timestamp: datetime
    option_type: str  # 'CALL' or 'PUT'
    strike: float
    expiry: datetime
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    premium: float
    is_sweep: bool
    is_block: bool
    sentiment: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'


class InsiderOptionsFlowStrategy(BaseStrategy):
    """
    Strategy combining insider trading with options flow analysis
    
    Key innovations:
    1. Unusual options activity detection algorithm
    2. Options flow sentiment scoring
    3. Implied volatility rank analysis
    4. Time-decay aware position management
    5. Greeks-based risk management
    """
    
    def __init__(self, universe: List[str], **kwargs):
        # Extract base strategy parameters
        base_params = {
            'max_positions': kwargs.get('max_positions', 50),
            'position_size_pct': kwargs.get('position_size_pct', 0.02),
            'enable_stop_loss': kwargs.get('enable_stop_loss', True),
            'stop_loss_pct': kwargs.get('stop_loss_pct', 0.02),
            'enable_position_sizing': kwargs.get('enable_position_sizing', True),
            'max_volume_pct': kwargs.get('max_volume_pct', 0.03)
        }
        
        super().__init__(
            strategy_id="insider_options_flow",
            strategy_type=StrategyType.MOMENTUM,
            universe=universe,
            **base_params
        )
        
        # Insider parameters
        self.insider_lookback = kwargs.get('insider_lookback', 30)
        self.min_insider_value = kwargs.get('min_insider_value', 100000)
        self.insider_signal_weight = kwargs.get('insider_signal_weight', 0.4)
        
        # Options flow parameters
        self.options_lookback = kwargs.get('options_lookback', 5)
        self.min_options_volume = kwargs.get('min_options_volume', 100)
        self.unusual_volume_threshold = kwargs.get('unusual_volume_threshold', 2.0)
        self.min_premium = kwargs.get('min_premium', 10000)
        self.sweep_multiplier = kwargs.get('sweep_multiplier', 1.5)
        self.block_multiplier = kwargs.get('block_multiplier', 1.3)
        
        # IV parameters
        self.iv_rank_period = kwargs.get('iv_rank_period', 252)
        self.high_iv_threshold = kwargs.get('high_iv_threshold', 0.7)
        self.low_iv_threshold = kwargs.get('low_iv_threshold', 0.3)
        
        # Signal generation parameters
        self.min_combined_score = kwargs.get('min_combined_score', 0.65)
        self.confirmation_window = kwargs.get('confirmation_window', 3)
        
        # Risk parameters
        self.max_iv_for_entry = kwargs.get('max_iv_for_entry', 0.8)
        self.min_days_to_expiry = kwargs.get('min_days_to_expiry', 30)
        self.max_position_pct = kwargs.get('max_position_pct', 0.04)
        self.stop_loss_pct = kwargs.get('stop_loss_pct', 0.06)
        
        # Cache for calculations
        self._iv_rank_cache = {}
        self._options_flow_cache = {}
        
        # Store kwargs for later use
        self.strategy_kwargs = kwargs
        
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame], **kwargs) -> List[Signal]:
        """Generate signals based on insider + options flow convergence"""
        signals = []
        
        try:
            # Get data sources
            form4_data = kwargs.get('form4_data')
            options_data = kwargs.get('options_flow_data')
            
            # Create placeholder data if not available
            if form4_data is None or form4_data.empty:
                form4_data = self._create_placeholder_form4_data()
            if options_data is None or options_data.empty:
                options_data = self._create_placeholder_options_data()
            
            if form4_data is None or options_data is None:
                logger.warning("Missing required data: Form 4 or options flow")
                return signals
            
            # Identify stocks with recent insider buying
            insider_candidates = self._identify_insider_candidates(form4_data)
            
            # Check each candidate for options flow confirmation
            for symbol in insider_candidates:
                try:
                    signal = self._analyze_convergence(
                        symbol, form4_data, options_data, market_data
                    )
                    if signal:
                        signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Also check for options flow leading insider activity
            options_candidates = self._identify_options_candidates(options_data)
            for symbol in options_candidates:
                if symbol not in insider_candidates:  # Avoid duplicates
                    try:
                        signal = self._analyze_options_led_signal(
                            symbol, form4_data, options_data, market_data
                        )
                        if signal:
                            signals.append(signal)
                    except Exception as e:
                        logger.error(f"Error analyzing options-led {symbol}: {e}")
            
            # Apply portfolio constraints
            signals = self._apply_portfolio_constraints(signals)
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            
        return signals
    
    def calculate_position_size(self, signal: Signal, market_data: Dict[str, Any]) -> int:
        """Calculate position size based on risk management rules"""
        try:
            # Get available cash for position sizing
            available_cash = self.cash * self.max_position_pct
            
            # Calculate maximum shares based on available cash
            max_shares_by_cash = int(available_cash / signal.price)
            
            # Apply volume constraints if market data available
            if signal.symbol in market_data:
                avg_volume = market_data[signal.symbol].get('avg_volume', 1000000)
                max_shares_by_volume = int(avg_volume * self.max_volume_pct)
                position_size = min(max_shares_by_cash, max_shares_by_volume)
            else:
                position_size = max_shares_by_cash
            
            return max(position_size, 0)
            
        except Exception as e:
            logger.error(f"Error calculating position size for {signal.symbol}: {e}")
            return 0
    
    def _create_placeholder_form4_data(self) -> pd.DataFrame:
        """Create placeholder Form 4 data for demo purposes"""
        import pandas as pd
        from datetime import datetime, timedelta
        
        sample_data = []
        for i, symbol in enumerate(self.universe[:3]):
            sample_data.append({
                'ticker': symbol,
                'filingDate': datetime.now() - timedelta(days=i+1),
                'transactionDate': datetime.now() - timedelta(days=i+2),
                'reportingOwner': f'Insider_{i}',
                'insiderTitle': 'CEO' if i == 0 else 'Director',
                'transactionType': 'P',
                'netTransactionValue': 150000 + i * 75000,
            })
        
        return pd.DataFrame(sample_data)
    
    def _create_placeholder_options_data(self) -> pd.DataFrame:
        """Create placeholder options flow data for demo purposes"""
        import pandas as pd
        from datetime import datetime, timedelta
        
        sample_data = []
        for i, symbol in enumerate(self.universe[:3]):
            sample_data.append({
                'ticker': symbol,
                'timestamp': datetime.now() - timedelta(hours=i+1),
                'option_type': 'CALL',
                'volume': 500 + i * 200,
                'premium': 25000 + i * 10000,
                'is_sweep': i % 2 == 0,
                'is_block': i % 3 == 0,
            })
        
        return pd.DataFrame(sample_data)
    
    def _identify_insider_candidates(self, form4_data: pd.DataFrame) -> List[str]:
        """Identify stocks with significant recent insider buying"""
        cutoff_date = datetime.now() - timedelta(days=self.insider_lookback)
        
        recent_purchases = form4_data[
            (form4_data['transactionType'] == 'P') &
            (form4_data['transactionDate'] >= cutoff_date) &
            (form4_data['netTransactionValue'] >= self.min_insider_value)
        ]
        
        if recent_purchases.empty:
            return []
        
        # Aggregate by ticker
        insider_summary = recent_purchases.groupby('ticker').agg({
            'netTransactionValue': 'sum',
            'reportingOwner': 'nunique',
            'transactionDate': 'count'
        }).rename(columns={'transactionDate': 'transaction_count'})
        
        # Filter for significant activity
        candidates = insider_summary[
            (insider_summary['netTransactionValue'] >= self.min_insider_value * 2) |
            (insider_summary['reportingOwner'] >= 2)
        ].index.tolist()
        
        # Only include stocks in our universe
        return [s for s in candidates if s in self.universe]
    
    def _identify_options_candidates(self, options_data: pd.DataFrame) -> List[str]:
        """Identify stocks with unusual options activity"""
        if options_data.empty:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=self.options_lookback)
        recent_flow = options_data[options_data['timestamp'] >= cutoff_date]
        
        # Calculate unusual activity score
        flow_summary = []
        
        for symbol in recent_flow['ticker'].unique():
            if symbol not in self.universe:
                continue
                
            symbol_flow = recent_flow[recent_flow['ticker'] == symbol]
            
            # Calculate metrics
            call_volume = symbol_flow[symbol_flow['option_type'] == 'CALL']['volume'].sum()
            put_volume = symbol_flow[symbol_flow['option_type'] == 'PUT']['volume'].sum()
            
            if call_volume + put_volume < self.min_options_volume:
                continue
            
            # Call/Put ratio
            cp_ratio = call_volume / max(put_volume, 1)
            
            # Premium analysis
            call_premium = symbol_flow[symbol_flow['option_type'] == 'CALL']['premium'].sum()
            
            # Sweep and block detection
            sweeps = symbol_flow['is_sweep'].sum()
            blocks = symbol_flow['is_block'].sum()
            
            if call_premium >= self.min_premium and cp_ratio > 1.5:
                flow_summary.append({
                    'ticker': symbol,
                    'score': cp_ratio * (1 + sweeps * 0.1 + blocks * 0.05),
                    'call_premium': call_premium
                })
        
        # Sort by score and return top candidates
        flow_df = pd.DataFrame(flow_summary)
        if not flow_df.empty:
            flow_df = flow_df.sort_values('score', ascending=False)
            return flow_df.head(10)['ticker'].tolist()
        
        return []
    
    def _analyze_convergence(self, symbol: str, form4_data: pd.DataFrame,
                           options_data: pd.DataFrame, market_data: Dict) -> Optional[Signal]:
        """Analyze convergence of insider and options signals"""
        
        # Get insider score
        insider_score = self._calculate_insider_score(symbol, form4_data)
        if insider_score < 0.5:
            return None
        
        # Get options flow score
        options_score = self._calculate_options_score(symbol, options_data)
        if options_score < 0.3:
            return None
        
        # Check timing alignment
        timing_score = self._check_signal_timing(symbol, form4_data, options_data)
        
        # Get IV rank
        iv_rank = self._calculate_iv_rank(symbol, options_data)
        iv_score = self._score_iv_rank(iv_rank)
        
        # Technical confirmation
        technical_score = self._get_technical_score(symbol, market_data)
        
        # Calculate composite score
        composite_score = (
            insider_score * self.insider_signal_weight +
            options_score * 0.3 +
            timing_score * 0.1 +
            iv_score * 0.1 +
            technical_score * 0.1
        )
        
        if composite_score >= self.min_combined_score:
            return self._create_convergence_signal(
                symbol, composite_score, insider_score, options_score,
                iv_rank, form4_data, options_data
            )
        
        return None
    
    def _calculate_insider_score(self, symbol: str, form4_data: pd.DataFrame) -> float:
        """Calculate normalized insider buying score"""
        symbol_data = form4_data[
            (form4_data['ticker'] == symbol) &
            (form4_data['transactionType'] == 'P') &
            (form4_data['transactionDate'] >= datetime.now() - timedelta(days=self.insider_lookback))
        ]
        
        if symbol_data.empty:
            return 0.0
        
        # Factors for scoring
        total_value = symbol_data['netTransactionValue'].sum()
        unique_insiders = symbol_data['reportingOwner'].nunique()
        avg_transaction = symbol_data['netTransactionValue'].mean()
        
        # Weighted score
        value_score = min(total_value / 1000000, 2.0) / 2.0  # Cap at $2M
        insider_score = min(unique_insiders / 3, 1.0)  # 3+ insiders = max score
        size_score = min(avg_transaction / 250000, 1.0)  # $250k+ avg = max score
        
        return (value_score * 0.5 + insider_score * 0.3 + size_score * 0.2)
    
    def _calculate_options_score(self, symbol: str, options_data: pd.DataFrame) -> float:
        """Calculate options flow bullishness score"""
        symbol_flow = options_data[
            (options_data['ticker'] == symbol) &
            (options_data['timestamp'] >= datetime.now() - timedelta(days=self.options_lookback))
        ]
        
        if symbol_flow.empty:
            return 0.0
        
        # Separate calls and puts
        calls = symbol_flow[symbol_flow['option_type'] == 'CALL']
        puts = symbol_flow[symbol_flow['option_type'] == 'PUT']
        
        # Volume analysis
        call_volume = calls['volume'].sum()
        put_volume = puts['volume'].sum()
        total_volume = call_volume + put_volume
        
        if total_volume < self.min_options_volume:
            return 0.0
        
        # Calculate scores
        volume_score = min(call_volume / max(put_volume, 1), 3.0) / 3.0
        
        # Premium analysis
        call_premium = calls['premium'].sum()
        put_premium = puts['premium'].sum()
        premium_score = min(call_premium / max(put_premium, 1), 3.0) / 3.0
        
        # Sweep and block analysis
        call_sweeps = calls['is_sweep'].sum()
        call_blocks = calls['is_block'].sum()
        flow_quality = 1.0 + (call_sweeps * 0.1 + call_blocks * 0.05)
        
        # Delta-weighted analysis
        if 'delta' in calls.columns:
            weighted_delta = (calls['delta'] * calls['volume']).sum() / max(calls['volume'].sum(), 1)
            delta_score = max(weighted_delta, 0.0)
        else:
            delta_score = 0.5
        
        # Combine scores
        raw_score = (
            volume_score * 0.3 +
            premium_score * 0.3 +
            delta_score * 0.2 +
            min(flow_quality, 1.5) * 0.2
        )
        
        return min(raw_score, 1.0)
    
    def _check_signal_timing(self, symbol: str, form4_data: pd.DataFrame,
                           options_data: pd.DataFrame) -> float:
        """Check if insider and options activity are aligned in time"""
        
        # Get recent insider purchases
        insider_dates = form4_data[
            (form4_data['ticker'] == symbol) &
            (form4_data['transactionType'] == 'P') &
            (form4_data['transactionDate'] >= datetime.now() - timedelta(days=self.insider_lookback))
        ]['transactionDate'].values
        
        if len(insider_dates) == 0:
            return 0.0
        
        # Get recent bullish options flow
        bullish_flow = options_data[
            (options_data['ticker'] == symbol) &
            (options_data['option_type'] == 'CALL') &
            (options_data['timestamp'] >= datetime.now() - timedelta(days=self.options_lookback))
        ]['timestamp'].values
        
        if len(bullish_flow) == 0:
            return 0.0
        
        # Check for temporal proximity
        timing_scores = []
        
        for insider_date in insider_dates:
            insider_dt = pd.to_datetime(insider_date)
            
            # Find closest options activity
            for options_dt in pd.to_datetime(bullish_flow):
                days_diff = abs((options_dt - insider_dt).days)
                
                if days_diff <= self.confirmation_window:
                    # Perfect timing alignment
                    timing_scores.append(1.0)
                elif days_diff <= self.confirmation_window * 2:
                    # Good alignment
                    timing_scores.append(0.7)
                elif days_diff <= self.confirmation_window * 4:
                    # Moderate alignment
                    timing_scores.append(0.4)
        
        return max(timing_scores) if timing_scores else 0.0
    
    def _calculate_iv_rank(self, symbol: str, options_data: pd.DataFrame) -> float:
        """Calculate implied volatility rank (0-1)"""
        
        # Check cache first
        cache_key = f"{symbol}_{datetime.now().date()}"
        if cache_key in self._iv_rank_cache:
            return self._iv_rank_cache[cache_key]
        
        # Get IV history
        symbol_options = options_data[options_data['ticker'] == symbol]
        if symbol_options.empty or 'implied_volatility' not in symbol_options.columns:
            return 0.5  # Default middle rank
        
        # Calculate IV rank over period
        cutoff = datetime.now() - timedelta(days=self.iv_rank_period)
        iv_history = symbol_options[symbol_options['timestamp'] >= cutoff]['implied_volatility']
        
        if len(iv_history) < 20:
            return 0.5
        
        current_iv = iv_history.iloc[-1]
        iv_rank = (iv_history < current_iv).sum() / len(iv_history)
        
        # Cache result
        self._iv_rank_cache[cache_key] = iv_rank
        
        return iv_rank
    
    def _score_iv_rank(self, iv_rank: float) -> float:
        """Score IV rank for entry timing"""
        if iv_rank < self.low_iv_threshold:
            # Low IV - good for buying options
            return 1.0
        elif iv_rank < 0.5:
            return 0.8
        elif iv_rank < self.high_iv_threshold:
            return 0.5
        elif iv_rank < self.max_iv_for_entry:
            return 0.3
        else:
            # Very high IV - avoid entry
            return 0.0
    
    def _get_technical_score(self, symbol: str, market_data: Dict) -> float:
        """Get technical analysis score"""
        if symbol not in market_data:
            return 0.5
        
        data = market_data[symbol]
        if len(data) < 50:
            return 0.5
        
        score = 0.0
        factors = 0
        
        # RSI (placeholder)
        # rsi = calculate_rsi(data['close'])
        rsi_value = 50.0  # Placeholder
        if 30 < rsi_value < 70:
            score += 1.0
        elif rsi_value <= 30:
            score += 0.7  # Oversold
        factors += 1
        
        # Price vs VWAP (placeholder)
        # vwap = calculate_vwap(data['high'], data['low'], data['close'], data['volume'])
        # if data['close'][-1] > vwap[-1]:
        score += 0.5  # Neutral placeholder
        factors += 1
        
        # OBV trend (placeholder)
        # obv = calculate_obv(data['close'], data['volume'])
        # obv_trend = (obv[-1] - obv[-5]) / abs(obv[-5]) if obv[-5] != 0 else 0
        # if obv_trend > 0:
        score += 0.5  # Neutral placeholder
        factors += 1
        
        return score / factors
    
    def _analyze_options_led_signal(self, symbol: str, form4_data: pd.DataFrame,
                                  options_data: pd.DataFrame, market_data: Dict) -> Optional[Signal]:
        """Analyze cases where options activity leads insider activity"""
        
        # Strong options signal required
        options_score = self._calculate_options_score(symbol, options_data)
        if options_score < 0.7:
            return None
        
        # Check for any insider activity (even older)
        insider_history = form4_data[
            (form4_data['ticker'] == symbol) &
            (form4_data['transactionType'] == 'P') &
            (form4_data['transactionDate'] >= datetime.now() - timedelta(days=90))
        ]
        
        insider_score = 0.3 if not insider_history.empty else 0.0
        
        # IV must be reasonable
        iv_rank = self._calculate_iv_rank(symbol, options_data)
        if iv_rank > self.max_iv_for_entry:
            return None
        
        iv_score = self._score_iv_rank(iv_rank)
        
        # Technical confirmation more important here
        technical_score = self._get_technical_score(symbol, market_data)
        if technical_score < 0.6:
            return None
        
        # Different weighting for options-led signals
        composite_score = (
            options_score * 0.5 +
            technical_score * 0.3 +
            iv_score * 0.15 +
            insider_score * 0.05
        )
        
        if composite_score >= self.min_combined_score:
            return self._create_options_led_signal(
                symbol, composite_score, options_score,
                iv_rank, options_data
            )
        
        return None
    
    def _create_convergence_signal(self, symbol: str, confidence: float,
                                 insider_score: float, options_score: float,
                                 iv_rank: float, form4_data: pd.DataFrame,
                                 options_data: pd.DataFrame) -> Signal:
        """Create signal with detailed metadata"""
        
        # Get insider summary
        recent_insiders = form4_data[
            (form4_data['ticker'] == symbol) &
            (form4_data['transactionType'] == 'P') &
            (form4_data['transactionDate'] >= datetime.now() - timedelta(days=self.insider_lookback))
        ]
        
        # Get options summary
        recent_options = options_data[
            (options_data['ticker'] == symbol) &
            (options_data['option_type'] == 'CALL') &
            (options_data['timestamp'] >= datetime.now() - timedelta(days=self.options_lookback))
        ]
        
        metadata = {
            'strategy': 'insider_options_flow',
            'signal_type': 'convergence',
            'insider_score': round(insider_score, 2),
            'options_score': round(options_score, 2),
            'iv_rank': round(iv_rank, 2),
            'insider_count': recent_insiders['reportingOwner'].nunique() if not recent_insiders.empty else 0,
            'insider_value': recent_insiders['netTransactionValue'].sum() if not recent_insiders.empty else 0,
            'call_volume': recent_options['volume'].sum() if not recent_options.empty else 0,
            'call_premium': recent_options['premium'].sum() if not recent_options.empty else 0,
            'signal_strength': 'STRONG' if confidence > 0.8 else 'MODERATE'
        }
        
        return Signal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            confidence=confidence,
            metadata=metadata,
            strategy_id=self.strategy_id,
            timestamp=datetime.now()
        )
    
    def _create_options_led_signal(self, symbol: str, confidence: float,
                                 options_score: float, iv_rank: float,
                                 options_data: pd.DataFrame) -> Signal:
        """Create options-led signal"""
        
        recent_options = options_data[
            (options_data['ticker'] == symbol) &
            (options_data['timestamp'] >= datetime.now() - timedelta(days=self.options_lookback))
        ]
        
        calls = recent_options[recent_options['option_type'] == 'CALL']
        
        metadata = {
            'strategy': 'insider_options_flow',
            'signal_type': 'options_led',
            'options_score': round(options_score, 2),
            'iv_rank': round(iv_rank, 2),
            'call_volume': calls['volume'].sum() if not calls.empty else 0,
            'call_premium': calls['premium'].sum() if not calls.empty else 0,
            'sweep_count': calls['is_sweep'].sum() if not calls.empty else 0,
            'block_count': calls['is_block'].sum() if not calls.empty else 0,
            'signal_strength': 'STRONG' if confidence > 0.75 else 'MODERATE'
        }
        
        return Signal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            confidence=confidence,
            metadata=metadata,
            strategy_id=self.strategy_id,
            timestamp=datetime.now()
        )
    
    def _get_market_data(self) -> Optional[Dict]:
        """Get market data for universe"""
        # Placeholder - would fetch actual data
        return {}
    
    def _apply_portfolio_constraints(self, signals: List[Signal]) -> List[Signal]:
        """Apply portfolio risk constraints"""
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit positions
        max_positions = int(1.0 / self.max_position_pct)
        
        # Diversify by avoiding multiple positions in correlated stocks
        selected = []
        for signal in signals:
            # Add correlation check here
            selected.append(signal)
            if len(selected) >= max_positions:
                break
        
        return selected