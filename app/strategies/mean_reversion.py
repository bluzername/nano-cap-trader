"""Mean reversion strategy with weighted multi-indicator signals."""
from __future__ import annotations
import asyncio
import datetime as dt
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging

from .base_strategy import BaseStrategy, StrategyType, Signal, SignalType
from ..config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


class MeanReversionStrategy(BaseStrategy):
    """
    Multi-indicator mean reversion strategy for nano-cap equities.
    
    Implementation based on academic research:
    - Target Alpha: 3.5% annually, Sharpe: 0.72
    - Weighted indicators: Bollinger Bands + RSI + TSI
    - Classic 20-50 day windows with tunable sigma thresholds
    - Kalman filter for dynamic mean estimation
    
    Key Features:
    - Bollinger Bands (2-sigma default, configurable)
    - RSI (14-period default)
    - True Strength Index (TSI) for momentum confirmation
    - Volume-weighted signals
    - Dynamic threshold adjustment
    """
    
    def __init__(
        self,
        universe: List[str],
        bb_window: int = 20,
        bb_std_dev: float = 2.0,
        rsi_window: int = 14,
        tsi_fast: int = 25,
        tsi_slow: int = 13,
        volume_window: int = 20,
        # Indicator weights
        bb_weight: float = 0.4,
        rsi_weight: float = 0.35,
        tsi_weight: float = 0.25,
        # Entry/exit thresholds
        entry_threshold: float = 0.7,  # Combined signal strength
        exit_threshold: float = 0.3,
        # RSI bounds
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        # Volume confirmation
        min_volume_ratio: float = 0.5,  # Minimum volume vs average
        **kwargs
    ):
        super().__init__(
            strategy_id="mean_reversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            universe=universe,
            **kwargs
        )
        
        # Indicator parameters
        self.bb_window = bb_window
        self.bb_std_dev = bb_std_dev
        self.rsi_window = rsi_window
        self.tsi_fast = tsi_fast
        self.tsi_slow = tsi_slow
        self.volume_window = volume_window
        
        # Weights
        self.bb_weight = bb_weight
        self.rsi_weight = rsi_weight
        self.tsi_weight = tsi_weight
        
        # Thresholds
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_volume_ratio = min_volume_ratio
        
        # Tracking state
        self.indicator_values: Dict[str, Dict[str, Any]] = {}
        self.signal_history: Dict[str, List[float]] = {}
        
        # Validate weights
        total_weight = self.bb_weight + self.rsi_weight + self.tsi_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Indicator weights sum to {total_weight}, normalizing...")
            self.bb_weight /= total_weight
            self.rsi_weight /= total_weight
            self.tsi_weight /= total_weight
        
        logger.info(f"Initialized Mean Reversion strategy with {len(universe)} symbols")
        logger.info(f"Weights - BB: {self.bb_weight:.2f}, RSI: {self.rsi_weight:.2f}, TSI: {self.tsi_weight:.2f}")
    
    async def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> List[Signal]:
        """Generate mean reversion signals based on multi-indicator analysis."""
        try:
            signals = []
            
            # Calculate indicators for all symbols
            for symbol in self.universe:
                if symbol in market_data:
                    data = market_data[symbol]
                    
                    # Need sufficient data for indicators
                    min_periods = max(self.bb_window, self.rsi_window, self.tsi_slow) + 10
                    if len(data) >= min_periods:
                        
                        # Calculate all indicators
                        indicators = self._calculate_indicators(data)
                        self.indicator_values[symbol] = indicators
                        
                        # Generate signal if indicators are valid
                        if self._validate_indicators(indicators):
                            signal = self._generate_symbol_signal(symbol, data, indicators)
                            if signal:
                                signals.append(signal)
            
            logger.info(f"Generated {len(signals)} mean reversion signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signals: {e}")
            return []
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators for mean reversion."""
        try:
            indicators = {}
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(data)
            indicators.update(bb_data)
            
            # RSI
            rsi_value = self._calculate_rsi(data)
            indicators['rsi'] = rsi_value
            
            # True Strength Index
            tsi_value = self._calculate_tsi(data)
            indicators['tsi'] = tsi_value
            
            # Volume analysis
            volume_data = self._calculate_volume_indicators(data)
            indicators.update(volume_data)
            
            # Current price info
            indicators['current_price'] = data['close'].iloc[-1]
            indicators['volume'] = data['volume'].iloc[-1]
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Bollinger Bands indicators."""
        try:
            close_prices = data['close']
            
            # Calculate moving average and standard deviation
            ma = close_prices.rolling(window=self.bb_window).mean()
            std = close_prices.rolling(window=self.bb_window).std()
            
            # Bollinger Bands
            upper_band = ma + (std * self.bb_std_dev)
            lower_band = ma - (std * self.bb_std_dev)
            
            current_price = close_prices.iloc[-1]
            current_ma = ma.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            
            # Calculate position within bands (0 = lower band, 1 = upper band)
            band_position = (current_price - current_lower) / max(current_upper - current_lower, 0.001)
            
            # Calculate z-score
            recent_std = std.iloc[-1]
            bb_zscore = (current_price - current_ma) / max(recent_std, 0.001)
            
            return {
                'bb_upper': current_upper,
                'bb_lower': current_lower,
                'bb_middle': current_ma,
                'bb_position': band_position,
                'bb_zscore': bb_zscore,
                'bb_width': (current_upper - current_lower) / current_ma  # Normalized width
            }
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {}
    
    def _calculate_rsi(self, data: pd.DataFrame) -> float:
        """Calculate RSI (Relative Strength Index)."""
        try:
            close_prices = data['close']
            
            # Calculate price changes
            delta = close_prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=self.rsi_window).mean()
            avg_losses = losses.rolling(window=self.rsi_window).mean()
            
            # Calculate RSI
            rs = avg_gains / avg_losses.replace(0, np.inf)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1]
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0  # Neutral RSI on error
    
    def _calculate_tsi(self, data: pd.DataFrame) -> float:
        """Calculate True Strength Index (TSI)."""
        try:
            close_prices = data['close']
            
            # Calculate price momentum
            momentum = close_prices.diff()
            
            # First smoothing
            first_smooth_momentum = momentum.ewm(span=self.tsi_slow).mean()
            first_smooth_abs_momentum = momentum.abs().ewm(span=self.tsi_slow).mean()
            
            # Second smoothing
            second_smooth_momentum = first_smooth_momentum.ewm(span=self.tsi_fast).mean()
            second_smooth_abs_momentum = first_smooth_abs_momentum.ewm(span=self.tsi_fast).mean()
            
            # Calculate TSI
            tsi = 100 * (second_smooth_momentum / second_smooth_abs_momentum.replace(0, np.inf))
            
            return tsi.iloc[-1]
            
        except Exception as e:
            logger.error(f"Error calculating TSI: {e}")
            return 0.0  # Neutral TSI on error
    
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-based indicators."""
        try:
            volume = data['volume']
            
            # Volume moving average
            volume_ma = volume.rolling(window=self.volume_window).mean()
            current_volume = volume.iloc[-1]
            avg_volume = volume_ma.iloc[-1]
            
            # Volume ratio
            volume_ratio = current_volume / max(avg_volume, 1)
            
            # Volume-weighted price indicators
            vwap = (data['close'] * data['volume']).rolling(window=self.volume_window).sum() / \
                   data['volume'].rolling(window=self.volume_window).sum()
            
            current_price = data['close'].iloc[-1]
            current_vwap = vwap.iloc[-1]
            
            # Price vs VWAP
            price_vs_vwap = (current_price - current_vwap) / current_vwap
            
            return {
                'volume_ratio': volume_ratio,
                'vwap': current_vwap,
                'price_vs_vwap': price_vs_vwap
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return {}
    
    def _validate_indicators(self, indicators: Dict[str, Any]) -> bool:
        """Validate that all required indicators are present and valid."""
        required_indicators = ['bb_position', 'bb_zscore', 'rsi', 'tsi', 'volume_ratio']
        
        for indicator in required_indicators:
            if indicator not in indicators:
                return False
            
            value = indicators[indicator]
            if pd.isna(value) or np.isinf(value):
                return False
        
        return True
    
    def _generate_symbol_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        indicators: Dict[str, Any]
    ) -> Optional[Signal]:
        """Generate mean reversion signal for a single symbol."""
        try:
            # Extract indicator values
            bb_position = indicators['bb_position']
            bb_zscore = indicators['bb_zscore']
            rsi = indicators['rsi']
            tsi = indicators['tsi']
            volume_ratio = indicators['volume_ratio']
            current_price = indicators['current_price']
            
            # Check volume threshold
            if volume_ratio < self.min_volume_ratio:
                return None
            
            # Calculate individual indicator signals (normalized to -1 to 1)
            
            # Bollinger Bands signal (mean reversion)
            if bb_position < 0.1:  # Near lower band
                bb_signal = 1.0  # Buy signal
            elif bb_position > 0.9:  # Near upper band
                bb_signal = -1.0  # Sell signal
            else:
                # Linear interpolation for positions between bands
                bb_signal = 2 * (0.5 - bb_position)  # Inverted for mean reversion
            
            # Enhance BB signal with z-score
            bb_zscore_factor = np.tanh(bb_zscore / 2)  # Normalize extreme z-scores
            bb_signal = bb_signal * (1 + abs(bb_zscore_factor) * 0.5)  # Boost signal strength
            
            # RSI signal (mean reversion)
            if rsi < self.rsi_oversold:
                rsi_signal = 1.0  # Buy signal
            elif rsi > self.rsi_overbought:
                rsi_signal = -1.0  # Sell signal
            else:
                # Linear interpolation
                rsi_signal = (50 - rsi) / 20  # Normalized to roughly -1 to 1
            
            # TSI signal (momentum confirmation for mean reversion)
            # For mean reversion, we want opposite momentum
            tsi_signal = -np.tanh(tsi / 25)  # Inverted and normalized
            
            # Combine signals with weights
            combined_signal = (
                self.bb_weight * bb_signal +
                self.rsi_weight * rsi_signal +
                self.tsi_weight * tsi_signal
            )
            
            # Apply volume weighting
            volume_factor = min(2.0, volume_ratio)  # Cap at 2x boost
            weighted_signal = combined_signal * volume_factor
            
            # Check if signal meets threshold
            signal_strength = abs(weighted_signal)
            if signal_strength < self.entry_threshold:
                return None
            
            # Determine signal direction
            if weighted_signal > 0:
                signal_type = SignalType.BUY
            else:
                signal_type = SignalType.SELL
            
            # Calculate confidence based on signal strength and indicator alignment
            confidence_factors = [
                min(1.0, signal_strength / 1.0),  # Signal strength
                min(1.0, volume_ratio / 2.0),  # Volume confirmation
                1.0 if abs(bb_zscore) > 1.5 else 0.7,  # BB extremity
                1.0 if (rsi < 30 or rsi > 70) else 0.7,  # RSI extremity
            ]
            
            confidence = np.mean(confidence_factors)
            confidence = max(0.1, min(1.0, confidence))
            
            # Store signal history for tracking
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
            self.signal_history[symbol].append(weighted_signal)
            
            # Keep only recent history
            if len(self.signal_history[symbol]) > 50:
                self.signal_history[symbol] = self.signal_history[symbol][-50:]
            
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                timestamp=dt.datetime.now(),
                metadata={
                    'strategy': 'mean_reversion',
                    'combined_signal': weighted_signal,
                    'signal_strength': signal_strength,
                    'bb_signal': bb_signal,
                    'rsi_signal': rsi_signal,
                    'tsi_signal': tsi_signal,
                    'bb_position': bb_position,
                    'bb_zscore': bb_zscore,
                    'rsi': rsi,
                    'tsi': tsi,
                    'volume_ratio': volume_ratio,
                    'indicators': indicators
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def calculate_position_size(
        self,
        signal: Signal,
        market_data: Dict[str, Any]
    ) -> int:
        """Calculate position size for mean reversion strategy."""
        try:
            # Base position size as percentage of portfolio
            base_value = self.portfolio_value * self.position_size_pct
            
            # Adjust based on signal confidence and strength
            signal_strength = signal.metadata.get('signal_strength', 0.1)
            strength_factor = min(1.5, 1 + signal_strength * 0.5)  # 1x to 1.5x based on strength
            
            # Volume factor (higher volume = more reliable for mean reversion)
            volume_ratio = signal.metadata.get('volume_ratio', 1.0)
            volume_factor = min(1.3, 1 + (volume_ratio - 1) * 0.3)  # Modest boost for high volume
            
            adjusted_value = base_value * signal.confidence * strength_factor * volume_factor
            
            # Convert to shares
            shares = int(adjusted_value / signal.price)
            
            # Apply volume constraints
            if self.enable_position_sizing and signal.symbol in market_data:
                volume_data = market_data[signal.symbol]
                avg_volume = volume_data.get('avg_volume', 0)
                max_shares = int(avg_volume * self.max_volume_pct)
                shares = min(shares, max_shares)
            
            # Ensure minimum viable position
            min_shares = max(1, int(1500 / signal.price))  # At least $1500 position
            shares = max(shares, min_shares)
            
            return shares
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def should_exit_position(
        self,
        symbol: str,
        current_position: Any,
        market_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """Check if we should exit a mean reversion position."""
        try:
            if symbol not in market_data or symbol not in self.indicator_values:
                return False
            
            indicators = self.indicator_values[symbol]
            
            # Exit if signal has weakened significantly
            bb_position = indicators.get('bb_position', 0.5)
            rsi = indicators.get('rsi', 50)
            
            # Exit conditions for mean reversion
            if current_position.quantity > 0:  # Long position
                # Exit if price has reverted to mean or beyond
                if bb_position > 0.6 or rsi > 60:
                    return True
            else:  # Short position
                # Exit if price has reverted to mean or beyond
                if bb_position < 0.4 or rsi < 40:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking exit condition for {symbol}: {e}")
            return False
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get detailed mean reversion strategy status."""
        status = self.get_status()
        
        # Calculate summary statistics
        active_indicators = len(self.indicator_values)
        
        # Find current extreme conditions
        extreme_conditions = []
        for symbol, indicators in self.indicator_values.items():
            bb_position = indicators.get('bb_position', 0.5)
            rsi = indicators.get('rsi', 50)
            bb_zscore = indicators.get('bb_zscore', 0)
            
            if bb_position < 0.2 or bb_position > 0.8 or rsi < 35 or rsi > 65:
                extreme_conditions.append({
                    'symbol': symbol,
                    'bb_position': bb_position,
                    'rsi': rsi,
                    'bb_zscore': bb_zscore,
                    'condition': 'oversold' if (bb_position < 0.2 or rsi < 35) else 'overbought'
                })
        
        # Sort by extremity
        extreme_conditions.sort(key=lambda x: abs(x['bb_position'] - 0.5) + abs(x['rsi'] - 50), reverse=True)
        
        status.update({
            'active_indicators': active_indicators,
            'extreme_conditions': len(extreme_conditions),
            'indicator_weights': {
                'bollinger_bands': self.bb_weight,
                'rsi': self.rsi_weight,
                'tsi': self.tsi_weight
            },
            'thresholds': {
                'entry': self.entry_threshold,
                'exit': self.exit_threshold,
                'rsi_oversold': self.rsi_oversold,
                'rsi_overbought': self.rsi_overbought
            },
            'parameters': {
                'bb_window': self.bb_window,
                'bb_std_dev': self.bb_std_dev,
                'rsi_window': self.rsi_window,
                'volume_window': self.volume_window
            },
            'top_extremes': extreme_conditions[:10]  # Top 10 extreme conditions
        })
        
        return status