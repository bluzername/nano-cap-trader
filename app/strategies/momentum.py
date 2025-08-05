"""Momentum strategy with multi-timeframe signals and news catalysts."""
from __future__ import annotations
import asyncio
import datetime as dt
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging

from .base_strategy import BaseStrategy, StrategyType, Signal, SignalType
from ..data_sources.news_data import NewsDataProvider
from ..config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


class MomentumStrategy(BaseStrategy):
    """
    Multi-timeframe momentum strategy for nano-cap equities.
    
    Implementation based on academic research:
    - Target Alpha: 4.2% annually, Sharpe: 0.58
    - Low-float momentum with news catalyst integration
    - Volume confirmation with configurable thresholds
    - Ensemble weighting across 1/3/5-day timeframes
    
    Key Features:
    - Price-based momentum signals
    - Multi-timeframe ensemble (1D, 3D, 5D)
    - News sentiment integration
    - Volume threshold confirmation
    - Float-weighted scoring for nano-caps
    """
    
    def __init__(
        self,
        universe: List[str],
        volume_threshold_multiplier: float = 3.0,
        momentum_timeframes: List[int] = None,
        float_threshold: float = 30_000_000,  # 30M shares
        news_weight: float = 0.3,
        momentum_weight: float = 0.7,
        min_momentum_threshold: float = 0.05,  # 5% minimum move
        max_momentum_threshold: float = 0.50,  # 50% maximum move filter
        **kwargs
    ):
        # Filter kwargs to avoid conflicts with explicit parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['strategy_id', 'strategy_type', 'universe']}
        super().__init__(
            strategy_id="momentum",
            strategy_type=StrategyType.MOMENTUM,
            universe=universe,
            **filtered_kwargs
        )
        
        # Strategy parameters
        self.volume_threshold_multiplier = volume_threshold_multiplier
        self.momentum_timeframes = momentum_timeframes or [1, 3, 5]
        self.float_threshold = float_threshold
        self.news_weight = news_weight
        self.momentum_weight = momentum_weight
        self.min_momentum_threshold = min_momentum_threshold
        self.max_momentum_threshold = max_momentum_threshold
        
        # Ensemble weights for different timeframes (shorter = higher weight for nano-caps)
        self.timeframe_weights = {
            1: 0.5,   # 1-day: highest weight for short-term momentum
            3: 0.3,   # 3-day: medium weight
            5: 0.2,   # 5-day: lower weight
            10: 0.1,  # 10-day: if used
            20: 0.05  # 20-day: if used
        }
        
        # News data provider
        self.news_provider = NewsDataProvider()
        
        # Tracking state
        self.momentum_scores: Dict[str, Dict[str, float]] = {}
        self.volume_ratios: Dict[str, float] = {}
        self.news_scores: Dict[str, float] = {}
        self.float_data: Dict[str, float] = {}
        
        logger.info(f"Initialized Momentum strategy with {len(universe)} symbols")
        logger.info(f"Timeframes: {self.momentum_timeframes}, Volume threshold: {volume_threshold_multiplier}x")
    
    async def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> List[Signal]:
        """Generate momentum signals based on multi-timeframe analysis, news, and insider activity."""
        try:
            signals = []
            
            # Get Form 4 insider data (CRITICAL FIX: This was missing!)
            insider_data = kwargs.get('form4_data', pd.DataFrame())
            if not insider_data.empty:
                logger.info(f"Processing insider data for {len(insider_data)} transactions")
            
            # Get float data for nano-cap filtering
            await self._update_float_data(market_data)
            
            # Calculate momentum scores for each timeframe
            await self._calculate_momentum_scores(market_data)
            
            # Calculate volume ratios
            self._calculate_volume_ratios(market_data)
            
            # Get news sentiment scores
            await self._calculate_news_scores()
            
            # Calculate insider momentum scores (NEW!)
            insider_scores = self._calculate_insider_momentum(insider_data)
            
            # Generate signals for each symbol
            for symbol in self.universe:
                if symbol in market_data and len(market_data[symbol]) > max(self.momentum_timeframes):
                    signal = await self._generate_symbol_signal(symbol, market_data[symbol], insider_scores.get(symbol, 0))
                    if signal:
                        signals.append(signal)
            
            logger.info(f"Generated {len(signals)} momentum signals with insider integration")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating momentum signals: {e}")
            return []
    
    async def _update_float_data(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Update float shares data for nano-cap filtering."""
        try:
            # This would typically come from fundamental data
            # For now, we'll simulate based on market cap and price
            for symbol in self.universe:
                if symbol in market_data:
                    # Estimate float from volume patterns (higher volume = likely higher float)
                    avg_volume = market_data[symbol]['volume'].tail(20).mean()
                    
                    # Rough estimate: nano-caps typically trade 1-5% of float daily
                    estimated_float = avg_volume * 50  # Assume 2% of float trades daily
                    self.float_data[symbol] = min(estimated_float, 100_000_000)  # Cap at 100M
                    
        except Exception as e:
            logger.error(f"Error updating float data: {e}")
    
    async def _calculate_momentum_scores(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Calculate momentum scores for all timeframes."""
        try:
            for symbol in self.universe:
                if symbol not in market_data:
                    continue
                
                data = market_data[symbol]
                if len(data) < max(self.momentum_timeframes):
                    continue
                
                symbol_scores = {}
                
                for timeframe in self.momentum_timeframes:
                    if len(data) >= timeframe:
                        # Calculate price momentum
                        current_price = data['close'].iloc[-1]
                        past_price = data['close'].iloc[-timeframe]
                        
                        momentum = (current_price / past_price) - 1
                        
                        # Apply float weighting (higher weight for lower float)
                        float_shares = self.float_data.get(symbol, 50_000_000)
                        float_weight = 1.0 if float_shares > self.float_threshold else 1.5
                        
                        # Volume-weighted momentum (higher volume = more reliable)
                        recent_volume = data['volume'].tail(timeframe).mean()
                        historical_volume = data['volume'].tail(timeframe * 4).mean()
                        volume_factor = min(3.0, recent_volume / max(historical_volume, 1))
                        
                        # Final momentum score
                        weighted_momentum = momentum * float_weight * volume_factor
                        symbol_scores[timeframe] = weighted_momentum
                
                self.momentum_scores[symbol] = symbol_scores
                
        except Exception as e:
            logger.error(f"Error calculating momentum scores: {e}")
    
    def _calculate_volume_ratios(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Calculate current volume vs average volume ratios."""
        try:
            for symbol in self.universe:
                if symbol not in market_data:
                    continue
                
                data = market_data[symbol]
                if len(data) < 20:
                    continue
                
                current_volume = data['volume'].iloc[-1]
                avg_volume = data['volume'].tail(20).mean()
                
                volume_ratio = current_volume / max(avg_volume, 1)
                self.volume_ratios[symbol] = volume_ratio
                
        except Exception as e:
            logger.error(f"Error calculating volume ratios: {e}")
    
    async def _calculate_news_scores(self) -> None:
        """Calculate news sentiment scores for all symbols."""
        try:
            # Get recent news for all symbols
            news_data = await self.news_provider.get_news_for_symbols(
                self.universe, hours_back=24, max_items=10
            )
            
            for symbol, news_items in news_data.items():
                if news_items:
                    # Calculate momentum score from news
                    momentum_score = self.news_provider.calculate_news_momentum(news_items)
                    
                    # Normalize to 0-1 range
                    normalized_score = min(1.0, momentum_score / 5.0)
                    self.news_scores[symbol] = normalized_score
                else:
                    self.news_scores[symbol] = 0.0
                    
        except Exception as e:
            logger.error(f"Error calculating news scores: {e}")
            # Set all scores to 0 on error
            for symbol in self.universe:
                self.news_scores[symbol] = 0.0
    
    def _calculate_insider_momentum(self, insider_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum scores based on insider trading activity (CRITICAL FIX)."""
        from ..signals import insider_buy_score
        
        insider_scores = {}
        
        if insider_data.empty:
            return {symbol: 0.0 for symbol in self.universe}
        
        try:
            # Use the existing insider_buy_score function
            insider_buy_scores = insider_buy_score(insider_data)
            
            # Convert to momentum signals
            for symbol in self.universe:
                if symbol in insider_buy_scores.index:
                    # Insider buying creates positive momentum
                    # Scale insider score (0-1 range) to momentum strength
                    raw_score = insider_buy_scores.loc[symbol]
                    
                    # Recent insider buying (last 30 days) gets momentum boost
                    if 'symbol' in insider_data.columns and 'transactionType' in insider_data.columns:
                        recent_insider_mask = (
                            (insider_data['symbol'] == symbol) &
                            (insider_data['transactionType'] == 'P') &  # Purchase
                            (pd.to_datetime(insider_data.get('timestamp', pd.Series()), errors='coerce') >= 
                             pd.Timestamp.now() - pd.Timedelta(days=30))
                        )
                        
                        recent_insider_activity = insider_data[recent_insider_mask]
                        
                        # Boost momentum for recent activity
                        momentum_multiplier = 1.0
                        if len(recent_insider_activity) > 0:
                            # More recent activity = higher momentum
                            total_recent_value = recent_insider_activity.get('transactionValue', pd.Series(0)).sum()
                            if total_recent_value > 100_000:  # $100k+ insider buying
                                momentum_multiplier = 2.0
                            elif total_recent_value > 50_000:  # $50k+ insider buying
                                momentum_multiplier = 1.5
                    else:
                        momentum_multiplier = 1.0
                    
                    insider_scores[symbol] = raw_score * momentum_multiplier
                else:
                    insider_scores[symbol] = 0.0
            
            logger.info(f"Calculated insider momentum for {len([s for s in insider_scores.values() if s > 0])} symbols")
            return insider_scores
            
        except Exception as e:
            logger.error(f"Error calculating insider momentum: {e}")
            return {symbol: 0.0 for symbol in self.universe}
    
    async def _generate_symbol_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        insider_score: float = 0.0
    ) -> Optional[Signal]:
        """Generate momentum signal for a single symbol."""
        try:
            # Get momentum scores for all timeframes
            symbol_momentum = self.momentum_scores.get(symbol, {})
            if not symbol_momentum:
                return None
            
            # Calculate ensemble momentum score
            ensemble_score = 0.0
            total_weight = 0.0
            
            for timeframe, momentum in symbol_momentum.items():
                weight = self.timeframe_weights.get(timeframe, 0.1)
                ensemble_score += momentum * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_score /= total_weight
            
            # Check minimum momentum threshold
            if abs(ensemble_score) < self.min_momentum_threshold:
                return None
            
            # Check maximum momentum threshold (avoid bubble stocks)
            if abs(ensemble_score) > self.max_momentum_threshold:
                logger.warning(f"Skipping {symbol}: momentum {ensemble_score:.3f} exceeds max threshold")
                return None
            
            # Get volume confirmation
            volume_ratio = self.volume_ratios.get(symbol, 1.0)
            if volume_ratio < self.volume_threshold_multiplier:
                return None  # Insufficient volume confirmation
            
            # Get news score
            news_score = self.news_scores.get(symbol, 0.0)
            
            # Combine momentum, news, and insider scores (CRITICAL FIX)
            combined_score = (
                ensemble_score * self.momentum_weight +
                news_score * self.news_weight * np.sign(ensemble_score) +
                insider_score * 0.4 * np.sign(ensemble_score)  # 40% weight for insider signals
            )
            
            # Determine signal direction
            if combined_score > 0:
                signal_type = SignalType.BUY
            elif combined_score < 0:
                signal_type = SignalType.SELL
            else:
                return None
            
            # Calculate confidence based on multiple factors
            confidence_factors = [
                min(1.0, abs(combined_score) / 0.2),  # Momentum strength
                min(1.0, volume_ratio / self.volume_threshold_multiplier / 2),  # Volume confirmation
                min(1.0, news_score * 2),  # News confirmation
                1.0 if self.float_data.get(symbol, 50_000_000) < self.float_threshold else 0.8  # Float bonus
            ]
            
            confidence = np.mean(confidence_factors)
            confidence = max(0.1, min(1.0, confidence))  # Ensure reasonable bounds
            
            current_price = data['close'].iloc[-1]
            
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                timestamp=dt.datetime.now(),
                metadata={
                    'strategy': 'momentum',
                    'ensemble_momentum': ensemble_score,
                    'news_score': news_score,
                    'volume_ratio': volume_ratio,
                    'combined_score': combined_score,
                    'timeframe_scores': symbol_momentum,
                    'float_shares': self.float_data.get(symbol, 0),
                    'signal_strength': abs(combined_score)
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
        """Calculate position size for momentum strategy."""
        try:
            # Base position size as percentage of portfolio
            base_value = self.portfolio_value * self.position_size_pct
            
            # Adjust based on signal confidence and momentum strength
            momentum_multiplier = signal.metadata.get('signal_strength', 0.1)
            momentum_factor = min(2.0, 1 + momentum_multiplier)  # 1x to 2x based on momentum
            
            adjusted_value = base_value * signal.confidence * momentum_factor
            
            # Convert to shares
            shares = int(adjusted_value / signal.price)
            
            # Apply volume constraints
            if self.enable_position_sizing and signal.symbol in market_data:
                volume_data = market_data[signal.symbol]
                avg_volume = volume_data.get('avg_volume', 0)
                
                # Be more conservative with momentum positions (use 2% instead of 3%)
                max_shares = int(avg_volume * (self.max_volume_pct * 0.67))
                shares = min(shares, max_shares)
            
            # Ensure minimum viable position
            min_shares = max(1, int(2000 / signal.price))  # At least $2000 position for momentum
            shares = max(shares, min_shares)
            
            return shares
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get detailed momentum strategy status."""
        status = self.get_status()
        
        # Calculate summary statistics
        momentum_symbols = len([s for s in self.momentum_scores.values() if s])
        high_volume_symbols = len([r for r in self.volume_ratios.values() if r >= self.volume_threshold_multiplier])
        news_symbols = len([s for s in self.news_scores.values() if s > 0])
        
        # Top momentum candidates
        top_momentum = []
        for symbol, scores in self.momentum_scores.items():
            if scores:
                ensemble_score = np.mean(list(scores.values()))
                volume_ratio = self.volume_ratios.get(symbol, 0)
                news_score = self.news_scores.get(symbol, 0)
                
                if volume_ratio >= self.volume_threshold_multiplier and abs(ensemble_score) >= self.min_momentum_threshold:
                    top_momentum.append({
                        'symbol': symbol,
                        'momentum': ensemble_score,
                        'volume_ratio': volume_ratio,
                        'news_score': news_score
                    })
        
        # Sort by combined score
        top_momentum.sort(key=lambda x: abs(x['momentum']) + x['news_score'], reverse=True)
        
        status.update({
            'momentum_symbols': momentum_symbols,
            'high_volume_symbols': high_volume_symbols,
            'news_symbols': news_symbols,
            'volume_threshold': self.volume_threshold_multiplier,
            'timeframes': self.momentum_timeframes,
            'top_candidates': top_momentum[:10],  # Top 10 candidates
            'float_threshold': self.float_threshold,
            'momentum_range': {
                'min_threshold': self.min_momentum_threshold,
                'max_threshold': self.max_momentum_threshold
            }
        })
        
        return status