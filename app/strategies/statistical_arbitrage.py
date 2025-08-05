"""Statistical Arbitrage strategy using hierarchical clustering and cointegration."""
from __future__ import annotations
import asyncio
import datetime as dt
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging

from .base_strategy import BaseStrategy, StrategyType, Signal, SignalType
from ..data_sources.correlation_data import CorrelationDataProvider
from ..config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Statistical Arbitrage strategy based on academic research.
    
    Implementation follows best practices from:
    - "Deep Learning Statistical Arbitrage" (2024)
    - "Machine Learning for Pairs Trading: A Clustering-based Approach" (2025)
    
    Key Features:
    - Hierarchical clustering on 60-day correlation matrices
    - 0.8 correlation threshold for pair selection
    - Cointegration testing using Engle-Granger
    - Z-score based entry/exit signals
    - Target Alpha: 4.5% annually, Sharpe: 0.89
    """
    
    def __init__(
        self,
        universe: List[str],
        lookback_days: int = 60,
        correlation_threshold: float = 0.8,
        z_score_entry: float = 2.0,
        z_score_exit: float = 0.5,
        cointegration_p_value: float = 0.05,
        max_pairs: int = 20,
        **kwargs
    ):
        # Filter kwargs to avoid conflicts with explicit parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['strategy_id', 'strategy_type', 'universe']}
        super().__init__(
            strategy_id="statistical_arbitrage",
            strategy_type=StrategyType.STATISTICAL_ARBITRAGE,
            universe=universe,
            **filtered_kwargs
        )
        
        # Strategy parameters
        self.lookback_days = lookback_days
        self.correlation_threshold = correlation_threshold
        self.z_score_entry = z_score_entry
        self.z_score_exit = z_score_exit
        self.cointegration_p_value = cointegration_p_value
        self.max_pairs = max_pairs
        
        # Data provider
        self.correlation_provider = CorrelationDataProvider()
        
        # Strategy state
        self.active_pairs: Dict[str, Dict[str, Any]] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.clusters: Dict[str, int] = {}
        self.cointegrated_pairs: List[Tuple[str, str, float]] = []
        self.last_cluster_update: Optional[dt.datetime] = None
        self.cluster_update_frequency = dt.timedelta(days=7)  # Weekly clustering
        
        logger.info(f"Initialized Statistical Arbitrage strategy with {len(universe)} symbols")
    
    async def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> List[Signal]:
        """Generate statistical arbitrage signals based on pair relationships."""
        try:
            signals = []
            
            # Update clustering if needed
            await self._update_clustering_if_needed()
            
            # Generate signals for active pairs
            for pair_key, pair_info in self.active_pairs.items():
                pair_signals = await self._generate_pair_signals(pair_info, market_data)
                signals.extend(pair_signals)
            
            # Look for new pair opportunities
            if len(self.active_pairs) < self.max_pairs:
                new_signals = await self._find_new_pair_opportunities(market_data)
                signals.extend(new_signals)
            
            logger.info(f"Generated {len(signals)} statistical arbitrage signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating statistical arbitrage signals: {e}")
            return []
    
    async def _update_clustering_if_needed(self) -> None:
        """Update clustering and pair relationships if needed."""
        try:
            now = dt.datetime.now()
            
            # Check if update is needed
            if (self.last_cluster_update is None or 
                now - self.last_cluster_update > self.cluster_update_frequency):
                
                logger.info("Updating correlation clustering...")
                
                # Get correlation matrix
                self.correlation_matrix = await self.correlation_provider.get_correlation_matrix(
                    self.universe, self.lookback_days
                )
                
                if not self.correlation_matrix.empty:
                    # Perform hierarchical clustering
                    n_clusters = min(10, len(self.universe) // 5)  # Adaptive cluster count
                    self.clusters = self.correlation_provider.hierarchical_clustering(
                        self.correlation_matrix, n_clusters
                    )
                    
                    # Find cointegrated pairs
                    self.cointegrated_pairs = await self.correlation_provider.get_cointegration_pairs(
                        self.universe, self.lookback_days, self.cointegration_p_value
                    )
                    
                    self.last_cluster_update = now
                    logger.info(f"Updated clustering: {len(self.clusters)} symbols in {n_clusters} clusters")
                    logger.info(f"Found {len(self.cointegrated_pairs)} cointegrated pairs")
                
        except Exception as e:
            logger.error(f"Error updating clustering: {e}")
    
    async def _generate_pair_signals(
        self,
        pair_info: Dict[str, Any],
        market_data: Dict[str, pd.DataFrame]
    ) -> List[Signal]:
        """Generate signals for an active pair."""
        try:
            signals = []
            symbol1, symbol2 = pair_info['symbol1'], pair_info['symbol2']
            
            # Get current prices
            if symbol1 not in market_data or symbol2 not in market_data:
                return signals
            
            price1 = market_data[symbol1]['close'].iloc[-1]
            price2 = market_data[symbol2]['close'].iloc[-1]
            
            # Calculate current spread ratio
            current_ratio = price1 / price2
            
            # Update spread statistics
            spread_ratios = pair_info.get('spread_ratios', [])
            spread_ratios.append(current_ratio)
            
            # Keep only recent ratios for z-score calculation
            if len(spread_ratios) > self.lookback_days:
                spread_ratios = spread_ratios[-self.lookback_days:]
            
            pair_info['spread_ratios'] = spread_ratios
            
            # Need minimum data points
            if len(spread_ratios) < 20:
                return signals
            
            # Calculate z-score
            mean_ratio = np.mean(spread_ratios)
            std_ratio = np.std(spread_ratios)
            
            if std_ratio == 0:
                return signals
            
            z_score = (current_ratio - mean_ratio) / std_ratio
            pair_info['current_z_score'] = z_score
            
            # Generate entry signals
            current_position = pair_info.get('position', 'none')
            
            if current_position == 'none':
                # Entry signals
                if z_score > self.z_score_entry:
                    # Spread is high: short symbol1, long symbol2
                    signals.append(Signal(
                        symbol=symbol1,
                        signal_type=SignalType.SELL,
                        confidence=min(1.0, abs(z_score) / 4.0),
                        price=price1,
                        timestamp=dt.datetime.now(),
                        metadata={
                            'strategy': 'statistical_arbitrage',
                            'pair': f"{symbol1}_{symbol2}",
                            'z_score': z_score,
                            'signal_type': 'mean_reversion_short'
                        }
                    ))
                    
                    signals.append(Signal(
                        symbol=symbol2,
                        signal_type=SignalType.BUY,
                        confidence=min(1.0, abs(z_score) / 4.0),
                        price=price2,
                        timestamp=dt.datetime.now(),
                        metadata={
                            'strategy': 'statistical_arbitrage',
                            'pair': f"{symbol1}_{symbol2}",
                            'z_score': z_score,
                            'signal_type': 'mean_reversion_long'
                        }
                    ))
                    
                    pair_info['position'] = 'short_symbol1'
                    pair_info['entry_z_score'] = z_score
                
                elif z_score < -self.z_score_entry:
                    # Spread is low: long symbol1, short symbol2
                    signals.append(Signal(
                        symbol=symbol1,
                        signal_type=SignalType.BUY,
                        confidence=min(1.0, abs(z_score) / 4.0),
                        price=price1,
                        timestamp=dt.datetime.now(),
                        metadata={
                            'strategy': 'statistical_arbitrage',
                            'pair': f"{symbol1}_{symbol2}",
                            'z_score': z_score,
                            'signal_type': 'mean_reversion_long'
                        }
                    ))
                    
                    signals.append(Signal(
                        symbol=symbol2,
                        signal_type=SignalType.SELL,
                        confidence=min(1.0, abs(z_score) / 4.0),
                        price=price2,
                        timestamp=dt.datetime.now(),
                        metadata={
                            'strategy': 'statistical_arbitrage',
                            'pair': f"{symbol1}_{symbol2}",
                            'z_score': z_score,
                            'signal_type': 'mean_reversion_short'
                        }
                    ))
                    
                    pair_info['position'] = 'long_symbol1'
                    pair_info['entry_z_score'] = z_score
            
            else:
                # Exit signals
                entry_z_score = pair_info.get('entry_z_score', 0)
                
                # Exit when z-score approaches zero or reverses significantly
                should_exit = False
                
                if abs(z_score) < self.z_score_exit:
                    should_exit = True
                    exit_reason = 'mean_reversion'
                elif (current_position == 'short_symbol1' and z_score < -self.z_score_entry) or \
                     (current_position == 'long_symbol1' and z_score > self.z_score_entry):
                    should_exit = True
                    exit_reason = 'trend_continuation'
                
                if should_exit:
                    # Generate exit signals (opposite of entry)
                    if current_position == 'short_symbol1':
                        signals.append(Signal(
                            symbol=symbol1,
                            signal_type=SignalType.BUY,  # Cover short
                            confidence=0.8,
                            price=price1,
                            timestamp=dt.datetime.now(),
                            metadata={
                                'strategy': 'statistical_arbitrage',
                                'pair': f"{symbol1}_{symbol2}",
                                'z_score': z_score,
                                'signal_type': 'exit',
                                'exit_reason': exit_reason
                            }
                        ))
                        
                        signals.append(Signal(
                            symbol=symbol2,
                            signal_type=SignalType.SELL,  # Sell long
                            confidence=0.8,
                            price=price2,
                            timestamp=dt.datetime.now(),
                            metadata={
                                'strategy': 'statistical_arbitrage',
                                'pair': f"{symbol1}_{symbol2}",
                                'z_score': z_score,
                                'signal_type': 'exit',
                                'exit_reason': exit_reason
                            }
                        ))
                    
                    elif current_position == 'long_symbol1':
                        signals.append(Signal(
                            symbol=symbol1,
                            signal_type=SignalType.SELL,  # Sell long
                            confidence=0.8,
                            price=price1,
                            timestamp=dt.datetime.now(),
                            metadata={
                                'strategy': 'statistical_arbitrage',
                                'pair': f"{symbol1}_{symbol2}",
                                'z_score': z_score,
                                'signal_type': 'exit',
                                'exit_reason': exit_reason
                            }
                        ))
                        
                        signals.append(Signal(
                            symbol=symbol2,
                            signal_type=SignalType.BUY,  # Cover short
                            confidence=0.8,
                            price=price2,
                            timestamp=dt.datetime.now(),
                            metadata={
                                'strategy': 'statistical_arbitrage',
                                'pair': f"{symbol1}_{symbol2}",
                                'z_score': z_score,
                                'signal_type': 'exit',
                                'exit_reason': exit_reason
                            }
                        ))
                    
                    pair_info['position'] = 'none'
                    pair_info['entry_z_score'] = None
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating pair signals: {e}")
            return []
    
    async def _find_new_pair_opportunities(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> List[Signal]:
        """Find new pair trading opportunities."""
        try:
            if not self.correlation_matrix.empty and self.clusters:
                # Get high-correlation pairs from clusters
                cluster_pairs = self.correlation_provider.get_cluster_pairs(
                    self.correlation_matrix, self.clusters, self.correlation_threshold
                )
                
                # Combine with cointegrated pairs
                all_potential_pairs = []
                
                # Add cluster pairs
                for symbol1, symbol2, correlation in cluster_pairs:
                    pair_key = f"{symbol1}_{symbol2}"
                    if pair_key not in self.active_pairs:
                        all_potential_pairs.append((symbol1, symbol2, abs(correlation), 'correlation'))
                
                # Add cointegrated pairs (prioritize these)
                for symbol1, symbol2, p_value in self.cointegrated_pairs:
                    pair_key = f"{symbol1}_{symbol2}"
                    if pair_key not in self.active_pairs:
                        # Lower p-value = stronger cointegration = higher score
                        score = 1.0 - p_value
                        all_potential_pairs.append((symbol1, symbol2, score, 'cointegration'))
                
                # Sort by score and take best candidates
                all_potential_pairs.sort(key=lambda x: x[2], reverse=True)
                
                # Initialize new pairs
                for symbol1, symbol2, score, pair_type in all_potential_pairs[:5]:  # Max 5 new pairs
                    if len(self.active_pairs) >= self.max_pairs:
                        break
                    
                    # Check if we have data for both symbols
                    if symbol1 in market_data and symbol2 in market_data:
                        pair_key = f"{symbol1}_{symbol2}"
                        
                        self.active_pairs[pair_key] = {
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'pair_type': pair_type,
                            'score': score,
                            'position': 'none',
                            'spread_ratios': [],
                            'created_at': dt.datetime.now()
                        }
                        
                        logger.info(f"Added new pair: {pair_key} ({pair_type}, score: {score:.3f})")
            
            return []  # New pairs don't generate immediate signals
            
        except Exception as e:
            logger.error(f"Error finding new pair opportunities: {e}")
            return []
    
    def calculate_position_size(
        self,
        signal: Signal,
        market_data: Dict[str, Any]
    ) -> int:
        """Calculate position size for statistical arbitrage."""
        try:
            # Base position size as percentage of portfolio
            base_value = self.portfolio_value * self.position_size_pct
            
            # Adjust based on signal confidence
            adjusted_value = base_value * signal.confidence
            
            # Convert to shares
            shares = int(adjusted_value / signal.price)
            
            # Apply volume constraints if enabled
            if self.enable_position_sizing and signal.symbol in market_data:
                volume_data = market_data[signal.symbol]
                avg_volume = volume_data.get('avg_volume', 0)
                max_shares = int(avg_volume * self.max_volume_pct)
                shares = min(shares, max_shares)
            
            # Ensure minimum viable position
            min_shares = max(1, int(1000 / signal.price))  # At least $1000 position
            shares = max(shares, min_shares)
            
            return shares
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get detailed strategy status."""
        status = self.get_status()
        
        # Add strategy-specific information
        status.update({
            'active_pairs': len(self.active_pairs),
            'cointegrated_pairs': len(self.cointegrated_pairs),
            'clusters': len(set(self.clusters.values())) if self.clusters else 0,
            'last_cluster_update': self.last_cluster_update.isoformat() if self.last_cluster_update else None,
            'pair_details': {
                pair_key: {
                    'symbols': f"{info['symbol1']}/{info['symbol2']}",
                    'type': info['pair_type'],
                    'position': info['position'],
                    'current_z_score': info.get('current_z_score', 0),
                    'score': info['score']
                }
                for pair_key, info in self.active_pairs.items()
            }
        })
        
        return status