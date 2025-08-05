"""Strategy factory for creating and managing trading strategies."""
from __future__ import annotations
from typing import Dict, List, Optional, Any, Type
import logging

from .base_strategy import BaseStrategy, StrategyType
from .statistical_arbitrage import StatisticalArbitrageStrategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .multi_strategy import MultiStrategy
from ..config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


class StrategyFactory:
    """Factory for creating and managing trading strategies."""
    
    # Strategy registry
    _strategies: Dict[str, Type[BaseStrategy]] = {
        'statistical_arbitrage': StatisticalArbitrageStrategy,
        'momentum': MomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'multi_strategy': MultiStrategy,
    }
    
    @classmethod
    def create_strategy(
        cls,
        strategy_type: str,
        universe: List[str],
        **kwargs
    ) -> Optional[BaseStrategy]:
        """Create a strategy instance."""
        try:
            if strategy_type not in cls._strategies:
                logger.error(f"Unknown strategy type: {strategy_type}")
                return None
            
            strategy_class = cls._strategies[strategy_type]
            
            # Get default parameters from settings
            default_params = cls._get_default_params(strategy_type)
            default_params.update(kwargs)
            
            # Special handling for multi_strategy to avoid parameter conflicts
            if strategy_type == 'multi_strategy':
                # Remove parameters that MultiStrategy handles internally
                filtered_params = {k: v for k, v in default_params.items() 
                                 if k not in ['strategy_id', 'strategy_type']}
                strategy = strategy_class(universe=universe, **filtered_params)
            else:
                strategy = strategy_class(universe=universe, **default_params)
            
            logger.info(f"Created {strategy_type} strategy with {len(universe)} symbols")
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating {strategy_type} strategy: {e}")
            return None
    
    @classmethod
    def create_multiple_strategies(
        cls,
        strategy_configs: List[Dict[str, Any]],
        universe: List[str]
    ) -> Dict[str, BaseStrategy]:
        """Create multiple strategies from configuration."""
        strategies = {}
        
        for config in strategy_configs:
            strategy_type = config.get('type')
            strategy_id = config.get('id', strategy_type)
            params = config.get('params', {})
            
            if strategy_type:
                strategy = cls.create_strategy(strategy_type, universe, **params)
                if strategy:
                    strategies[strategy_id] = strategy
                    
        logger.info(f"Created {len(strategies)} strategies")
        return strategies
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategy types."""
        return list(cls._strategies.keys())
    
    @classmethod
    def get_strategy_info(cls, strategy_type: str) -> Dict[str, Any]:
        """Get information about a strategy type."""
        if strategy_type not in cls._strategies:
            return {}
        
        strategy_class = cls._strategies[strategy_type]
        
        # Extract docstring info
        docstring = strategy_class.__doc__ or ""
        lines = docstring.strip().split('\n')
        description = lines[0] if lines else "No description available"
        
        # Get default parameters
        default_params = cls._get_default_params(strategy_type)
        
        return {
            'type': strategy_type,
            'class_name': strategy_class.__name__,
            'description': description,
            'default_params': default_params,
            'expected_performance': cls._get_expected_performance(strategy_type)
        }
    
    @classmethod
    def _get_default_params(cls, strategy_type: str) -> Dict[str, Any]:
        """Get default parameters for a strategy type."""
        base_params = {
            'max_positions': 50,
            'position_size_pct': 0.02,
            'enable_stop_loss': True,
            'stop_loss_pct': 0.02,
            'enable_position_sizing': True,
            'max_volume_pct': 0.03,
        }
        
        # Strategy-specific defaults
        strategy_defaults = {
            'statistical_arbitrage': {
                'lookback_days': 60,
                'correlation_threshold': 0.8,
                'z_score_entry': 2.0,
                'z_score_exit': 0.5,
                'cointegration_p_value': 0.05,
                'max_pairs': 20,
            },
            'momentum': {
                'volume_threshold_multiplier': getattr(_settings, 'momentum_volume_threshold', 3.0),
                'momentum_timeframes': [1, 3, 5],
                'float_threshold': 30_000_000,
                'news_weight': 0.3,
                'momentum_weight': 0.7,
                'min_momentum_threshold': 0.05,
                'max_momentum_threshold': 0.50,
            },
            'mean_reversion': {
                'bb_window': 20,
                'bb_std_dev': 2.0,
                'rsi_window': 14,
                'tsi_fast': 25,
                'tsi_slow': 13,
                'bb_weight': 0.4,
                'rsi_weight': 0.35,
                'tsi_weight': 0.25,
                'entry_threshold': 0.7,
                'exit_threshold': 0.3,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
            },
            'multi_strategy': {
                'stat_arb_weight': 0.60,
                'momentum_weight': 0.25,
                'mean_rev_weight': 0.15,
                'min_confidence_threshold': 0.3,
                'max_correlation_threshold': 0.7,
                'diversification_bonus': 0.1,
                'rebalance_frequency_hours': 6,
            }
        }
        
        # Combine base and strategy-specific params
        params = base_params.copy()
        if strategy_type in strategy_defaults:
            params.update(strategy_defaults[strategy_type])
        
        return params
    
    @classmethod
    def _get_expected_performance(cls, strategy_type: str) -> Dict[str, float]:
        """Get expected performance metrics for a strategy type."""
        performance_targets = {
            'statistical_arbitrage': {
                'annual_alpha': 0.045,  # 4.5%
                'sharpe_ratio': 0.89,
                'max_drawdown': -0.08,  # -8%
                'win_rate': 0.58,
            },
            'momentum': {
                'annual_alpha': 0.042,  # 4.2%
                'sharpe_ratio': 0.58,
                'max_drawdown': -0.15,  # -15%
                'win_rate': 0.52,
            },
            'mean_reversion': {
                'annual_alpha': 0.035,  # 3.5%
                'sharpe_ratio': 0.72,
                'max_drawdown': -0.10,  # -10%
                'win_rate': 0.65,
            },
            'multi_strategy': {
                'annual_alpha': 0.040,  # 4.0% (weighted average)
                'sharpe_ratio': 0.70,
                'max_drawdown': -0.12,  # -12%
                'win_rate': 0.60,
            }
        }
        
        return performance_targets.get(strategy_type, {})
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> Dict[str, BaseStrategy]:
        """Create strategies from configuration dictionary."""
        strategies = {}
        
        # Get universe
        universe = config.get('universe', [])
        if not universe:
            logger.error("No universe specified in configuration")
            return strategies
        
        # Get strategy configurations
        strategy_configs = config.get('strategies', [])
        
        for strategy_config in strategy_configs:
            strategy_type = strategy_config.get('type')
            enabled = strategy_config.get('enabled', True)
            
            if not enabled:
                continue
            
            if strategy_type:
                strategy_id = strategy_config.get('id', f"{strategy_type}_1")
                params = strategy_config.get('params', {})
                
                strategy = cls.create_strategy(strategy_type, universe, **params)
                if strategy:
                    strategies[strategy_id] = strategy
        
        return strategies
    
    @classmethod
    def validate_strategy_config(cls, config: Dict[str, Any]) -> List[str]:
        """Validate strategy configuration and return any errors."""
        errors = []
        
        # Check required fields
        if 'universe' not in config or not config['universe']:
            errors.append("Universe is required and cannot be empty")
        
        if 'strategies' not in config or not config['strategies']:
            errors.append("At least one strategy configuration is required")
        
        # Validate each strategy config
        for i, strategy_config in enumerate(config.get('strategies', [])):
            strategy_type = strategy_config.get('type')
            
            if not strategy_type:
                errors.append(f"Strategy {i}: type is required")
                continue
            
            if strategy_type not in cls._strategies:
                errors.append(f"Strategy {i}: unknown type '{strategy_type}'")
                continue
            
            # Validate parameters
            params = strategy_config.get('params', {})
            param_errors = cls._validate_strategy_params(strategy_type, params)
            for error in param_errors:
                errors.append(f"Strategy {i} ({strategy_type}): {error}")
        
        return errors
    
    @classmethod
    def _validate_strategy_params(cls, strategy_type: str, params: Dict[str, Any]) -> List[str]:
        """Validate parameters for a specific strategy type."""
        errors = []
        
        # Common parameter validations
        if 'max_positions' in params and params['max_positions'] <= 0:
            errors.append("max_positions must be positive")
        
        if 'position_size_pct' in params and not (0 < params['position_size_pct'] <= 1):
            errors.append("position_size_pct must be between 0 and 1")
        
        # Strategy-specific validations
        if strategy_type == 'statistical_arbitrage':
            if 'correlation_threshold' in params and not (0 <= params['correlation_threshold'] <= 1):
                errors.append("correlation_threshold must be between 0 and 1")
            
            if 'z_score_entry' in params and params['z_score_entry'] <= 0:
                errors.append("z_score_entry must be positive")
        
        elif strategy_type == 'momentum':
            if 'volume_threshold_multiplier' in params and params['volume_threshold_multiplier'] <= 0:
                errors.append("volume_threshold_multiplier must be positive")
            
            if 'momentum_timeframes' in params:
                timeframes = params['momentum_timeframes']
                if not isinstance(timeframes, list) or not all(isinstance(x, int) and x > 0 for x in timeframes):
                    errors.append("momentum_timeframes must be a list of positive integers")
        
        elif strategy_type == 'mean_reversion':
            if 'bb_window' in params and params['bb_window'] <= 0:
                errors.append("bb_window must be positive")
            
            if 'bb_std_dev' in params and params['bb_std_dev'] <= 0:
                errors.append("bb_std_dev must be positive")
            
            weights = ['bb_weight', 'rsi_weight', 'tsi_weight']
            if all(w in params for w in weights):
                total_weight = sum(params[w] for w in weights)
                if abs(total_weight - 1.0) > 0.1:
                    errors.append("indicator weights should sum to approximately 1.0")
        
        elif strategy_type == 'multi_strategy':
            weights = ['stat_arb_weight', 'momentum_weight', 'mean_rev_weight']
            if all(w in params for w in weights):
                total_weight = sum(params[w] for w in weights)
                if total_weight <= 0:
                    errors.append("strategy weights must be positive")
        
        return errors