"""Position-level risk management, including stop-loss, take-profit, and time-based exits."""
from __future__ import annotations
import datetime as dt
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging

from ..strategies.base_strategy import Position, Signal, SignalType
from ..config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


class PositionRiskManager:
    """
    Manages risk for individual positions, including stop-loss, take-profit,
    and time-based exits.
    """
    
    def __init__(self):
        self.enable_stop_loss = _settings.enable_stop_loss
        self.stop_loss_pct = 0.02  # 2% stop loss default
        self.transaction_cost_pct = 0.001  # 0.1% per trade
        
    def check_stop_loss(self, position: Position, current_price: float) -> Optional[Signal]:
        """
        Checks if a stop-loss condition is met for a given position.
        
        Args:
            position: The Position object to check.
            current_price: The current market price of the asset.
            
        Returns:
            A Signal object to close the position if stop-loss is hit, otherwise None.
        """
        if not self.enable_stop_loss or position.stop_loss is None:
            return None
        
        # For long positions, stop loss is below entry
        if position.quantity > 0 and current_price <= position.stop_loss:
            logger.warning(f"STOP LOSS HIT for LONG {position.symbol}: Current Price ${current_price:.2f} <= Stop Loss ${position.stop_loss:.2f}")
            return Signal(
                symbol=position.symbol,
                signal_type=SignalType.SELL,  # Sell to close long
                confidence=1.0,  # High confidence for risk management
                price=current_price,
                timestamp=dt.datetime.now(),
                metadata={"reason": "stop_loss", "original_strategy": position.strategy_id}
            )
        
        # For short positions, stop loss is above entry
        elif position.quantity < 0 and current_price >= position.stop_loss:
            logger.warning(f"STOP LOSS HIT for SHORT {position.symbol}: Current Price ${current_price:.2f} >= Stop Loss ${position.stop_loss:.2f}")
            return Signal(
                symbol=position.symbol,
                signal_type=SignalType.BUY,  # Buy to close short
                confidence=1.0,
                price=current_price,
                timestamp=dt.datetime.now(),
                metadata={"reason": "stop_loss", "original_strategy": position.strategy_id}
            )
            
        return None
    
    def check_take_profit(self, position: Position, current_price: float, take_profit_pct: float = 0.05) -> Optional[Signal]:
        """
        Checks if a take-profit condition is met.
        
        Args:
            position: The Position object to check.
            current_price: The current market price of the asset.
            take_profit_pct: Percentage gain at which to take profit.
            
        Returns:
            A Signal object to close the position if take-profit is hit, otherwise None.
        """
        if position.take_profit is None:
            # Calculate dynamic take profit if not set
            if position.quantity > 0:  # Long
                position.take_profit = position.entry_price * (1 + take_profit_pct)
            elif position.quantity < 0:  # Short
                position.take_profit = position.entry_price * (1 - take_profit_pct)
        
        if position.take_profit is not None:
            if position.quantity > 0 and current_price >= position.take_profit:
                logger.info(f"TAKE PROFIT HIT for LONG {position.symbol}: Current Price ${current_price:.2f} >= Take Profit ${position.take_profit:.2f}")
                return Signal(
                    symbol=position.symbol,
                    signal_type=SignalType.SELL,
                    confidence=0.9,
                    price=current_price,
                    timestamp=dt.datetime.now(),
                    metadata={"reason": "take_profit", "original_strategy": position.strategy_id}
                )
            elif position.quantity < 0 and current_price <= position.take_profit:
                logger.info(f"TAKE PROFIT HIT for SHORT {position.symbol}: Current Price ${current_price:.2f} <= Take Profit ${position.take_profit:.2f}")
                return Signal(
                    symbol=position.symbol,
                    signal_type=SignalType.BUY,
                    confidence=0.9,
                    price=current_price,
                    timestamp=dt.datetime.now(),
                    metadata={"reason": "take_profit", "original_strategy": position.strategy_id}
                )
        return None
    
    def check_time_based_exit(self, position: Position, max_hold_days: int = 5) -> Optional[Signal]:
        """
        Checks if a position should be exited due to time elapsed.
        
        Args:
            position: The Position object to check.
            max_hold_days: Maximum number of days to hold a position.
            
        Returns:
            A Signal object to close the position if time limit is exceeded, otherwise None.
        """
        time_in_position = (dt.datetime.now() - position.entry_time).days
        
        if time_in_position >= max_hold_days:
            logger.info(f"TIME-BASED EXIT for {position.symbol}: Held for {time_in_position} days (max {max_hold_days})")
            return Signal(
                symbol=position.symbol,
                signal_type=SignalType.SELL if position.quantity > 0 else SignalType.BUY,
                confidence=0.7,  # Moderate confidence for time-based exit
                price=position.current_price or position.entry_price,  # Use current price if available
                timestamp=dt.datetime.now(),
                metadata={"reason": "time_based_exit", "original_strategy": position.strategy_id}
            )
        return None
    
    def calculate_transaction_cost(self, quantity: int, price: float) -> float:
        """
        Calculates realistic transaction cost with minimum fee.
        
        Args:
            quantity: Number of shares.
            price: Price per share.
            
        Returns:
            Total transaction cost (higher of percentage or minimum).
        """
        from ..config import get_settings
        settings = get_settings()
        
        percentage_cost = abs(quantity) * price * settings.transaction_cost_pct  # 0.1%
        minimum_cost = settings.min_transaction_cost  # $20 minimum
        
        # Return the higher of percentage cost or minimum
        return max(percentage_cost, minimum_cost)