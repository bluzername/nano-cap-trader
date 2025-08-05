"""
Machine Learning Insider Prediction Strategy

This strategy uses ML to predict which insider trades will generate the highest returns:
- Feature engineering from Form 4 filing patterns
- Historical insider track record analysis
- Market regime classification
- Ensemble model combining multiple ML algorithms
- Real-time model updates based on performance

Target Performance: 8-10% annual alpha
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import pickle
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score

from .base_strategy import BaseStrategy, Signal
from .strategy_types import StrategyType, SignalType
from ..utils.technical_indicators import (
    calculate_rsi, calculate_atr, calculate_bollinger_bands,
    calculate_macd, calculate_obv
)

logger = logging.getLogger(__name__)


@dataclass
class InsiderFeatures:
    """Comprehensive feature set for ML model"""
    # Insider characteristics
    insider_type_score: float  # CEO=1.0, CFO=0.8, Director=0.5, etc.
    historical_success_rate: float  # Past profitable trades %
    avg_holding_period: float  # Days
    total_historical_trades: int
    
    # Transaction features
    transaction_size_zscore: float  # Relative to insider's history
    transaction_value_pct_mcap: float  # % of market cap
    days_since_last_purchase: float
    cluster_score: float  # Multiple insiders buying
    
    # Market context
    stock_momentum_30d: float
    sector_momentum_30d: float
    market_regime: int  # 0=bear, 1=neutral, 2=bull
    volatility_regime: int  # 0=low, 1=normal, 2=high
    
    # Technical features
    rsi: float
    distance_from_52w_low: float
    volume_ratio: float
    price_vs_vwap: float
    
    # Fundamental proxies
    price_to_book_estimate: float
    earnings_momentum: float
    institutional_ownership_change: float


class InsiderMLPredictorStrategy(BaseStrategy):
    """
    ML-based strategy for predicting profitable insider trades
    
    Key innovations:
    1. Feature engineering from multiple data sources
    2. Online learning with performance feedback
    3. Ensemble of multiple models
    4. Market regime adaptation
    5. Explainable AI for trade reasoning
    """
    
    def __init__(self, universe: List[str], **kwargs):
        super().__init__(
            strategy_id="insider_ml_predictor",
            strategy_type=StrategyType.MOMENTUM,
            universe=universe,
            **kwargs
        )
        
        # ML parameters
        self.retrain_frequency = kwargs.get('retrain_frequency', 30)  # days
        self.min_training_samples = kwargs.get('min_training_samples', 1000)
        self.prediction_horizon = kwargs.get('prediction_horizon', 30)  # days
        self.min_prediction_confidence = kwargs.get('min_prediction_confidence', 0.65)
        self.use_ensemble = kwargs.get('use_ensemble', True)
        
        # Feature parameters
        self.lookback_window = kwargs.get('lookback_window', 90)
        self.min_insider_history = kwargs.get('min_insider_history', 3)
        
        # Risk parameters
        self.max_correlated_positions = kwargs.get('max_correlated_positions', 3)
        self.max_position_pct = kwargs.get('max_position_pct', 0.04)
        self.stop_loss_pct = kwargs.get('stop_loss_pct', 0.07)
        
        # Initialize models
        self.models = self._initialize_models()
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
        # Performance tracking
        self.prediction_history = []
        self.last_retrain = datetime.now()
        
        # Load pre-trained model if available
        self._load_models()
        
    def _initialize_models(self) -> Dict:
        """Initialize ML models"""
        models = {
            'rf_classifier': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            ),
            'gb_regressor': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        if self.use_ensemble:
            models['xgb_classifier'] = self._get_xgb_classifier()
            models['nn_model'] = self._get_neural_network()
        
        return models
    
    def generate_signals(self) -> List[Signal]:
        """Generate ML-based trading signals"""
        signals = []
        
        try:
            # Check if retraining needed
            if self._should_retrain():
                self._retrain_models()
            
            # Get data
            form4_data = self.kwargs.get('form4_data')
            market_data = self._get_market_data()
            additional_data = self._get_additional_data()
            
            if form4_data is None or form4_data.empty:
                logger.warning("No Form 4 data available")
                return signals
            
            # Get recent insider trades to evaluate
            recent_trades = self._get_recent_insider_trades(form4_data)
            
            # Generate features and predictions for each trade
            for _, trade in recent_trades.iterrows():
                try:
                    # Extract features
                    features = self._extract_features(
                        trade, form4_data, market_data, additional_data
                    )
                    
                    if features is None:
                        continue
                    
                    # Make prediction
                    prediction = self._predict_trade_outcome(features)
                    
                    if prediction['confidence'] >= self.min_prediction_confidence:
                        signal = self._create_ml_signal(
                            trade, prediction, features
                        )
                        signals.append(signal)
                        
                        # Store for future retraining
                        self._store_prediction(trade, prediction, features)
                        
                except Exception as e:
                    logger.error(f"Error processing trade for {trade.get('ticker', 'Unknown')}: {e}")
                    continue
            
            # Apply portfolio constraints and ranking
            signals = self._rank_and_filter_signals(signals)
            
        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")
            
        return signals
    
    def _get_recent_insider_trades(self, form4_data: pd.DataFrame) -> pd.DataFrame:
        """Get recent insider purchases to evaluate"""
        cutoff_date = datetime.now() - timedelta(days=5)  # Look at last 5 days
        
        recent_purchases = form4_data[
            (form4_data['transactionType'] == 'P') &
            (form4_data['filingDate'] >= cutoff_date) &
            (form4_data['ticker'].isin(self.universe))
        ]
        
        return recent_purchases
    
    def _extract_features(self, trade: pd.Series, form4_data: pd.DataFrame,
                         market_data: Dict, additional_data: Dict) -> Optional[InsiderFeatures]:
        """Extract ML features for a single insider trade"""
        
        try:
            symbol = trade['ticker']
            insider_name = trade['reportingOwner']
            
            # Get insider history
            insider_history = form4_data[
                (form4_data['reportingOwner'] == insider_name) &
                (form4_data['transactionDate'] < trade['transactionDate'])
            ]
            
            if len(insider_history) < self.min_insider_history:
                return None  # Not enough history
            
            # Calculate insider features
            insider_type_score = self._score_insider_type(trade)
            success_rate = self._calculate_historical_success_rate(insider_history)
            avg_holding = self._calculate_avg_holding_period(insider_history)
            
            # Transaction features
            transaction_zscore = self._calculate_transaction_zscore(trade, insider_history)
            mcap_pct = self._calculate_mcap_percentage(trade, market_data.get(symbol))
            days_since_last = self._days_since_last_purchase(trade, insider_history)
            cluster = self._calculate_cluster_score(symbol, trade['transactionDate'], form4_data)
            
            # Market context
            stock_momentum = self._calculate_momentum(symbol, market_data, 30)
            sector_momentum = self._calculate_sector_momentum(symbol, additional_data, 30)
            market_regime = self._classify_market_regime(additional_data)
            vol_regime = self._classify_volatility_regime(market_data)
            
            # Technical features
            tech_features = self._calculate_technical_features(symbol, market_data)
            
            # Fundamental proxies
            fundamental_features = self._calculate_fundamental_proxies(
                symbol, market_data, additional_data
            )
            
            return InsiderFeatures(
                insider_type_score=insider_type_score,
                historical_success_rate=success_rate,
                avg_holding_period=avg_holding,
                total_historical_trades=len(insider_history),
                transaction_size_zscore=transaction_zscore,
                transaction_value_pct_mcap=mcap_pct,
                days_since_last_purchase=days_since_last,
                cluster_score=cluster,
                stock_momentum_30d=stock_momentum,
                sector_momentum_30d=sector_momentum,
                market_regime=market_regime,
                volatility_regime=vol_regime,
                rsi=tech_features['rsi'],
                distance_from_52w_low=tech_features['distance_52w_low'],
                volume_ratio=tech_features['volume_ratio'],
                price_vs_vwap=tech_features['price_vs_vwap'],
                price_to_book_estimate=fundamental_features['pb_estimate'],
                earnings_momentum=fundamental_features['earnings_momentum'],
                institutional_ownership_change=fundamental_features['inst_change']
            )
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _score_insider_type(self, trade: pd.Series) -> float:
        """Score based on insider position"""
        title = trade.get('insiderTitle', '').lower()
        
        if any(term in title for term in ['ceo', 'chief executive']):
            return 1.0
        elif any(term in title for term in ['cfo', 'chief financial']):
            return 0.8
        elif any(term in title for term in ['president', 'coo']):
            return 0.7
        elif 'director' in title:
            return 0.5
        elif trade.get('is10PercentOwner', False):
            return 0.4
        else:
            return 0.2
    
    def _calculate_historical_success_rate(self, history: pd.DataFrame) -> float:
        """Calculate insider's historical trade success rate"""
        if history.empty:
            return 0.5  # No history, neutral
        
        # This would ideally look at actual returns
        # For now, use a proxy based on subsequent price action
        purchases = history[history['transactionType'] == 'P']
        if purchases.empty:
            return 0.5
        
        # Placeholder - would calculate actual success rate
        return 0.6
    
    def _calculate_avg_holding_period(self, history: pd.DataFrame) -> float:
        """Calculate average holding period in days"""
        # Placeholder - would analyze sale patterns
        return 365.0  # Default 1 year
    
    def _calculate_transaction_zscore(self, trade: pd.Series, history: pd.DataFrame) -> float:
        """Calculate z-score of transaction size relative to insider's history"""
        if history.empty:
            return 0.0
        
        historical_values = history['netTransactionValue']
        if len(historical_values) < 2:
            return 0.0
        
        mean_value = historical_values.mean()
        std_value = historical_values.std()
        
        if std_value == 0:
            return 0.0
        
        return (trade['netTransactionValue'] - mean_value) / std_value
    
    def _calculate_mcap_percentage(self, trade: pd.Series, market_data: Optional[pd.DataFrame]) -> float:
        """Calculate transaction value as % of market cap"""
        if market_data is None or market_data.empty:
            return 0.0
        
        # Estimate market cap from price and assumed shares
        latest_price = market_data['close'].iloc[-1]
        estimated_mcap = latest_price * 10_000_000  # Placeholder
        
        return (trade['netTransactionValue'] / estimated_mcap) * 100
    
    def _days_since_last_purchase(self, trade: pd.Series, history: pd.DataFrame) -> float:
        """Calculate days since insider's last purchase"""
        purchases = history[history['transactionType'] == 'P']
        if purchases.empty:
            return 365.0  # Default if no prior purchases
        
        last_purchase = purchases['transactionDate'].max()
        days_diff = (trade['transactionDate'] - last_purchase).days
        
        return float(days_diff)
    
    def _calculate_cluster_score(self, symbol: str, trade_date: pd.Timestamp,
                               form4_data: pd.DataFrame) -> float:
        """Calculate cluster score for multiple insiders buying"""
        window_start = trade_date - timedelta(days=10)
        window_end = trade_date + timedelta(days=10)
        
        cluster_trades = form4_data[
            (form4_data['ticker'] == symbol) &
            (form4_data['transactionType'] == 'P') &
            (form4_data['transactionDate'] >= window_start) &
            (form4_data['transactionDate'] <= window_end)
        ]
        
        unique_insiders = cluster_trades['reportingOwner'].nunique()
        total_value = cluster_trades['netTransactionValue'].sum()
        
        # Score based on number of insiders and total value
        score = np.log1p(unique_insiders) * np.log1p(total_value / 1_000_000)
        
        return min(score / 5, 1.0)  # Normalize to [0, 1]
    
    def _calculate_momentum(self, symbol: str, market_data: Dict, days: int) -> float:
        """Calculate price momentum"""
        if symbol not in market_data or len(market_data[symbol]) < days:
            return 0.0
        
        data = market_data[symbol]
        current_price = data['close'].iloc[-1]
        past_price = data['close'].iloc[-days]
        
        return (current_price - past_price) / past_price
    
    def _calculate_sector_momentum(self, symbol: str, additional_data: Dict, days: int) -> float:
        """Calculate sector momentum"""
        # Placeholder - would use sector ETF or peer group
        return 0.0
    
    def _classify_market_regime(self, additional_data: Dict) -> int:
        """Classify current market regime"""
        # Placeholder - would use SPY/VIX analysis
        # 0 = bear, 1 = neutral, 2 = bull
        return 1
    
    def _classify_volatility_regime(self, market_data: Dict) -> int:
        """Classify volatility regime"""
        # Placeholder - would use VIX or realized vol
        # 0 = low, 1 = normal, 2 = high
        return 1
    
    def _calculate_technical_features(self, symbol: str, market_data: Dict) -> Dict:
        """Calculate technical indicators"""
        features = {
            'rsi': 50.0,
            'distance_52w_low': 0.5,
            'volume_ratio': 1.0,
            'price_vs_vwap': 1.0
        }
        
        if symbol not in market_data:
            return features
        
        data = market_data[symbol]
        if len(data) < 50:
            return features
        
        # RSI
        rsi = calculate_rsi(data['close'])
        features['rsi'] = rsi[-1]
        
        # Distance from 52-week low
        low_52w = data['low'][-252:].min() if len(data) >= 252 else data['low'].min()
        current = data['close'].iloc[-1]
        features['distance_52w_low'] = (current - low_52w) / low_52w
        
        # Volume ratio
        recent_vol = data['volume'][-5:].mean()
        avg_vol = data['volume'][-20:].mean()
        features['volume_ratio'] = recent_vol / avg_vol if avg_vol > 0 else 1.0
        
        return features
    
    def _calculate_fundamental_proxies(self, symbol: str, market_data: Dict,
                                     additional_data: Dict) -> Dict:
        """Calculate fundamental feature proxies"""
        # Placeholder - would use actual fundamental data
        return {
            'pb_estimate': 1.5,
            'earnings_momentum': 0.0,
            'inst_change': 0.0
        }
    
    def _predict_trade_outcome(self, features: InsiderFeatures) -> Dict:
        """Make prediction using ensemble of models"""
        
        # Convert features to array
        feature_array = np.array([
            features.insider_type_score,
            features.historical_success_rate,
            features.avg_holding_period,
            np.log1p(features.total_historical_trades),
            features.transaction_size_zscore,
            features.transaction_value_pct_mcap,
            np.log1p(features.days_since_last_purchase),
            features.cluster_score,
            features.stock_momentum_30d,
            features.sector_momentum_30d,
            features.market_regime,
            features.volatility_regime,
            features.rsi,
            features.distance_from_52w_low,
            features.volume_ratio,
            features.price_vs_vwap,
            features.price_to_book_estimate,
            features.earnings_momentum,
            features.institutional_ownership_change
        ]).reshape(1, -1)
        
        # Scale features
        if hasattr(self.scaler, 'mean_'):  # Check if fitted
            feature_array_scaled = self.scaler.transform(feature_array)
        else:
            feature_array_scaled = feature_array
        
        predictions = {}
        
        # Get predictions from each model
        if 'rf_classifier' in self.models and hasattr(self.models['rf_classifier'], 'n_classes_'):
            rf_pred = self.models['rf_classifier'].predict_proba(feature_array_scaled)[0, 1]
            predictions['rf'] = rf_pred
        
        if 'gb_regressor' in self.models and hasattr(self.models['gb_regressor'], 'n_estimators'):
            gb_pred = self.models['gb_regressor'].predict(feature_array_scaled)[0]
            # Convert regression to probability
            gb_pred = 1 / (1 + np.exp(-gb_pred))  # Sigmoid
            predictions['gb'] = gb_pred
        
        # Ensemble prediction
        if predictions:
            ensemble_pred = np.mean(list(predictions.values()))
        else:
            # No trained models, use rule-based scoring
            ensemble_pred = self._rule_based_prediction(features)
        
        # Get feature importance for explainability
        top_features = self._get_top_features(feature_array[0])
        
        return {
            'confidence': ensemble_pred,
            'predictions': predictions,
            'top_features': top_features,
            'expected_return': self._estimate_return(ensemble_pred)
        }
    
    def _rule_based_prediction(self, features: InsiderFeatures) -> float:
        """Rule-based prediction when ML models not available"""
        score = 0.0
        
        # Insider quality
        score += features.insider_type_score * 0.2
        
        # Transaction significance
        if features.transaction_size_zscore > 1.5:
            score += 0.15
        
        # Cluster buying
        score += features.cluster_score * 0.15
        
        # Market conditions
        if features.rsi < 40:
            score += 0.1
        
        if features.stock_momentum_30d < -0.1:
            score += 0.1  # Contrarian
        
        # Historical success
        if features.historical_success_rate > 0.6:
            score += 0.15
        
        return min(score, 1.0)
    
    def _get_top_features(self, feature_values: np.ndarray) -> List[Tuple[str, float]]:
        """Get most important features for this prediction"""
        feature_names = [
            'insider_type', 'success_rate', 'holding_period', 'trade_count',
            'size_zscore', 'mcap_pct', 'days_since', 'cluster',
            'stock_momentum', 'sector_momentum', 'market_regime', 'vol_regime',
            'rsi', 'dist_52w_low', 'volume_ratio', 'price_vwap',
            'pb_estimate', 'earnings_mom', 'inst_change'
        ]
        
        # Use feature importance if available
        if hasattr(self.models.get('rf_classifier'), 'feature_importances_'):
            importances = self.models['rf_classifier'].feature_importances_
            top_indices = np.argsort(importances)[-5:][::-1]
            
            return [(feature_names[i], feature_values[i]) for i in top_indices]
        
        # Otherwise return features with extreme values
        z_scores = np.abs(feature_values - np.mean(feature_values)) / (np.std(feature_values) + 1e-6)
        top_indices = np.argsort(z_scores)[-5:][::-1]
        
        return [(feature_names[i], feature_values[i]) for i in top_indices]
    
    def _estimate_return(self, confidence: float) -> float:
        """Estimate expected return based on confidence"""
        # Historical calibration would be done here
        base_return = 0.15  # 15% base for high confidence
        return base_return * confidence
    
    def _create_ml_signal(self, trade: pd.Series, prediction: Dict,
                         features: InsiderFeatures) -> Signal:
        """Create signal with ML metadata"""
        
        # Determine signal strength
        if prediction['confidence'] > 0.85:
            strength = 'VERY_STRONG'
        elif prediction['confidence'] > 0.75:
            strength = 'STRONG'
        else:
            strength = 'MODERATE'
        
        metadata = {
            'strategy': 'insider_ml_predictor',
            'ml_confidence': round(prediction['confidence'], 3),
            'expected_return': round(prediction['expected_return'], 3),
            'insider_name': trade['reportingOwner'],
            'insider_title': trade.get('insiderTitle', 'Unknown'),
            'transaction_value': trade['netTransactionValue'],
            'cluster_score': round(features.cluster_score, 2),
            'insider_success_rate': round(features.historical_success_rate, 2),
            'top_factors': prediction['top_features'][:3],
            'signal_strength': strength,
            'models_used': list(prediction['predictions'].keys())
        }
        
        return Signal(
            symbol=trade['ticker'],
            signal_type=SignalType.BUY,
            confidence=prediction['confidence'],
            metadata=metadata,
            strategy_id=self.strategy_id,
            timestamp=datetime.now()
        )
    
    def _rank_and_filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """Rank signals and apply portfolio constraints"""
        if not signals:
            return signals
        
        # Sort by ML confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit positions
        max_positions = int(1.0 / self.max_position_pct)
        
        # Apply correlation filter
        selected = []
        for signal in signals:
            if self._check_correlation_constraint(signal, selected):
                selected.append(signal)
                if len(selected) >= max_positions:
                    break
        
        return selected
    
    def _check_correlation_constraint(self, signal: Signal, selected: List[Signal]) -> bool:
        """Check if signal violates correlation constraints"""
        # Placeholder - would check actual correlations
        return True
    
    def _should_retrain(self) -> bool:
        """Check if models should be retrained"""
        days_since_retrain = (datetime.now() - self.last_retrain).days
        return days_since_retrain >= self.retrain_frequency
    
    def _retrain_models(self):
        """Retrain ML models with recent data"""
        logger.info("Retraining ML models...")
        
        try:
            # Get training data
            training_data = self._prepare_training_data()
            
            if len(training_data) < self.min_training_samples:
                logger.warning(f"Insufficient training data: {len(training_data)} samples")
                return
            
            # Split features and labels
            X = training_data.drop(['symbol', 'return', 'profitable'], axis=1)
            y_class = training_data['profitable'].astype(int)
            y_reg = training_data['return']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            self.models['rf_classifier'].fit(X_scaled, y_class)
            self.models['gb_regressor'].fit(X_scaled, y_reg)
            
            # Store feature importance
            self.feature_importance = dict(zip(
                X.columns,
                self.models['rf_classifier'].feature_importances_
            ))
            
            # Update last retrain time
            self.last_retrain = datetime.now()
            
            # Save models
            self._save_models()
            
            logger.info("Model retraining completed successfully")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    def _prepare_training_data(self) -> pd.DataFrame:
        """Prepare historical data for training"""
        # Placeholder - would load historical trades and outcomes
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def _store_prediction(self, trade: pd.Series, prediction: Dict, features: InsiderFeatures):
        """Store prediction for future training"""
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'symbol': trade['ticker'],
            'prediction': prediction['confidence'],
            'features': features,
            'trade': trade.to_dict()
        })
        
        # Limit history size
        if len(self.prediction_history) > 10000:
            self.prediction_history = self.prediction_history[-10000:]
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'last_retrain': self.last_retrain
            }
            
            # Save to file
            # with open('insider_ml_models.pkl', 'wb') as f:
            #     pickle.dump(model_data, f)
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            # Load from file if exists
            # with open('insider_ml_models.pkl', 'rb') as f:
            #     model_data = pickle.load(f)
            #     self.models = model_data['models']
            #     self.scaler = model_data['scaler']
            #     self.feature_importance = model_data['feature_importance']
            #     self.last_retrain = model_data['last_retrain']
            pass
            
        except Exception as e:
            logger.info("No pre-trained models found, will train on first run")
    
    def _get_xgb_classifier(self):
        """Get XGBoost classifier if available"""
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        except ImportError:
            logger.info("XGBoost not available, skipping")
            return None
    
    def _get_neural_network(self):
        """Get neural network model if available"""
        # Placeholder - would use TensorFlow/PyTorch
        return None
    
    def _get_market_data(self) -> Dict:
        """Get market data for analysis"""
        # Placeholder
        return {}
    
    def _get_additional_data(self) -> Dict:
        """Get additional data sources"""
        # Placeholder
        return {}
    
    def get_model_performance(self) -> Dict:
        """Get current model performance metrics"""
        return {
            'last_retrain': self.last_retrain.isoformat(),
            'predictions_made': len(self.prediction_history),
            'feature_importance': self.feature_importance,
            'model_types': list(self.models.keys())
        }