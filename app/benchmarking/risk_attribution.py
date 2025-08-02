"""Risk attribution and factor analysis for strategy performance decomposition."""
from __future__ import annotations
import asyncio
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import logging

from ..data_sources.correlation_data import CorrelationDataProvider
from .performance_metrics import PerformanceAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class FactorExposure:
    """Factor exposure analysis result."""
    factor_name: str
    exposure: float  # Beta to this factor
    r_squared: float  # Explanatory power
    p_value: float  # Statistical significance
    contribution_to_return: float  # Factor's contribution to total return
    contribution_to_risk: float  # Factor's contribution to total risk


@dataclass
class AttributionResult:
    """Complete attribution analysis result."""
    strategy_name: str
    analysis_period: Tuple[dt.datetime, dt.datetime]
    
    # Factor exposures
    factor_exposures: List[FactorExposure]
    
    # Performance decomposition
    total_return: float
    factor_return: float  # Return explained by factors
    idiosyncratic_return: float  # Unexplained return (alpha)
    
    # Risk decomposition
    total_risk: float
    factor_risk: float  # Risk from factor exposures
    idiosyncratic_risk: float  # Stock-specific risk
    
    # Sector analysis
    sector_exposures: Dict[str, float]
    sector_contributions: Dict[str, float]
    
    # Style factors
    style_analysis: Dict[str, float]
    
    # Model diagnostics
    model_r_squared: float
    tracking_error: float
    information_ratio: float


class RiskAttributionAnalyzer:
    """
    Advanced risk attribution and factor analysis engine.
    
    Performs multi-factor risk attribution including:
    - Fama-French factor analysis
    - Sector attribution
    - Style factor analysis (momentum, value, quality, etc.)
    - Principal component analysis
    - Performance attribution
    """
    
    def __init__(self):
        self.correlation_provider = CorrelationDataProvider()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Predefined factor portfolios (ETFs as proxies)
        self.factor_etfs = {
            # Market factors
            'market': 'VTI',  # Total stock market
            'small_cap': 'VB',  # Small cap value
            'large_cap': 'VV',  # Large cap value
            
            # Fama-French factors
            'value': 'VTV',  # Value
            'growth': 'VUG',  # Growth
            'momentum': 'MTUM',  # Momentum
            'quality': 'QUAL',  # Quality
            'low_vol': 'USMV',  # Low volatility
            'size': 'VB',  # Size (small cap)
            
            # Sector ETFs
            'technology': 'XLK',
            'healthcare': 'XLV',
            'financials': 'XLF',
            'energy': 'XLE',
            'industrials': 'XLI',
            'consumer_discretionary': 'XLY',
            'consumer_staples': 'XLP',
            'utilities': 'XLU',
            'materials': 'XLB',
            'real_estate': 'XLRE',
            'communication': 'XLC',
            
            # Style factors
            'profitability': 'MTUM',  # Proxy using momentum
            'investment': 'QUAL',  # Proxy using quality
            'leverage': 'XLF',  # Proxy using financials
        }
        
        # Factor categories for analysis
        self.factor_categories = {
            'market': ['market'],
            'size_value': ['small_cap', 'large_cap', 'value', 'growth'],
            'style': ['momentum', 'quality', 'low_vol'],
            'sector': ['technology', 'healthcare', 'financials', 'energy', 'industrials',
                      'consumer_discretionary', 'consumer_staples', 'utilities', 'materials',
                      'real_estate', 'communication']
        }
        
        # Cache for factor returns
        self.factor_returns_cache: Dict[str, pd.Series] = {}
        self.cache_duration = dt.timedelta(hours=6)
        self.cache_timestamps: Dict[str, dt.datetime] = {}
    
    async def perform_attribution_analysis(
        self,
        strategy_returns: pd.Series,
        strategy_name: str,
        factor_categories: List[str] = None
    ) -> AttributionResult:
        """Perform comprehensive risk attribution analysis."""
        try:
            if strategy_returns.empty:
                raise ValueError("Strategy returns cannot be empty")
            
            start_date = strategy_returns.index[0]
            end_date = strategy_returns.index[-1]
            
            # Default to all categories if not specified
            if factor_categories is None:
                factor_categories = list(self.factor_categories.keys())
            
            # Get factor returns
            factor_returns = await self._get_factor_returns(start_date, end_date, factor_categories)
            
            if factor_returns.empty:
                raise ValueError("Could not obtain factor returns data")
            
            # Align returns
            aligned_data = pd.concat([strategy_returns, factor_returns], axis=1, join='inner')
            aligned_data.columns = ['strategy'] + list(factor_returns.columns)
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) < 20:
                raise ValueError("Insufficient data for attribution analysis")
            
            # Perform factor regression
            factor_exposures = await self._calculate_factor_exposures(
                aligned_data['strategy'], 
                aligned_data.drop('strategy', axis=1)
            )
            
            # Calculate performance attribution
            performance_attribution = self._calculate_performance_attribution(
                aligned_data['strategy'],
                aligned_data.drop('strategy', axis=1),
                factor_exposures
            )
            
            # Calculate risk attribution
            risk_attribution = self._calculate_risk_attribution(
                aligned_data['strategy'],
                aligned_data.drop('strategy', axis=1),
                factor_exposures
            )
            
            # Sector analysis
            sector_analysis = await self._perform_sector_analysis(
                strategy_returns, start_date, end_date
            )
            
            # Style analysis
            style_analysis = await self._perform_style_analysis(
                strategy_returns, start_date, end_date
            )
            
            # Model diagnostics
            model_diagnostics = self._calculate_model_diagnostics(
                aligned_data['strategy'],
                aligned_data.drop('strategy', axis=1),
                factor_exposures
            )
            
            # Create result
            result = AttributionResult(
                strategy_name=strategy_name,
                analysis_period=(start_date, end_date),
                factor_exposures=factor_exposures,
                total_return=performance_attribution['total_return'],
                factor_return=performance_attribution['factor_return'],
                idiosyncratic_return=performance_attribution['idiosyncratic_return'],
                total_risk=risk_attribution['total_risk'],
                factor_risk=risk_attribution['factor_risk'],
                idiosyncratic_risk=risk_attribution['idiosyncratic_risk'],
                sector_exposures=sector_analysis['exposures'],
                sector_contributions=sector_analysis['contributions'],
                style_analysis=style_analysis,
                model_r_squared=model_diagnostics['r_squared'],
                tracking_error=model_diagnostics['tracking_error'],
                information_ratio=model_diagnostics['information_ratio']
            )
            
            logger.info(f"Completed attribution analysis for {strategy_name}")
            logger.info(f"Model R²: {model_diagnostics['r_squared']:.3f}, Alpha: {performance_attribution['idiosyncratic_return']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in attribution analysis for {strategy_name}: {e}")
            raise
    
    async def _get_factor_returns(
        self,
        start_date: dt.datetime,
        end_date: dt.datetime,
        factor_categories: List[str]
    ) -> pd.DataFrame:
        """Get factor returns for specified categories."""
        try:
            # Determine which factors to include
            factors_to_include = []
            for category in factor_categories:
                if category in self.factor_categories:
                    factors_to_include.extend(self.factor_categories[category])
            
            # Remove duplicates
            factors_to_include = list(set(factors_to_include))
            
            # Get ETF symbols for these factors
            etf_symbols = [self.factor_etfs[factor] for factor in factors_to_include if factor in self.factor_etfs]
            etf_symbols = list(set(etf_symbols))  # Remove duplicates
            
            if not etf_symbols:
                return pd.DataFrame()
            
            # Fetch price data
            price_data = await self.correlation_provider._fetch_price_data(
                etf_symbols, (end_date - start_date).days + 30
            )
            
            if price_data.empty:
                return pd.DataFrame()
            
            # Calculate returns
            factor_returns = price_data.pct_change().dropna()
            
            # Map ETF returns to factor names
            factor_mapping = {}
            for factor, etf in self.factor_etfs.items():
                if etf in factor_returns.columns and factor in factors_to_include:
                    factor_mapping[etf] = factor
            
            # Rename columns
            renamed_returns = pd.DataFrame()
            for etf, factor in factor_mapping.items():
                if etf in factor_returns.columns:
                    renamed_returns[factor] = factor_returns[etf]
            
            # Filter by date range
            mask = (renamed_returns.index >= start_date) & (renamed_returns.index <= end_date)
            return renamed_returns[mask]
            
        except Exception as e:
            logger.error(f"Error getting factor returns: {e}")
            return pd.DataFrame()
    
    async def _calculate_factor_exposures(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> List[FactorExposure]:
        """Calculate exposures to each factor using regression."""
        try:
            exposures = []
            
            for factor_name in factor_returns.columns:
                factor_series = factor_returns[factor_name].dropna()
                
                # Align data
                aligned_strategy, aligned_factor = strategy_returns.align(factor_series, join='inner')
                
                if len(aligned_strategy) < 10:
                    continue
                
                # Single factor regression
                X = aligned_factor.values.reshape(-1, 1)
                y = aligned_strategy.values
                
                model = LinearRegression()
                model.fit(X, y)
                
                beta = model.coef_[0]
                r_squared = model.score(X, y)
                
                # Statistical significance
                residuals = y - model.predict(X)
                mse = np.mean(residuals ** 2)
                var_beta = mse / np.sum((aligned_factor - aligned_factor.mean()) ** 2)
                t_stat = beta / np.sqrt(var_beta) if var_beta > 0 else 0
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(aligned_strategy) - 2))
                
                # Contribution calculations
                factor_contribution = beta * aligned_factor.mean()
                risk_contribution = abs(beta) * aligned_factor.std()
                
                exposure = FactorExposure(
                    factor_name=factor_name,
                    exposure=beta,
                    r_squared=r_squared,
                    p_value=p_value,
                    contribution_to_return=factor_contribution,
                    contribution_to_risk=risk_contribution
                )
                
                exposures.append(exposure)
            
            # Sort by contribution to return
            exposures.sort(key=lambda x: abs(x.contribution_to_return), reverse=True)
            
            return exposures
            
        except Exception as e:
            logger.error(f"Error calculating factor exposures: {e}")
            return []
    
    def _calculate_performance_attribution(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
        factor_exposures: List[FactorExposure]
    ) -> Dict[str, float]:
        """Calculate performance attribution."""
        try:
            # Multi-factor regression
            aligned_data = pd.concat([strategy_returns, factor_returns], axis=1, join='inner').dropna()
            
            if len(aligned_data) < 10:
                return {'total_return': 0, 'factor_return': 0, 'idiosyncratic_return': 0}
            
            y = aligned_data.iloc[:, 0].values  # Strategy returns
            X = aligned_data.iloc[:, 1:].values  # Factor returns
            
            # Regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate attribution
            total_return = strategy_returns.mean() * 252  # Annualized
            
            # Factor contribution
            factor_return = 0
            for i, exposure in enumerate(factor_exposures):
                if i < len(model.coef_):
                    factor_mean_return = factor_returns[exposure.factor_name].mean()
                    factor_return += model.coef_[i] * factor_mean_return
            
            factor_return *= 252  # Annualized
            
            # Idiosyncratic return (alpha)
            idiosyncratic_return = total_return - factor_return
            
            return {
                'total_return': total_return,
                'factor_return': factor_return,
                'idiosyncratic_return': idiosyncratic_return
            }
            
        except Exception as e:
            logger.error(f"Error in performance attribution: {e}")
            return {'total_return': 0, 'factor_return': 0, 'idiosyncratic_return': 0}
    
    def _calculate_risk_attribution(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
        factor_exposures: List[FactorExposure]
    ) -> Dict[str, float]:
        """Calculate risk attribution."""
        try:
            # Calculate total risk (volatility)
            total_risk = strategy_returns.std() * np.sqrt(252)
            
            # Multi-factor regression for residuals
            aligned_data = pd.concat([strategy_returns, factor_returns], axis=1, join='inner').dropna()
            
            if len(aligned_data) < 10:
                return {'total_risk': total_risk, 'factor_risk': 0, 'idiosyncratic_risk': total_risk}
            
            y = aligned_data.iloc[:, 0].values
            X = aligned_data.iloc[:, 1:].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Predicted returns (factor component)
            factor_component = model.predict(X)
            factor_risk = np.std(factor_component) * np.sqrt(252)
            
            # Residuals (idiosyncratic component)
            residuals = y - factor_component
            idiosyncratic_risk = np.std(residuals) * np.sqrt(252)
            
            return {
                'total_risk': total_risk,
                'factor_risk': factor_risk,
                'idiosyncratic_risk': idiosyncratic_risk
            }
            
        except Exception as e:
            logger.error(f"Error in risk attribution: {e}")
            return {'total_risk': 0, 'factor_risk': 0, 'idiosyncratic_risk': 0}
    
    async def _perform_sector_analysis(
        self,
        strategy_returns: pd.Series,
        start_date: dt.datetime,
        end_date: dt.datetime
    ) -> Dict[str, Dict[str, float]]:
        """Perform sector-specific attribution analysis."""
        try:
            # Get sector ETF returns
            sector_etfs = {k: v for k, v in self.factor_etfs.items() if k in self.factor_categories['sector']}
            
            if not sector_etfs:
                return {'exposures': {}, 'contributions': {}}
            
            etf_symbols = list(sector_etfs.values())
            price_data = await self.correlation_provider._fetch_price_data(
                etf_symbols, (end_date - start_date).days + 30
            )
            
            if price_data.empty:
                return {'exposures': {}, 'contributions': {}}
            
            sector_returns = price_data.pct_change().dropna()
            
            # Filter by date range
            mask = (sector_returns.index >= start_date) & (sector_returns.index <= end_date)
            sector_returns = sector_returns[mask]
            
            # Calculate exposures to each sector
            exposures = {}
            contributions = {}
            
            for sector, etf in sector_etfs.items():
                if etf in sector_returns.columns:
                    aligned_strategy, aligned_sector = strategy_returns.align(
                        sector_returns[etf], join='inner'
                    )
                    
                    if len(aligned_strategy) >= 10:
                        # Regression
                        correlation = aligned_strategy.corr(aligned_sector)
                        beta = correlation * (aligned_strategy.std() / aligned_sector.std())
                        
                        exposures[sector] = beta
                        contributions[sector] = beta * aligned_sector.mean() * 252  # Annualized
            
            return {'exposures': exposures, 'contributions': contributions}
            
        except Exception as e:
            logger.error(f"Error in sector analysis: {e}")
            return {'exposures': {}, 'contributions': {}}
    
    async def _perform_style_analysis(
        self,
        strategy_returns: pd.Series,
        start_date: dt.datetime,
        end_date: dt.datetime
    ) -> Dict[str, float]:
        """Perform style factor analysis."""
        try:
            # Get style factor returns
            style_factors = self.factor_categories['style'] + self.factor_categories['size_value']
            style_etfs = {k: v for k, v in self.factor_etfs.items() if k in style_factors}
            
            if not style_etfs:
                return {}
            
            etf_symbols = list(style_etfs.values())
            price_data = await self.correlation_provider._fetch_price_data(
                etf_symbols, (end_date - start_date).days + 30
            )
            
            if price_data.empty:
                return {}
            
            style_returns = price_data.pct_change().dropna()
            
            # Filter by date range
            mask = (style_returns.index >= start_date) & (style_returns.index <= end_date)
            style_returns = style_returns[mask]
            
            # Align with strategy returns
            aligned_data = pd.concat([strategy_returns, style_returns], axis=1, join='inner').dropna()
            
            if len(aligned_data) < 10:
                return {}
            
            # Style analysis using constrained regression (weights sum to 1)
            y = aligned_data.iloc[:, 0].values
            X = aligned_data.iloc[:, 1:].values
            
            # Simple approach: calculate correlations and normalize
            correlations = {}
            for i, factor in enumerate(style_factors):
                if factor in style_etfs:
                    etf = style_etfs[factor]
                    if etf in aligned_data.columns:
                        corr = aligned_data.iloc[:, 0].corr(aligned_data[etf])
                        correlations[factor] = max(0, corr)  # Only positive exposures
            
            # Normalize to sum to 1
            total = sum(correlations.values())
            if total > 0:
                correlations = {k: v/total for k, v in correlations.items()}
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error in style analysis: {e}")
            return {}
    
    def _calculate_model_diagnostics(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
        factor_exposures: List[FactorExposure]
    ) -> Dict[str, float]:
        """Calculate model diagnostics and goodness of fit."""
        try:
            # Multi-factor regression
            aligned_data = pd.concat([strategy_returns, factor_returns], axis=1, join='inner').dropna()
            
            if len(aligned_data) < 10:
                return {'r_squared': 0, 'tracking_error': 0, 'information_ratio': 0}
            
            y = aligned_data.iloc[:, 0].values
            X = aligned_data.iloc[:, 1:].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # R-squared
            r_squared = model.score(X, y)
            
            # Tracking error (residual volatility)
            residuals = y - model.predict(X)
            tracking_error = np.std(residuals) * np.sqrt(252)
            
            # Information ratio (alpha / tracking error)
            alpha = model.intercept_ * 252  # Annualized alpha
            information_ratio = alpha / tracking_error if tracking_error > 0 else 0
            
            return {
                'r_squared': r_squared,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating model diagnostics: {e}")
            return {'r_squared': 0, 'tracking_error': 0, 'information_ratio': 0}
    
    async def compare_strategies_attribution(
        self,
        strategy_returns: Dict[str, pd.Series],
        factor_categories: List[str] = None
    ) -> Dict[str, AttributionResult]:
        """Compare attribution analysis across multiple strategies."""
        try:
            results = {}
            
            for strategy_name, returns in strategy_returns.items():
                try:
                    result = await self.perform_attribution_analysis(
                        returns, strategy_name, factor_categories
                    )
                    results[strategy_name] = result
                except Exception as e:
                    logger.error(f"Error analyzing {strategy_name}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comparative attribution analysis: {e}")
            return {}
    
    def generate_attribution_report(self, result: AttributionResult) -> str:
        """Generate a comprehensive attribution report."""
        try:
            report = []
            report.append(f"RISK ATTRIBUTION ANALYSIS: {result.strategy_name}")
            report.append("=" * 60)
            report.append(f"Analysis Period: {result.analysis_period[0].strftime('%Y-%m-%d')} to {result.analysis_period[1].strftime('%Y-%m-%d')}")
            report.append("")
            
            # Performance Summary
            report.append("PERFORMANCE ATTRIBUTION:")
            report.append(f"  Total Return (Annualized): {result.total_return:.2%}")
            report.append(f"  Factor Return: {result.factor_return:.2%}")
            report.append(f"  Alpha (Idiosyncratic): {result.idiosyncratic_return:.2%}")
            report.append("")
            
            # Risk Summary
            report.append("RISK ATTRIBUTION:")
            report.append(f"  Total Risk (Volatility): {result.total_risk:.2%}")
            report.append(f"  Factor Risk: {result.factor_risk:.2%}")
            report.append(f"  Idiosyncratic Risk: {result.idiosyncratic_risk:.2%}")
            report.append("")
            
            # Factor Exposures
            report.append("FACTOR EXPOSURES:")
            for exposure in result.factor_exposures[:10]:  # Top 10
                significance = "***" if exposure.p_value < 0.01 else "**" if exposure.p_value < 0.05 else "*" if exposure.p_value < 0.10 else ""
                report.append(f"  {exposure.factor_name:20s}: β={exposure.exposure:6.3f} {significance} (R²={exposure.r_squared:.3f})")
            report.append("")
            
            # Sector Analysis
            if result.sector_exposures:
                report.append("SECTOR EXPOSURES:")
                sorted_sectors = sorted(result.sector_exposures.items(), key=lambda x: abs(x[1]), reverse=True)
                for sector, exposure in sorted_sectors[:5]:  # Top 5
                    contribution = result.sector_contributions.get(sector, 0)
                    report.append(f"  {sector:20s}: β={exposure:6.3f} (Contrib: {contribution:.2%})")
                report.append("")
            
            # Style Analysis
            if result.style_analysis:
                report.append("STYLE ANALYSIS:")
                sorted_styles = sorted(result.style_analysis.items(), key=lambda x: x[1], reverse=True)
                for style, weight in sorted_styles:
                    report.append(f"  {style:20s}: {weight:.1%}")
                report.append("")
            
            # Model Diagnostics
            report.append("MODEL DIAGNOSTICS:")
            report.append(f"  R-Squared: {result.model_r_squared:.3f}")
            report.append(f"  Tracking Error: {result.tracking_error:.2%}")
            report.append(f"  Information Ratio: {result.information_ratio:.3f}")
            report.append("")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating attribution report: {e}")
            return f"Error generating report: {e}"