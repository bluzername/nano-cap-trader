"""
Enhanced insider trading data integration

Integrates multiple insider data sources beyond basic Form 4:
- SEC EDGAR direct access for real-time filings
- Options flow data integration
- Institutional ownership changes (13F)
- Short interest data
"""

import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import logging
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

from ..config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


class EnhancedInsiderDataProvider:
    """Enhanced provider for insider trading and related data"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # API endpoints
        self.sec_edgar_base = "https://www.sec.gov/Archives/edgar"
        self.fintel_base = "https://api.fintel.io/api/v1"
        self.whalewisdom_base = "https://whalewisdom.com/api/v1"
        
    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=30.0)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def get_enhanced_form4_data(self, symbols: List[str], 
                                    lookback_days: int = 90) -> pd.DataFrame:
        """
        Get enhanced Form 4 data with additional context
        
        Returns DataFrame with columns:
        - All standard Form 4 fields
        - insider_ranking: Historical success score
        - cluster_score: Multiple insider activity score
        - transaction_significance: Size relative to insider's history
        """
        try:
            # Get base Form 4 data from multiple sources
            form4_tasks = [
                self._get_polygon_form4(symbols, lookback_days),
                self._get_sec_edgar_form4(symbols, lookback_days)
            ]
            
            form4_results = await asyncio.gather(*form4_tasks, return_exceptions=True)
            
            # Combine results
            form4_data = pd.DataFrame()
            for result in form4_results:
                if isinstance(result, pd.DataFrame) and not result.empty:
                    form4_data = pd.concat([form4_data, result], ignore_index=True)
            
            if form4_data.empty:
                return pd.DataFrame()
            
            # Remove duplicates
            form4_data = form4_data.drop_duplicates(
                subset=['ticker', 'reportingOwner', 'transactionDate', 'shares'],
                keep='first'
            )
            
            # Enhance with additional metrics
            form4_data = await self._enhance_insider_data(form4_data)
            
            return form4_data
            
        except Exception as e:
            logger.error(f"Error getting enhanced Form 4 data: {e}")
            return pd.DataFrame()
    
    async def _get_polygon_form4(self, symbols: List[str], lookback_days: int) -> pd.DataFrame:
        """Get Form 4 data from Polygon.io"""
        if not _settings.polygon_api_key:
            return pd.DataFrame()
        
        try:
            since = datetime.now() - timedelta(days=lookback_days)
            
            params = {
                "apiKey": _settings.polygon_api_key,
                "type": "4",
                "limit": 1000,
                "sort": "timestamp",
                "order": "desc",
                "timestamp.gte": int(since.timestamp() * 1000),
            }
            
            response = await self.session.get(
                "https://api.polygon.io/vX/reference/sec_filings",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    df = pd.DataFrame(data['results'])
                    # Filter for our symbols
                    df = df[df['ticker'].isin(symbols)]
                    return self._standardize_form4_data(df, 'polygon')
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching Polygon Form 4 data: {e}")
            return pd.DataFrame()
    
    async def _get_sec_edgar_form4(self, symbols: List[str], lookback_days: int) -> pd.DataFrame:
        """Get Form 4 data directly from SEC EDGAR"""
        try:
            # SEC EDGAR requires specific headers
            headers = {
                'User-Agent': 'NanoCapTrader/1.0 (contact@example.com)'
            }
            
            all_filings = []
            
            for symbol in symbols:
                # Search for company CIK
                cik = await self._get_company_cik(symbol)
                if not cik:
                    continue
                
                # Get recent Form 4 filings
                url = f"{self.sec_edgar_base}/data/{cik}/index.json"
                
                response = await self.session.get(url, headers=headers)
                if response.status_code != 200:
                    continue
                
                filings_data = response.json()
                form4_filings = [
                    f for f in filings_data.get('directory', {}).get('item', [])
                    if '4' in f.get('name', '')
                ]
                
                # Parse each Form 4
                for filing in form4_filings[-20:]:  # Last 20 Form 4s
                    filing_data = await self._parse_form4_filing(cik, filing['name'])
                    if filing_data:
                        filing_data['ticker'] = symbol
                        all_filings.append(filing_data)
            
            if all_filings:
                df = pd.DataFrame(all_filings)
                return self._standardize_form4_data(df, 'edgar')
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching SEC EDGAR data: {e}")
            return pd.DataFrame()
    
    async def _get_company_cik(self, symbol: str) -> Optional[str]:
        """Get company CIK from ticker symbol"""
        try:
            # Use SEC's company tickers file
            url = "https://www.sec.gov/files/company_tickers.json"
            
            cache_key = "company_tickers"
            if cache_key in self.cache:
                tickers_data = self.cache[cache_key]
            else:
                response = await self.session.get(url)
                if response.status_code == 200:
                    tickers_data = response.json()
                    self.cache[cache_key] = tickers_data
                else:
                    return None
            
            # Find CIK for symbol
            for company in tickers_data.values():
                if company.get('ticker') == symbol:
                    return str(company.get('cik_str')).zfill(10)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting CIK for {symbol}: {e}")
            return None
    
    async def _parse_form4_filing(self, cik: str, filing_name: str) -> Optional[Dict]:
        """Parse individual Form 4 XML filing"""
        try:
            url = f"{self.sec_edgar_base}/data/{cik}/{filing_name}"
            
            response = await self.session.get(url)
            if response.status_code != 200:
                return None
            
            # Parse XML
            root = ET.fromstring(response.text)
            
            # Extract key information
            reporting_owner = root.find('.//reportingOwner/reportingOwnerId/rptOwnerName')
            transaction = root.find('.//nonDerivativeTransaction')
            
            if reporting_owner is None or transaction is None:
                return None
            
            # Build filing data
            filing_data = {
                'reportingOwner': reporting_owner.text,
                'transactionDate': transaction.find('.//transactionDate/value').text,
                'transactionType': transaction.find('.//transactionCoding/transactionCode').text,
                'shares': float(transaction.find('.//transactionShares/value').text or 0),
                'pricePerShare': float(transaction.find('.//transactionPricePerShare/value').text or 0),
                'filingDate': root.find('.//periodOfReport').text,
            }
            
            # Calculate net transaction value
            filing_data['netTransactionValue'] = filing_data['shares'] * filing_data['pricePerShare']
            
            # Get insider title
            title_elem = root.find('.//reportingOwner/reportingOwnerRelationship/officerTitle')
            if title_elem is not None:
                filing_data['insiderTitle'] = title_elem.text
            
            return filing_data
            
        except Exception as e:
            logger.error(f"Error parsing Form 4 filing: {e}")
            return None
    
    def _standardize_form4_data(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Standardize Form 4 data from different sources"""
        
        # Map column names to standard format
        column_mappings = {
            'polygon': {
                'tickers': 'ticker',
                'reporting_owner': 'reportingOwner',
                'transaction_date': 'transactionDate',
                'transaction_type': 'transactionType',
                'shares': 'shares',
                'price_per_share': 'pricePerShare',
                'filing_date': 'filingDate',
                'net_amount': 'netTransactionValue'
            },
            'edgar': {}  # Already in standard format
        }
        
        if source in column_mappings:
            df = df.rename(columns=column_mappings[source])
        
        # Ensure required columns exist
        required_columns = [
            'ticker', 'reportingOwner', 'transactionDate', 'transactionType',
            'shares', 'pricePerShare', 'filingDate', 'netTransactionValue'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # Convert dates
        date_columns = ['transactionDate', 'filingDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    async def _enhance_insider_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced metrics to insider data"""
        
        # Calculate insider ranking based on historical performance
        df['insider_ranking'] = df.groupby('reportingOwner').apply(
            self._calculate_insider_ranking
        ).reset_index(level=0, drop=True)
        
        # Calculate cluster scores
        df['cluster_score'] = df.groupby('ticker').apply(
            self._calculate_cluster_score
        ).reset_index(level=0, drop=True)
        
        # Calculate transaction significance
        df['transaction_significance'] = df.groupby('reportingOwner').apply(
            self._calculate_transaction_significance
        ).reset_index(level=0, drop=True)
        
        return df
    
    def _calculate_insider_ranking(self, group: pd.DataFrame) -> pd.Series:
        """Calculate insider's historical success ranking"""
        # Simplified ranking based on transaction patterns
        num_transactions = len(group)
        total_value = group['netTransactionValue'].sum()
        
        # Score based on consistency and size
        score = np.log1p(num_transactions) * np.log1p(total_value / 1_000_000)
        
        return pd.Series([score] * len(group), index=group.index)
    
    def _calculate_cluster_score(self, group: pd.DataFrame) -> pd.Series:
        """Calculate cluster score for coordinated buying"""
        scores = []
        
        for idx, row in group.iterrows():
            # Find transactions within 10 days
            window_start = row['transactionDate'] - timedelta(days=10)
            window_end = row['transactionDate'] + timedelta(days=10)
            
            window_transactions = group[
                (group['transactionDate'] >= window_start) &
                (group['transactionDate'] <= window_end) &
                (group['transactionType'] == 'P')  # Purchases only
            ]
            
            # Score based on number of unique insiders and total value
            unique_insiders = window_transactions['reportingOwner'].nunique()
            total_value = window_transactions['netTransactionValue'].sum()
            
            score = np.log1p(unique_insiders) * np.log1p(total_value / 1_000_000)
            scores.append(min(score / 5, 1.0))  # Normalize to [0, 1]
        
        return pd.Series(scores, index=group.index)
    
    def _calculate_transaction_significance(self, group: pd.DataFrame) -> pd.Series:
        """Calculate how significant this transaction is for the insider"""
        if len(group) < 2:
            return pd.Series([1.0] * len(group), index=group.index)
        
        # Calculate z-score of transaction values
        mean_value = group['netTransactionValue'].mean()
        std_value = group['netTransactionValue'].std()
        
        if std_value == 0:
            return pd.Series([1.0] * len(group), index=group.index)
        
        z_scores = (group['netTransactionValue'] - mean_value) / std_value
        
        # Convert to significance score (sigmoid)
        significance = 1 / (1 + np.exp(-z_scores / 2))
        
        return significance
    
    async def get_options_flow_data(self, symbols: List[str], 
                                  lookback_days: int = 5) -> pd.DataFrame:
        """
        Get unusual options activity data
        
        Returns DataFrame with:
        - ticker, timestamp, option_type (CALL/PUT)
        - strike, expiry, volume, open_interest
        - implied_volatility, premium
        - is_sweep, is_block, sentiment
        """
        
        # This would integrate with options data providers
        # For now, return empty DataFrame as placeholder
        logger.info("Options flow data integration not yet implemented")
        return pd.DataFrame()
    
    async def get_institutional_flow_data(self, symbols: List[str],
                                        lookback_days: int = 90) -> pd.DataFrame:
        """
        Get institutional ownership changes from 13F filings
        
        Returns DataFrame with:
        - ticker, institution_name, report_date
        - shares_held, shares_change, value_held
        """
        
        # This would integrate with 13F data providers
        # For now, return empty DataFrame as placeholder
        logger.info("Institutional flow data integration not yet implemented")
        return pd.DataFrame()
    
    async def get_short_interest_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get short interest data
        
        Returns DataFrame with:
        - ticker, short_interest, short_percent_float
        - days_to_cover, cost_to_borrow
        """
        
        # This would integrate with short interest data providers
        # For now, return empty DataFrame as placeholder
        logger.info("Short interest data integration not yet implemented")
        return pd.DataFrame()


async def get_enhanced_insider_signals(symbols: List[str], 
                                     lookback_days: int = 90) -> Dict[str, Any]:
    """
    Main function to get all enhanced insider-related signals
    
    Returns dictionary with:
    - form4_enhanced: Enhanced Form 4 data
    - options_flow: Unusual options activity
    - institutional_flow: 13F changes
    - short_interest: Short interest data
    - composite_scores: Combined signal strength per symbol
    """
    
    async with EnhancedInsiderDataProvider() as provider:
        # Gather all data sources in parallel
        tasks = [
            provider.get_enhanced_form4_data(symbols, lookback_days),
            provider.get_options_flow_data(symbols, lookback_days),
            provider.get_institutional_flow_data(symbols, lookback_days),
            provider.get_short_interest_data(symbols)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        form4_data = results[0] if isinstance(results[0], pd.DataFrame) else pd.DataFrame()
        options_data = results[1] if isinstance(results[1], pd.DataFrame) else pd.DataFrame()
        inst_data = results[2] if isinstance(results[2], pd.DataFrame) else pd.DataFrame()
        short_data = results[3] if isinstance(results[3], pd.DataFrame) else pd.DataFrame()
        
        # Calculate composite scores
        composite_scores = {}
        for symbol in symbols:
            score = 0.0
            factors = 0
            
            # Form 4 score
            if not form4_data.empty:
                symbol_form4 = form4_data[form4_data['ticker'] == symbol]
                if not symbol_form4.empty:
                    # Weight recent purchases
                    recent_purchases = symbol_form4[
                        (symbol_form4['transactionType'] == 'P') &
                        (symbol_form4['transactionDate'] >= datetime.now() - timedelta(days=30))
                    ]
                    if not recent_purchases.empty:
                        form4_score = (
                            recent_purchases['cluster_score'].mean() * 0.4 +
                            recent_purchases['transaction_significance'].mean() * 0.3 +
                            recent_purchases['insider_ranking'].mean() * 0.3
                        )
                        score += form4_score
                        factors += 1
            
            # Options flow score (when implemented)
            # Institutional flow score (when implemented)
            # Short interest score (when implemented)
            
            if factors > 0:
                composite_scores[symbol] = score / factors
            else:
                composite_scores[symbol] = 0.0
        
        return {
            'form4_enhanced': form4_data,
            'options_flow': options_data,
            'institutional_flow': inst_data,
            'short_interest': short_data,
            'composite_scores': composite_scores
        }