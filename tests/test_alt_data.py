import pytest
import pandas as pd
from unittest.mock import AsyncMock, patch
from app.alt_data import (
    calculate_days_to_cover,
    get_short_interest_data,
    analyze_daily_short_pressure,
    get_ortex_alternative_data,
)


def test_calculate_days_to_cover():
    """Test days to cover calculation."""
    # Normal case
    assert calculate_days_to_cover(1000000, 100000, 10.0) == 10.0
    
    # Edge cases
    assert calculate_days_to_cover(0, 100000, 10.0) == 0.0
    assert calculate_days_to_cover(1000000, 0, 10.0) == 1000000.0
    assert calculate_days_to_cover(1000000, 100000, 0) == 0.0


@pytest.mark.asyncio
async def test_get_short_interest_data():
    """Test short interest data aggregation."""
    # Mock HTTP session
    session = AsyncMock()
    
    # Mock successful responses
    with patch('app.alt_data.fetch_finnhub_short_interest') as mock_finnhub, \
         patch('app.alt_data.fetch_fmp_short_interest') as mock_fmp, \
         patch('app.alt_data.fetch_finra_short_interest') as mock_finra:
        
        mock_finnhub.return_value = {"shortInterest": 1000000, "daysToCover": 5.0}
        mock_fmp.return_value = {"shortInterest": 1100000, "daysToCover": 5.5}
        mock_finra.return_value = pd.DataFrame([{
            "shortInterest": 1050000,
            "shortInterestRatio": 0.05
        }])
        
        symbols = ["AAPL", "MSFT"]
        result = await get_short_interest_data(session, symbols)
        
        assert not result.empty
        assert len(result) == 2
        assert "shares_short_consensus" in result.columns
        assert "days_to_cover_consensus" in result.columns


@pytest.mark.asyncio 
async def test_analyze_daily_short_pressure():
    """Test daily short pressure analysis."""
    session = AsyncMock()
    
    # Mock FINRA short sale volume data
    with patch('app.alt_data.fetch_finra_short_sale_volume') as mock_finra:
        mock_finra.return_value = pd.DataFrame([
            {"symbol": "AAPL", "shortVolume": 40000, "totalVolume": 100000, "short_ratio": 0.4},
            {"symbol": "MSFT", "shortVolume": 60000, "totalVolume": 100000, "short_ratio": 0.6},
        ])
        
        result = await analyze_daily_short_pressure(session, "2024-01-01")
        
        assert not result.empty
        assert "short_pressure_score" in result.columns
        assert "high_short_pressure" in result.columns
        assert result.iloc[0]["high_short_pressure"] == True  # 60% > 40% threshold


@pytest.mark.asyncio
async def test_get_ortex_alternative_data():
    """Test Ortex alternative data aggregation."""
    session = AsyncMock()
    
    with patch('app.alt_data.get_short_interest_data') as mock_si, \
         patch('app.alt_data.analyze_daily_short_pressure') as mock_pressure:
        
        mock_si.return_value = pd.DataFrame([
            {"symbol": "AAPL", "shares_short_consensus": 1000000, "days_to_cover_consensus": 5.0}
        ])
        
        mock_pressure.return_value = pd.DataFrame([
            {"symbol": "AAPL", "short_ratio": 0.45, "short_pressure_score": 45.0, "high_short_pressure": True}
        ])
        
        result = await get_ortex_alternative_data(session, ["AAPL"])
        
        assert not result.empty
        assert "utilization_proxy" in result.columns
        assert result.iloc[0]["symbol"] == "AAPL"


def test_config_ortex_toggle(settings):
    """Test that Ortex toggle works in configuration."""
    from app.config import Settings
    
    # Test default (free sources)
    settings_free = Settings(polygon_api_key="test", use_ortex=False)
    assert settings_free.use_ortex == False
    
    # Test with Ortex enabled
    settings_ortex = Settings(polygon_api_key="test", use_ortex=True, ortex_token="test_token")
    assert settings_ortex.use_ortex == True