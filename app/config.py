from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    polygon_api_key: str = Field(..., env="POLYGON_API_KEY")
    ortex_token: str | None = Field(None, env="ORTEX_KEY")
    db_url: str = Field("sqlite:///data.db", env="DB_URL")
    max_position_value: float = 8000.0  # per‑name $ cap (16% of $50k portfolio)
    max_portfolio_value: float = 50000.0  # $50k AUM (realistic for nano-cap)
    insider_weight: float = 0.30
    gaprev_weight: float = 0.10
    alt_growth_weight: float = 0.25
    short_weight: float = 0.15
    momo_weight: float = 0.20
    
    # Short interest data source selection
    use_ortex: bool = Field(False, env="USE_ORTEX")  # Default to free sources
    finnhub_api_key: str | None = Field(None, env="FINNHUB_API_KEY")
    fmp_api_key: str | None = Field(None, env="FMP_API_KEY")
    
    # Free data source URLs
    finra_short_sale_url: str = "https://api.finra.org/data/group/otcMarket/name/regShoDaily"
    finra_short_interest_url: str = "https://api.finra.org/data/group/otcMarket/name/shortInterest"
    finnhub_base_url: str = "https://finnhub.io/api/v1"
    fmp_base_url: str = "https://financialmodelingprep.com/api/v4"
    
    # News API keys (optional)
    newsapi_key: str | None = Field(None, env="NEWSAPI_KEY")
    alpha_vantage_key: str | None = Field(None, env="ALPHA_VANTAGE_KEY")
    
    # Strategy selection and configuration
    enabled_strategies: str = Field("multi_strategy", env="ENABLED_STRATEGIES")  # Comma-separated
    
    # Strategy-specific parameters
    momentum_volume_threshold: float = Field(3.0, env="MOMENTUM_VOLUME_THRESHOLD")
    stat_arb_correlation_threshold: float = Field(0.8, env="STAT_ARB_CORRELATION_THRESHOLD")
    mean_rev_bb_std_dev: float = Field(2.0, env="MEAN_REV_BB_STD_DEV")
    
    # Multi-strategy weights
    multi_stat_arb_weight: float = Field(0.60, env="MULTI_STAT_ARB_WEIGHT")
    multi_momentum_weight: float = Field(0.25, env="MULTI_MOMENTUM_WEIGHT")
    multi_mean_rev_weight: float = Field(0.15, env="MULTI_MEAN_REV_WEIGHT")
    
    # Risk management toggles
    enable_position_sizing: bool = Field(True, env="ENABLE_POSITION_SIZING")
    enable_stop_loss: bool = Field(True, env="ENABLE_STOP_LOSS")
    enable_short_selling: bool = Field(False, env="ENABLE_SHORT_SELLING")  # Long-only for nano-caps
    max_volume_pct: float = Field(0.005, env="MAX_VOLUME_PCT")  # 0.5% of daily volume max (realistic)
    
    # Transaction cost parameters (realistic broker fees)
    transaction_cost_pct: float = Field(0.001, env="TRANSACTION_COST_PCT")  # 0.1%
    min_transaction_cost: float = Field(20.0, env="MIN_TRANSACTION_COST")  # $20 minimum
    min_position_value: float = Field(4000.0, env="MIN_POSITION_VALUE")  # $4k minimum for cost efficiency
    
    # Enhanced data source APIs (optional)
    fintel_api_key: str | None = Field(None, env="FINTEL_API_KEY")
    whalewisdom_api_key: str | None = Field(None, env="WHALEWISDOM_API_KEY")
    tradier_api_key: str | None = Field(None, env="TRADIER_API_KEY")
    benzinga_api_key: str | None = Field(None, env="BENZINGA_API_KEY")
    
    # Remote access security settings
    enable_auth: bool = Field(False, env="ENABLE_AUTH")
    auth_username: str = Field("admin", env="AUTH_USERNAME")
    auth_password: str = Field("changeme123", env="AUTH_PASSWORD")
    allowed_ips: str = Field("", env="ALLOWED_IPS")
    rate_limit: int = Field(100, env="RATE_LIMIT")

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()