"""Application settings loaded from environment variables and an optional .env file.

Every field maps to an environment variable of the same name in upper case
(POLYGON_API_KEY, MAX_POSITION_VALUE, ...). Fields that were renamed keep their
old variable name as an accepted alias so existing .env files keep working.
"""
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    # Required
    polygon_api_key: str

    # Storage. DATABASE_URL is canonical; DB_URL is accepted for older .env files.
    database_url: str = Field(
        "sqlite:///data.db",
        validation_alias=AliasChoices("DATABASE_URL", "DB_URL", "database_url", "db_url"),
    )

    # Portfolio limits
    max_position_value: float = 8000.0  # per-name cap in dollars (16% of a $50k book)
    max_portfolio_value: float = 50000.0  # $50k AUM (realistic for nano-cap)
    max_daily_trades: int = 100

    # Composite signal weights
    insider_weight: float = 0.30
    gaprev_weight: float = 0.10
    alt_growth_weight: float = 0.25
    short_weight: float = 0.15
    momo_weight: float = 0.20

    # Short interest data source selection. use_ortex is the on/off switch,
    # ortex_token is the credential. ORTEX_KEY is accepted as an alias.
    use_ortex: bool = False
    ortex_token: str | None = Field(
        None,
        validation_alias=AliasChoices("ORTEX_TOKEN", "ORTEX_KEY", "ortex_token"),
    )
    finnhub_api_key: str | None = None
    fmp_api_key: str | None = None

    # Free data source URLs
    finra_short_sale_url: str = "https://api.finra.org/data/group/otcMarket/name/regShoDaily"
    finra_short_interest_url: str = "https://api.finra.org/data/group/otcMarket/name/shortInterest"
    finnhub_base_url: str = "https://finnhub.io/api/v1"
    fmp_base_url: str = "https://financialmodelingprep.com/api/v4"

    # News and fundamentals (optional). ALPHA_VANTAGE_KEY is accepted as an alias.
    newsapi_key: str | None = None
    alpha_vantage_api_key: str | None = Field(
        None,
        validation_alias=AliasChoices(
            "ALPHA_VANTAGE_API_KEY", "ALPHA_VANTAGE_KEY", "alpha_vantage_api_key"
        ),
    )

    # Strategy selection (comma-separated list of strategy_factory names)
    enabled_strategies: str = "multi_strategy"

    # Strategy-specific parameters
    momentum_volume_threshold: float = 3.0
    stat_arb_correlation_threshold: float = 0.8
    mean_rev_bb_std_dev: float = 2.0

    # Multi-strategy weights
    multi_stat_arb_weight: float = 0.60
    multi_momentum_weight: float = 0.25
    multi_mean_rev_weight: float = 0.15

    # Risk management toggles
    enable_position_sizing: bool = True
    enable_stop_loss: bool = True
    enable_short_selling: bool = False  # long-only for nano-caps
    max_volume_pct: float = 0.005  # 0.5% of daily volume max

    # Transaction cost parameters
    transaction_cost_pct: float = 0.001  # 0.1%
    min_transaction_cost: float = 20.0  # $20 minimum
    min_position_value: float = 4000.0  # $4k minimum for cost efficiency

    # Enhanced insider data sources (optional)
    fintel_api_key: str | None = None
    whalewisdom_api_key: str | None = None
    tradier_api_key: str | None = None
    benzinga_api_key: str | None = None

    # Remote access security
    enable_auth: bool = False
    auth_username: str = "admin"
    auth_password: str = "changeme123"
    allowed_ips: str = ""
    rate_limit: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
