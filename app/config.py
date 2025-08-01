from pydantic import BaseSettings, Field
from functools import lru_cache

class Settings(BaseSettings):
    polygon_api_key: str = Field(..., env="POLYGON_API_KEY")
    ortex_token: str | None = Field(None, env="ORTEX_KEY")
    db_url: str = Field("sqlite:///data.db", env="DB_URL")
    max_position_value: float = 15000.0  # per‑name $ cap
    max_portfolio_value: float = 1000000.0  # $1 M AUM
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

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()