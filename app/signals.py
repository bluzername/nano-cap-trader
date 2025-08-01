"""Signal calculations."""
import pandas as pd
import httpx
from .utils import zscore
from .config import get_settings
from .alt_data import get_ortex_alternative_data

_settings = get_settings()

# Insider buy intensity ------------------------------------------------------

def insider_buy_score(df_form4: pd.DataFrame) -> pd.Series:
    buys = df_form4[df_form4.transactionType == "P"]  # open‑market purchase
    grouped = buys.groupby("ticker").netTransactionValue.sum()
    return zscore(grouped)

# Gap reversion --------------------------------------------------------------

def gap_reversion_score(prices: pd.DataFrame) -> pd.Series:
    g = prices.groupby("ticker").apply(lambda d: d.iloc[-1].close / d.iloc[-1].open - 1)
    return -zscore(g)

# Alt‑growth --------------------------------------------------------------

def alt_growth_score(card: pd.Series, web: pd.Series) -> pd.Series:
    s = zscore(card) + zscore(web)
    return zscore(s)

# Short squeeze --------------------------------------------------------------

def squeeze_score(short_util: pd.Series, d2c: pd.Series) -> pd.Series:
    return zscore(short_util) * zscore(d2c)

# Ortex alternative squeeze score using free data sources
async def squeeze_score_free_sources(symbols: list[str]) -> pd.Series:
    """Calculate squeeze score using free data sources instead of Ortex.
    
    Args:
        symbols: List of stock symbols
        
    Returns:
        Series with squeeze scores indexed by symbol
    """
    async with httpx.AsyncClient(timeout=30) as session:
        # Get alternative short interest data
        alt_data = await get_ortex_alternative_data(session, symbols)
        
        if alt_data.empty:
            return pd.Series(index=symbols, dtype=float).fillna(0)
        
        # Calculate squeeze metrics using free data
        utilization = alt_data.get("utilization_proxy", pd.Series()).fillna(0)
        days_to_cover = alt_data.get("days_to_cover_consensus", pd.Series()).fillna(0)
        short_pressure = alt_data.get("short_pressure_score", pd.Series()).fillna(0)
        
        # Combine metrics for squeeze score
        # High utilization + high days-to-cover + recent short pressure = squeeze potential
        squeeze_raw = (
            zscore(utilization) * 0.4 +
            zscore(days_to_cover) * 0.4 +
            zscore(short_pressure) * 0.2
        )
        
        return squeeze_raw.fillna(0)

# Low‑float momentum ---------------------------------------------------------

def momo_score(prices_20d: pd.DataFrame, floats: pd.Series) -> pd.Series:
    ret20 = prices_20d.groupby("ticker").apply(lambda d: d.close.iloc[-1] / d.close.iloc[0] - 1)
    mask = floats < 30_000_000
    return zscore(ret20[mask])

# Fusion ---------------------------------------------------------------------

def composite_signal(**kwargs: pd.Series) -> pd.Series:
    w = {
        "insider": _settings.insider_weight,
        "gap": _settings.gaprev_weight,
        "alt": _settings.alt_growth_weight,
        "squeeze": _settings.short_weight,
        "momo": _settings.momo_weight,
    }
    aligned = pd.concat(kwargs, axis=1).fillna(0)
    score = (
        w["insider"] * aligned.iloc[:, 0]
        + w["gap"] * aligned.iloc[:, 1]
        + w["alt"] * aligned.iloc[:, 2]
        + w["squeeze"] * aligned.iloc[:, 3]
        + w["momo"] * aligned.iloc[:, 4]
    )
    return score

# Main signal generation function with Ortex toggle -------------------------
async def generate_all_signals(
    symbols: list[str],
    form4_data: pd.DataFrame,
    price_data: pd.DataFrame,
    card_data: pd.Series,
    web_data: pd.Series,
    floats: pd.Series,
    ortex_util: pd.Series = None,
    ortex_d2c: pd.Series = None,
) -> pd.Series:
    """Generate composite signal using either Ortex or free data sources.
    
    Args:
        symbols: List of stock symbols
        form4_data: Insider trading data
        price_data: Price data for gap reversion and momentum
        card_data: Card spending data for alt-growth
        web_data: Web traffic data for alt-growth  
        floats: Float shares data
        ortex_util: Ortex utilization data (optional)
        ortex_d2c: Ortex days-to-cover data (optional)
        
    Returns:
        Series with composite signals
    """
    # Calculate signals that don't depend on short interest data
    insider = insider_buy_score(form4_data)
    gap_rev = gap_reversion_score(price_data)
    alt_growth = alt_growth_score(card_data, web_data)
    momo = momo_score(price_data, floats)
    
    # Calculate squeeze signal based on configuration
    if _settings.use_ortex and ortex_util is not None and ortex_d2c is not None:
        # Use Ortex data if available and enabled
        squeeze = squeeze_score(ortex_util, ortex_d2c)
        print("Using Ortex data for short squeeze signals")
    else:
        # Use free data sources
        squeeze = await squeeze_score_free_sources(symbols)
        print("Using free data sources for short squeeze signals")
    
    # Combine all signals
    return composite_signal(
        insider=insider,
        gap=gap_rev,
        alt=alt_growth,
        squeeze=squeeze,
        momo=momo,
    )