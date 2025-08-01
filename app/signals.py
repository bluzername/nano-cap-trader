"""Signal calculations."""
import pandas as pd
from .utils import zscore
from .config import get_settings
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