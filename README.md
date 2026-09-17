# NanoCap Trader

A FastAPI + Dash platform for researching and paper-trading US nano-cap
equities. It pulls market data from Polygon (plus optional free and paid
sources), scores stocks with a set of pluggable strategies, sizes long-only
positions under explicit liquidity and cost limits, and exposes everything
through a JSON API, a Dash dashboard and a backtesting/benchmarking toolkit.

Not financial advice. This is research software; see Limitations before
putting money behind anything it outputs.

## Quick start

Requires Python 3.11 or newer.

```bash
git clone https://github.com/bluzername/nano-cap-trader.git
cd nano-cap-trader
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                      # or: uv sync
cp env.template .env                                 # then set POLYGON_API_KEY
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000/> for the landing page, `/dash/` for the dashboard
and `/docs` for the interactive API reference.

## Configuration

Settings live in `app/config.py` (pydantic-settings) and are read from the
environment or a `.env` file. `env.template` lists every field with its
default. Only one variable is required:

| Variable | Required | Purpose |
| --- | --- | --- |
| `POLYGON_API_KEY` | yes | Market data, Form 4 insider filings, universe screening |
| `DATABASE_URL` | no | SQLAlchemy URL, default `sqlite:///data.db` (`DB_URL` also accepted) |
| `FINNHUB_API_KEY`, `FMP_API_KEY` | no | Extra short-interest sources next to free FINRA data |
| `USE_ORTEX`, `ORTEX_TOKEN` | no | Switch short-interest data to Ortex (`ORTEX_KEY` also accepted) |
| `NEWSAPI_KEY`, `ALPHA_VANTAGE_API_KEY` | no | News sentiment and fallback prices (`ALPHA_VANTAGE_KEY` also accepted) |
| `FINTEL_API_KEY`, `WHALEWISDOM_API_KEY`, `TRADIER_API_KEY`, `BENZINGA_API_KEY` | no | Paid insider, 13F and options-flow feeds for the insider strategies |
| `ENABLED_STRATEGIES` | no | Comma-separated strategy names, default `multi_strategy` |
| `MAX_PORTFOLIO_VALUE`, `MAX_POSITION_VALUE`, `MAX_VOLUME_PCT`, `TRANSACTION_COST_PCT`, ... | no | Portfolio, liquidity and cost limits (see `env.template`) |
| `ENABLE_AUTH`, `AUTH_USERNAME`, `AUTH_PASSWORD`, `ALLOWED_IPS`, `RATE_LIMIT` | no | Optional HTTP auth, IP allow-list and rate limit for remote access |

Without a real Polygon key the server still starts, but every data call
returns empty results.

## Running

```bash
# development
uvicorn main:app --reload

# production (same command the Dockerfile uses)
gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 1

# container
docker build -t nano-cap-trader .
docker run --env-file .env -p 8000:8000 nano-cap-trader
```

`main.py` builds the FastAPI app, mounts the API router under `/api`, mounts
the Dash dashboard at `/dash` (`app/dash_app_simple.py`) and enables the
security middleware when `ENABLE_AUTH` or `ALLOWED_IPS` is set.

### API endpoints

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/api/status` | Cash, positions and scheduler state |
| GET | `/api/portfolio` | Portfolio detail page |
| GET | `/api/signals` | Current composite signals |
| GET | `/api/signals/{symbol}` | Signal breakdown for one ticker |
| POST | `/api/signals/generate` | Run the enabled strategies and refresh signals |
| GET | `/api/signals/dashboard` | HTML signal dashboard |
| GET | `/api/benchmark` | Benchmarking landing page |
| GET | `/api/benchmark/single` | Benchmark one strategy against a baseline |
| POST | `/api/benchmark/ab-test` | A/B test two strategies |
| GET | `/api/benchmark/results` | Stored benchmark results |

## Strategies

Registered in `app/strategies/strategy_factory.py`; enable any subset with
`ENABLED_STRATEGIES`:

| Name | Approach |
| --- | --- |
| `statistical_arbitrage` | Correlation clustering and cointegration pairs |
| `momentum` | Multi-timeframe momentum with volume and news confirmation |
| `mean_reversion` | Bollinger Bands, RSI and TSI reversal signals |
| `multi_strategy` | Weighted ensemble of the three above (60/25/15 by default) |
| `insider_momentum_advanced` | Form 4 insider buying with technical confirmation |
| `insider_options_flow` | Insider filings combined with unusual options activity |
| `insider_ml_predictor` | Gradient-boosted model over insider and price features |

Signal building blocks (insider buying, gap reversion, alternative growth,
short squeeze, momentum) are in `app/signals.py` and `app/alt_data.py`;
position sizing and stop-loss logic in `app/risk_management/`; backtesting and
attribution in `app/backtesting/`; benchmarking in `app/benchmarking/`.

## Tests

```bash
python -m pytest -q
```

`tests/conftest.py` injects a dummy `POLYGON_API_KEY`, so the suite runs
offline. Tests that need live Polygon data are skipped unless a real key is
exported, and the web-API tests skip when no server is listening on port 8000.
One test is marked `xfail` for a known scoring edge case (see Limitations).

Continuous integration (`.github/workflows/ci.yml`) runs the same command on
Python 3.11 and 3.12.

## Project layout

```
main.py                 ASGI entry point
app/config.py           Settings (pydantic-settings)
app/api.py              FastAPI router (/api)
app/dash_app_simple.py  Dash dashboard mounted at /dash
app/strategies/         Strategy implementations and factory
app/data_sources/       Polygon, Finnhub, FINRA, news, insider and options feeds
app/backtesting/        Backtest engine, attribution, strategy comparison
app/benchmarking/       Benchmark and A/B test framework
app/risk_management/    Position sizing, stop losses, portfolio risk
app/universe*.py        Nano-cap universe screening
tests/                  pytest suite
docs/                   Guides (see below)
```

## Documentation

- [System overview](docs/SYSTEM_OVERVIEW.md)
- [Deployment guide](docs/DEPLOYMENT_GUIDE.md)
- [Remote access guide](docs/REMOTE_ACCESS_GUIDE.md)
- [Backtesting guide](docs/BACKTESTING_GUIDE.md)
- [Benchmarking guide](docs/BENCHMARKING_GUIDE.md)
- [Insider strategies guide](docs/INSIDER_STRATEGIES_GUIDE.md)
- [Nano-cap universe](docs/NANO_CAP_UNIVERSE.md)
- [Testing guide](docs/TESTING_GUIDE.md)

The guides predate the current code in places and describe some features
aspirationally; the code and this README are authoritative.

## Limitations

- Cost and liquidity modelling is simple. Fees are `TRANSACTION_COST_PCT`
  with a `MIN_TRANSACTION_COST` floor, and volume participation is capped at
  `MAX_VOLUME_PCT`. Real nano-cap trading also carries wide bid-ask spreads,
  market impact, stale prints and delisting risk that the models do not price.
  Backtest and benchmark numbers are optimistic.
- The backtesting and benchmarking engines fall back to synthetic returns
  where historical data is missing, so their output is only as good as the
  data you feed them.
- Long-only by default (`ENABLE_SHORT_SELLING=false`); most nano-caps cannot
  be borrowed at reasonable cost.
- `insider_buy_score` z-scores across tickers, so a single buying ticker yields
  NaN rather than a positive score (`tests/test_signals.py` is xfail for this).
- No broker execution path is wired to a live account; `app/broker.py` is a
  thin ib-insync wrapper and orders are not placed automatically.
- Optional paid feeds (Fintel, WhaleWisdom, Tradier, Benzinga, Ortex) are
  integrated but untested against live accounts.

## License

MIT, see [LICENSE](LICENSE).
