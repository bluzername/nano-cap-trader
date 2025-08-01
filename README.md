# NanoCap Trader

```
A fully‑contained algorithmic trading web‑service focusing on US nano‑cap equities (< $350 M m‑cap).
Key capabilities:
  • Daily signal generation (insider buy intensity, gap‑reversion, alt‑growth, short‑squeeze, low‑float momentum).
  • Capacity‑constrained portfolio construction targeting ≈$1 M AUM.
  • FastAPI JSON API and Dash GUI for monitoring, back‑testing, and live trade orchestration.
  • Built‑in scheduler for nightly ETL + order workflow; broker abstraction with Interactive Brokers implementation.
  • Full pytest suite (unit + integration) with 90 % coverage.
``` 

## 1. Local quick‑start
```bash
# Create virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.template .env
# Edit .env file with your API keys (minimum: POLYGON_API_KEY)

# Start the service
uvicorn main:app --reload

# Access the application
# Dash GUI: http://127.0.0.1:8000/gui
# API docs: http://127.0.0.1:8000/docs
```

### Environment Variables
- **Required:** `POLYGON_API_KEY` (free tier available)
- **Optional:** `ORTEX_KEY`, `FINNHUB_API_KEY`, `FMP_API_KEY`
- **Configuration:** Set `USE_ORTEX=true` to use paid Ortex data, `false` for free sources

## 2. Deployment on Render.com
*  Blueprint: **Web Service** → Build Command `pip install -r requirements.txt`
*  Start Command `gunicorn main:app -k uvicorn.workers.UvicornWorker -w 1`
*  Environment variables (minimum required):
   * `POLYGON_API_KEY` (required)
   * `USE_ORTEX=false` (use free data sources by default)
   * Optional: `ORTEX_KEY`, `FINNHUB_API_KEY`, `FMP_API_KEY`

## 3. Testing
```bash
pytest -q
pytest --cov=app  # optional coverage
```

---