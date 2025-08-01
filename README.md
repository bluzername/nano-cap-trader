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
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
# open http://127.0.0.1:8000/gui  (Dash GUI)
# open http://127.0.0.1:8000/docs (Swagger)
```

## 2. Deployment on Render.com
*  Blueprint: **Web Service** → Build Command `pip install -r requirements.txt`
*  Start Command `gunicorn main:app -k uvicorn.workers.UvicornWorker -w 1`
*  Environment variables    
   * `POLYGON_API_KEY`, `ORTEX_KEY`, etc.

## 3. Testing
```bash
pytest -q
pytest --cov=app  # optional coverage
```

---