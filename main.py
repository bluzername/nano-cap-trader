"""Main ASGI entry point used by Render deployment.
Run with:  gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:10000
"""
from fastapi import FastAPI
from app.api import router as api_router
from app.dash_app import mount_dash

app = FastAPI(title="NanoCap Trader")
app.include_router(api_router)

# Mount Dash GUI under "/gui"
mount_dash(app, path="/gui")