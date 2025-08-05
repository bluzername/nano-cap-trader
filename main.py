"""Main ASGI entry point used by Render deployment.
Run with:  gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:10000
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.api import router as api_router
from app.dash_app_simple import mount_dash

app = FastAPI(title="NanoCap Trader")
app.include_router(api_router)

# Mount Dash GUI
mount_dash(app, path="/dash")

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NanoCap Trader</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2E86AB; text-align: center; }
            .status { background: #d4edda; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .links { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 30px 0; }
            .link-card { background: #007bff; color: white; padding: 20px; text-align: center; border-radius: 8px; text-decoration: none; transition: transform 0.2s; }
            .link-card:hover { transform: translateY(-2px); text-decoration: none; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 NanoCap Trader</h1>
            <p style="text-align: center; color: #666;">Advanced algorithmic trading platform for nano-cap equities</p>
            
            <div class="status">
                <strong>✅ System Status:</strong> Online and ready for trading
            </div>
            
            <div class="links">
                <a href="/dash/" class="link-card">
                    <h3>🚀 Trading Dashboard</h3>
                    <p>Complete trading interface and controls</p>
                </a>
                <a href="/docs" class="link-card">
                    <h3>📚 API Documentation</h3>
                    <p>Interactive API docs with Swagger UI</p>
                </a>
                <a href="/api/portfolio" class="link-card">
                    <h3>📊 Portfolio Status</h3>
                    <p>Current portfolio status and positions</p>
                </a>
                <a href="/api/benchmark" class="link-card">
                    <h3>🧪 Benchmarking</h3>
                    <p>Strategy performance analysis and A/B testing</p>
                </a>
                <a href="/api/signals/dashboard" class="link-card">
                    <h3>📈 Trading Signals</h3>
                    <p>Get buy/sell recommendations and trading signals</p>
                </a>
            </div>
            
            <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; text-align: center; color: #666;">
                <p>Ready to deploy strategies and start trading!</p>
                <p><small>Use the API endpoints to manage strategies, view performance, and execute trades.</small></p>
            </div>
        </div>
    </body>
    </html>
    """
