#!/bin/bash
# Start NanoCap Trader with UV virtual environment for external access

echo "🚀 Starting NanoCap Trader with UV venv..."
echo "========================================"

# Navigate to project directory
cd "$(dirname "$0")"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating UV virtual environment..."
    source $HOME/.local/bin/env
    uv venv .venv --python 3.11
    source .venv/bin/activate
    uv pip install -r requirements.txt
else
    echo "✅ Found existing virtual environment"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check Python version
echo "🐍 Python version: $(python --version)"

# Check if uvicorn is installed
if ! python -m uvicorn --version >/dev/null 2>&1; then
    echo "❌ uvicorn not found in venv. Installing..."
    pip install uvicorn[standard]
fi

# Get server IP for display
SERVER_IP=$(hostname -I | awk '{print $1}')
PUBLIC_IP=$(curl -s https://ipinfo.io/ip 2>/dev/null || echo "unknown")

echo ""
echo "🌐 Starting server on all interfaces (0.0.0.0:8000)"
echo "========================================"
echo "📱 Local access: http://127.0.0.1:8000"
echo "🏠 Network access: http://${SERVER_IP}:8000"
echo "🌍 Public access: http://${PUBLIC_IP}:8000"
echo ""
echo "📊 Dashboards:"
echo "  - Main: http://${SERVER_IP}:8000/"
echo "  - Trading: http://${SERVER_IP}:8000/dash/"
echo "  - API Docs: http://${SERVER_IP}:8000/docs"
echo "  - Portfolio: http://${SERVER_IP}:8000/api/portfolio"
echo ""
echo "🛡️ Security: Authentication is ${ENABLE_AUTH:-disabled}"
echo "🛑 Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Start the server with external access
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload