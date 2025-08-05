#!/bin/bash
# Production remote access script for NanoCap Trader
# Usage: ./start_remote.sh

echo "🚀 Starting NanoCap Trader for remote access..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please copy env.template to .env and configure your API keys"
    exit 1
fi

# Start with gunicorn for production
echo "🌐 Starting server on all interfaces (0.0.0.0:8000)"
echo "📱 Access via: http://YOUR_SERVER_IP:8000"
echo "🛑 Press Ctrl+C to stop"

gunicorn main:app \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --workers 1 \
    --timeout 300 \
    --keep-alive 30 \
    --max-requests 1000 \
    --preload \
    --access-logfile - \
    --error-logfile -