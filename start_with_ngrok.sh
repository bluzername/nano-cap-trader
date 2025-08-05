#!/bin/bash
# Start NanoCap Trader with ngrok tunnel
# Usage: ./start_with_ngrok.sh

echo "🚀 Starting NanoCap Trader with ngrok tunnel..."

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok not found. Installing..."
    
    # Download and install ngrok
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
    sudo apt update && sudo apt install ngrok
    
    echo "📝 Please run 'ngrok config add-authtoken YOUR_TOKEN' with your ngrok token"
    echo "Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken"
    exit 1
fi

# Start the app in background
echo "🌐 Starting NanoCap Trader server..."
uvicorn main:app --host 127.0.0.1 --port 8000 &
APP_PID=$!

# Wait for server to start
sleep 3

# Start ngrok tunnel
echo "🔗 Creating ngrok tunnel..."
ngrok http 8000 &
NGROK_PID=$!

echo "✅ Setup complete!"
echo "📱 Your app will be available at the ngrok URL (usually https://xxxxx.ngrok.io)"
echo "🖥️  ngrok dashboard: http://127.0.0.1:4040"
echo "🛑 Press Ctrl+C to stop both services"

# Wait for interrupt
trap "kill $APP_PID $NGROK_PID; exit" INT
wait