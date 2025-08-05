# 🚀 NanoCap Trader - Complete User Guide

## 📋 What is NanoCap Trader?

**NanoCap Trader** is a professional-grade algorithmic trading system designed specifically for US nano-cap equities (companies with market capitalization under $350 million). It implements cutting-edge academic research with institutional-level risk management and provides both web and command-line interfaces.

###  **Key Features:**
- **4 Research-Backed Trading Strategies** with proven alpha generation
- **Professional Web Dashboard** for monitoring and control
- **Advanced Risk Management** with real-time monitoring
- **A/B Testing Framework** for strategy comparison
- **Multi-Source Data Integration** with free and paid options
- **Production-Ready Deployment** for both local and cloud hosting

---

## 🚀 Quick Start (5 Minutes)

### **Step 1: Prerequisites**
```bash
# Ensure you have Python 3.11+ installed
python --version

# Create and navigate to project directory
mkdir nano_cap_trader && cd nano_cap_trader
```

### **Step 2: Setup Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Step 3: Get API Keys**
1. **Polygon.io** (Required - Free tier available):
   - Visit: https://polygon.io/
   - Sign up for free account
   - Copy your API key from dashboard

2. **Optional APIs** (Enhanced features):
   - **NewsAPI.org**: https://newsapi.org/ (1000 free requests/day)
   - **Alpha Vantage**: https://www.alphavantage.co/ (25 free requests/day)
   - **Finnhub**: https://finnhub.io/ (free tier available)

### **Step 4: Configure Environment**
```bash
# Copy environment template
cp env.template .env

# Edit .env file with your API keys
# Minimum required: POLYGON_API_KEY
```

### **Step 5: Start the System**
```bash
# Start the web server
uvicorn main:app --reload

# Access the application
# Main Dashboard: http://127.0.0.1:8000
# Trading GUI: http://127.0.0.1:8000/dash/
# API Documentation: http://127.0.0.1:8000/docs
```

---

## 🌐 Web Interface Guide

### **Main Dashboard** (`http://127.0.0.1:8000`)
The landing page provides quick access to all system components:
- **📊 Portfolio Status**: View current positions and portfolio value
- **🧪 Benchmarking**: Compare strategies and run A/B tests
- **📚 API Documentation**: Interactive API reference
- **📊 Trading Dashboard**: Full trading interface

### **Trading Dashboard** (`http://127.0.0.1:8000/dash/`)
The main trading interface with real-time monitoring:

#### **System Status Section**
- **API Connection Status**: Shows if Polygon.io API is connected
- **Strategy Status**: Lists all available trading strategies
- **Risk Management**: Shows active risk controls

#### **Quick Links**
- **API Documentation**: Interactive API reference
- **Portfolio Status**: Current positions and performance
- **Benchmarking**: Strategy comparison tools
- **Raw API Data**: Direct API access

### **Portfolio Status** (`http://127.0.0.1:8000/api/portfolio`)
Comprehensive portfolio overview:
- **Available Cash**: Current cash balance
- **Total Portfolio Value**: Cash + positions
- **Active Positions**: Number of current holdings
- **Position Details**: Symbol, shares, average price, current value, P&L
- **System Information**: API status, strategy status, risk management status

### **Benchmarking Dashboard** (`http://127.0.0.1:8000/api/benchmark`)
Advanced strategy analysis and comparison:

#### **Single Strategy Benchmark**
Compare one strategy against market benchmarks:
1. **Select Strategy**: Choose from Statistical Arbitrage, Momentum, Mean Reversion, or Multi-Strategy
2. **Select Benchmark**: Russell 2000, S&P 600, or custom
3. **Set Date Range**: Choose start and end dates
4. **Run Analysis**: Get comprehensive performance metrics

**Results Include:**
- **Strategy Performance**: Total return, annualized return, Sharpe ratio, max drawdown
- **vs Benchmark**: Alpha, beta, excess return, information ratio

#### **A/B Testing**
Compare multiple strategies head-to-head:
1. **Select Strategies**: Choose 2-5 strategies to compare
2. **Set Date Range**: Choose testing period
3. **Run Comparison**: Get statistical analysis

**Results Include:**
- **Recommended Strategy**: Best performing with confidence level
- **Performance Table**: Return, Sharpe, max drawdown for each strategy
- **Statistical Significance**: Confidence intervals and p-values

---

##  Trading Strategies

### **1. Statistical Arbitrage** (4.5% Alpha Target)
**How it works:** Identifies pairs of stocks that move together and trades when they diverge.

**Key Features:**
- Hierarchical clustering to find correlated stocks
- Cointegration testing for stable relationships
- Z-score based entry/exit signals
- Maximum 20 pairs simultaneously

**Best for:** Low volatility, consistent returns

### **2. Momentum Trading** (4.2% Alpha Target)
**How it works:** Identifies stocks with strong upward price momentum and volume.

**Key Features:**
- Multi-timeframe momentum analysis
- Volume confirmation signals
- News sentiment integration
- Float analysis for liquidity

**Best for:** Trending markets, high volatility periods

### **3. Mean Reversion** (3.5% Alpha Target)
**How it works:** Trades stocks that have moved too far from their average price.

**Key Features:**
- Bollinger Bands for overbought/oversold detection
- RSI for momentum confirmation
- Volume ratio analysis
- Time-based position management

**Best for:** Sideways markets, contrarian opportunities

### **4. Multi-Strategy Ensemble** (4.0% Alpha Target)
**How it works:** Combines all three strategies with intelligent weighting.

**Key Features:**
- 60% Statistical Arbitrage
- 25% Momentum Trading
- 15% Mean Reversion
- Dynamic rebalancing every 6 hours
- Correlation-based position limits

**Best for:** Balanced approach, reduced volatility

---

## ⚙️ Configuration Options

### **Environment Variables** (`.env` file)

#### **Required Settings:**
```bash
# Required - Get from https://polygon.io/
POLYGON_API_KEY=your_polygon_api_key_here
```

#### **Optional APIs (Enhanced Features):**
```bash
# News sentiment for momentum strategy
NEWSAPI_KEY=your_newsapi_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here

# Additional data sources
FINNHUB_API_KEY=your_finnhub_key_here
FMP_API_KEY=your_fmp_key_here
```

#### **Strategy Configuration:**
```bash
# Which strategies to enable
ENABLED_STRATEGIES=multi_strategy

# Data source preference
USE_ORTEX=false  # Use free sources instead of paid Ortex
```

#### **Risk Management:**
```bash
# Position sizing controls
ENABLE_POSITION_SIZING=true
MAX_VOLUME_PCT=0.03  # 3% of daily volume limit

# Stop-loss controls
ENABLE_STOP_LOSS=true
STOP_LOSS_PCT=0.02  # 2% stop-loss
```

---

## ️ Risk Management

### **Position Sizing**
- **Volume-Based Limits**: Maximum 3% of daily trading volume per position
- **Cross-Strategy Aggregation**: Total exposure across all strategies
- **Portfolio Percentage**: Maximum 2% of portfolio per position

### **Stop-Loss Protection**
- **2-Standard Deviation Stops**: Automatic exit on significant moves
- **Time-Based Exits**: Close positions that don't move as expected
- **Market Orders**: Fast execution for risk management

### **Portfolio Risk Controls**
- **VaR Monitoring**: 95% Value at Risk limits
- **Sector Concentration**: Maximum 25% in any sector
- **Correlation Limits**: Prevent over-concentration in similar stocks
- **Leverage Controls**: Maximum 1.5x leverage

### **Emergency Controls**
- **Emergency Stop**: Immediately close all positions
- **Risk Alerts**: Real-time notifications for limit breaches
- **Automatic Shutdown**: System stops trading on critical errors

---

## 📊 Performance Metrics

### **Return Metrics**
- **Total Return**: Overall portfolio performance
- **Annualized Return**: Yearly performance rate
- **Alpha**: Excess return vs benchmark
- **Beta**: Market sensitivity

### **Risk Metrics**
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk adjustment
- **Maximum Drawdown**: Largest peak-to-trough decline
- **VaR (95%)**: 95% confidence loss limit

### **Trading Metrics**
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **Average Trade**: Mean profit/loss per trade
- **Information Ratio**: Alpha / tracking error

---

## 🔧 Command Line Interface

### **Starting the System**
```bash
# Development mode (with auto-reload)
uvicorn main:app --reload

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000

# With specific configuration
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### **Running Tests**
```bash
# Quick test run
pytest -q

# With coverage report
pytest --cov=app

# Specific test file
pytest tests/test_strategies.py
```

### **Database Operations**
```bash
# Initialize database
python -c "from app.database import init_db; init_db()"

# Reset database
python -c "from app.database import reset_db; reset_db()"
```

### **Strategy Management**
```bash
# List available strategies
python -c "from app.strategies import StrategyFactory; print(StrategyFactory.get_available_strategies())"

# Test strategy creation
python -c "from app.strategies import StrategyFactory; strategy = StrategyFactory.create_strategy('multi_strategy', ['AAPL', 'MSFT']); print(f'Strategy created: {strategy.strategy_id}')"
```

---

##  Deployment Options

### **Local Development**
```bash
# Standard local setup
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### **Production Deployment**
```bash
# Using gunicorn for production
gunicorn main:app -k uvicorn.workers.UvicornWorker -w 4 --bind 0.0.0.0:8000
```

### **Cloud Deployment (Render.com)**
1. **Create Render Account**: https://render.com/
2. **New Web Service**: Connect your GitHub repository
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `gunicorn main:app -k uvicorn.workers.UvicornWorker -w 1`
5. **Environment Variables**: Add your API keys in Render dashboard

### **Docker Deployment**
```bash
# Build Docker image
docker build -t nano-cap-trader .

# Run container
docker run -p 8000:8000 --env-file .env nano-cap-trader
```

---

## 🔍 Troubleshooting

### **Common Issues**

#### **"Module not found" errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### **API connection errors**
```bash
# Check API key in .env file
cat .env | grep POLYGON_API_KEY

# Test API connection
curl "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-01?apiKey=YOUR_KEY"
```

#### **Port already in use**
```bash
# Kill existing processes
pkill -f uvicorn

# Use different port
uvicorn main:app --port 8001
```

#### **Database errors**
```bash
# Reset database
python -c "from app.database import reset_db; reset_db()"

# Check database file
ls -la *.db
```

### **Performance Issues**
- **High Memory Usage**: Reduce number of concurrent strategies
- **Slow API Calls**: Check API rate limits and upgrade if needed
- **Database Slowdown**: Consider database optimization or migration

### **Getting Help**
1. **Check Logs**: Look at server output for error messages
2. **API Documentation**: Visit `/docs` for interactive API reference
3. **Test Suite**: Run `pytest` to verify system functionality
4. **Configuration**: Verify all environment variables are set correctly

---

## 📚 Advanced Usage

### **Custom Strategy Development**
```python
from app.strategies.base_strategy import BaseStrategy
from app.strategies.strategy_types import StrategyType

class MyCustomStrategy(BaseStrategy):
    def __init__(self, universe, **kwargs):
        super().__init__(
            strategy_id="my_custom",
            strategy_type=StrategyType.CUSTOM,
            universe=universe,
            **kwargs
        )
    
    def generate_signals(self):
        # Your custom signal logic here
        pass
```

### **Custom Risk Management**
```python
from app.risk_management.portfolio_risk import PortfolioRiskManager

# Custom risk limits
risk_manager = PortfolioRiskManager(
    max_var_pct=0.02,  # 2% VaR limit
    max_leverage=1.5,  # 1.5x leverage limit
    sector_limit=0.25  # 25% sector concentration
)
```

### **Data Source Integration**
```python
from app.data_sources.polygon_data import PolygonDataSource

# Custom data source
data_source = PolygonDataSource(api_key="your_key")
data = data_source.get_daily_bars("AAPL", "2024-01-01", "2024-01-31")
```

---

## 📈 Performance Expectations

### **Historical Performance Targets**
- **Multi-Strategy Alpha**: 4.0% annually vs Russell 2000
- **Maximum Sharpe Ratio**: 0.89 (Statistical Arbitrage)
- **Target Drawdown**: 12-15% maximum
- **Win Rate**: 52-61% across strategies
- **Transaction Costs**: 0.1% per trade (built-in)

### **Risk-Adjusted Metrics**
- **Information Ratio**: 0.65+ target
- **Calmar Ratio**: 0.28+ target
- **Sortino Ratio**: 0.85+ target
- **VaR (95%)**: 2% daily maximum

### **Realistic Expectations**
- **Paper Trading**: Test strategies before live trading
- **Market Conditions**: Performance varies with market environment
- **Transaction Costs**: Account for slippage and fees
- **Risk Management**: Always use stop-losses and position sizing

---

## 🔐 Security Best Practices

### **API Key Management**
- **Never commit API keys** to version control
- **Use environment variables** for all sensitive data
- **Rotate keys regularly** for production systems
- **Monitor API usage** for unusual activity

### **System Security**
- **Use HTTPS** for production deployments
- **Implement authentication** for web interfaces
- **Regular backups** of configuration and data
- **Monitor system logs** for security events

### **Trading Security**
- **Paper trading first** to validate strategies
- **Start with small positions** when going live
- **Monitor risk limits** continuously
- **Have emergency procedures** for system failures

---

## 📞 Support & Resources

### **Documentation**
- **API Reference**: `/docs` (interactive)
- **System Overview**: `SYSTEM_OVERVIEW.md`
- **Deployment Guide**: `DEPLOYMENT_GUIDE.md`

### **Testing**
- **Unit Tests**: `pytest tests/`
- **Integration Tests**: `pytest tests/integration/`
- **Web Interface Tests**: `python comprehensive_web_test.py`

### **Monitoring**
- **System Status**: Check web dashboard
- **Performance Metrics**: Use benchmarking tools
- **Risk Monitoring**: Real-time alerts in web interface

---

## 🎯 Next Steps

1. **Start with Paper Trading**: Test strategies without real money
2. **Monitor Performance**: Use benchmarking tools to track results
3. **Adjust Parameters**: Fine-tune strategy parameters based on results
4. **Scale Gradually**: Increase position sizes as confidence grows
5. **Stay Updated**: Monitor system logs and performance metrics

---

**🚀 Ready to start trading? Visit http://127.0.0.1:8000 to access your NanoCap Trader dashboard!**