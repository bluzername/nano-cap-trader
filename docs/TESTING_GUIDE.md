# Testing Guide for Nano-Cap Trading System

This guide covers testing procedures for the nano-cap trading system, including unit tests, integration tests, and end-to-end validation.

## 📋 Overview

The testing suite validates:

- **API Integration**: All mandatory APIs work with proper keys
- **Market Data**: Real-time data fetching and processing
- **Strategy Execution**: All trading strategies function correctly
- **Portfolio Management**: Position tracking and risk management
- **Web Interface**: API endpoints and user interface
- **Data Quality**: Universe integrity and validation
- **Error Handling**: Graceful failure recovery

## 🚀 Quick Health Check

For a rapid system health check, run:

```bash
# Quick health check (recommended first step)
python tests/health_check.py
```

This performs essential validations and provides immediate feedback on system status.

## 📊 Test Categories

### 1. Health Check (`tests/health_check.py`)
- **Purpose**: Quick system validation
- **Runtime**: ~30 seconds
- **Scope**: Core functionality, API connectivity, basic strategy execution

### 2. End-to-End Integration Tests (`tests/test_e2e_integration.py`)
- **Purpose**: Comprehensive system testing
- **Runtime**: ~5-10 minutes
- **Scope**: Full workflow testing with real APIs

## 🔧 Prerequisites

### Required API Keys

**Mandatory:**
- `POLYGON_API_KEY` - Polygon.io API key for market data

**Optional (for enhanced functionality):**
- `ALPHA_VANTAGE_API_KEY`
- `ORTEX_TOKEN`
- `FINTEL_API_KEY`
- `WHALEWISDOM_API_KEY`
- `TRADIER_API_KEY`
- `BENZINGA_API_KEY`

### Environment Setup

1. **Configure API Keys**:
   ```bash
   # Copy and edit environment file
   cp env.template .env
   # Edit .env with your API keys
   nano .env
   ```

2. **Install Test Dependencies**:
   ```bash
   # Install pytest and testing tools
   pip install pytest pytest-asyncio requests
   ```

3. **Start the Server**:
   ```bash
   # Start server for API tests
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## 🏃‍♂️ Running Tests

### Quick Health Check
```bash
# Basic system health check
python tests/health_check.py

# Expected output:
# 🚀 Starting Nano-Cap Trading System Health Check
# 🔍 Checking API Keys Configuration... ✅ PASS
# 🔍 Checking Market Data Connectivity... ✅ PASS
# ...
# 🎉 ALL SYSTEMS OPERATIONAL!
```

### Comprehensive Integration Tests
```bash
# Run all integration tests
python -m pytest tests/test_e2e_integration.py -v

# Run specific test categories
python -m pytest tests/test_e2e_integration.py::TestConfiguration -v
python -m pytest tests/test_e2e_integration.py::TestMarketDataIntegration -v
python -m pytest tests/test_e2e_integration.py::TestStrategyExecution -v

# Run with detailed output
python -m pytest tests/test_e2e_integration.py -v -s --tb=long
```

### Test Markers
```bash
# Run only fast tests
python -m pytest -m "not slow" -v

# Run only integration tests
python -m pytest -m integration -v

# Run tests requiring API keys
python -m pytest -m requires_keys -v
```

## 🧪 Test Scenarios

### Configuration Tests
- ✅ API keys properly configured
- ✅ Environment variables set
- ✅ Database connectivity
- ✅ Portfolio limits configured

### Market Data Tests
- ✅ Polygon.io API connectivity
- ✅ Real-time data fetching
- ✅ Data quality validation
- ✅ Error handling for invalid symbols
- ✅ Rate limiting management

### Strategy Tests
- ✅ All strategies instantiate correctly
- ✅ Signal generation works
- ✅ Insider strategies with placeholder data
- ✅ Position sizing calculations
- ✅ Risk management controls

### API Endpoint Tests
- ✅ Server running and accessible
- ✅ Status API endpoint
- ✅ Signals generation API
- ✅ Benchmarking API
- ✅ Dashboard accessibility

### Data Quality Tests
- ✅ Universe integrity (100 nano-cap stocks)
- ✅ Symbol validation
- ✅ Data availability rates
- ✅ Sector diversification

## 🔍 Interpreting Results

### Health Check Results

**✅ PASS**: Feature working correctly
**⚠️ WARN**: Minor issues, system functional
**❌ FAIL**: Critical issues requiring attention

### Common Issues and Solutions

#### API Key Issues
```
❌ FAIL - Polygon.io API key not configured or using demo key
```
**Solution**: Configure valid Polygon.io API key in `.env` file

#### Server Connectivity Issues
```
❌ FAIL - Web server not accessible
```
**Solution**: Start server with `uvicorn main:app --host 0.0.0.0 --port 8000`

#### Market Data Issues
```
⚠️ WARN - Data availability too low: 45%
```
**Solution**: Check API rate limits, verify symbol validity

#### Strategy Execution Issues
```
❌ FAIL - Strategy momentum failed: ImportError
```
**Solution**: Check dependencies, review error logs

## 📈 Performance Benchmarks

### Expected Performance Metrics

| Test Category | Expected Runtime | Success Rate |
|---------------|------------------|--------------|
| Health Check | < 30 seconds | > 95% |
| Market Data | < 60 seconds | > 80% |
| Strategy Tests | < 120 seconds | > 90% |
| API Tests | < 30 seconds | > 95% |

### Universe Coverage

- **Total Universe**: 100 nano-cap stocks
- **High Volume Subset**: 30 most liquid stocks
- **Expected Data Availability**: > 70% for high-volume subset
- **Sector Diversification**: 8 sectors represented

## 🚨 Troubleshooting

### Test Fails with Import Errors
```bash
# Ensure you're in the project root
cd /path/to/nano-cap-trader

# Check Python path
python -c "import sys; print(sys.path)"

# Install missing dependencies
pip install -r requirements.txt
```

### Server Not Running
```bash
# Check if server is running
curl http://localhost:8000/api/status

# Start server manually
uvicorn main:app --host 0.0.0.0 --port 8000

# Check server logs
tail -f server.log
```

### API Rate Limits
```bash
# Reduce test universe size
export TEST_UNIVERSE_SIZE=10

# Use high-volume subset only
export USE_HIGH_VOLUME_ONLY=true
```

### Database Issues
```bash
# Check database configuration
python -c "from app.config import get_settings; print(get_settings().database_url)"

# Reset database (if needed)
python -c "from app.database import reset_database; reset_database()"
```

## 🔄 Continuous Integration

### GitHub Actions Integration
```yaml
# Example workflow for automated testing
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run health check
        run: python tests/health_check.py
        env:
          POLYGON_API_KEY: ${{ secrets.POLYGON_API_KEY }}
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## 📊 Test Coverage

### Coverage Analysis
```bash
# Install coverage tools
pip install pytest-cov

# Run with coverage
python -m pytest --cov=app tests/ --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Target Coverage
- **Overall**: > 80%
- **Core Strategies**: > 90%
- **API Endpoints**: > 95%
- **Configuration**: > 90%

## 🎯 Best Practices

### Writing New Tests
1. Use descriptive test names
2. Test both success and failure cases  
3. Mock external dependencies when appropriate
4. Use fixtures for common setup
5. Add appropriate test markers

### Test Organization
```
tests/
├── __init__.py
├── pytest.ini                 # Pytest configuration
├── health_check.py            # Quick health check
├── test_e2e_integration.py    # End-to-end tests
├── unit/                      # Unit tests
│   ├── test_strategies.py
│   ├── test_portfolio.py
│   └── test_universe.py
└── fixtures/                  # Test data and fixtures
    ├── mock_data.py
    └── test_config.py
```

### Performance Testing
```bash
# Profile test performance
python -m pytest --profile-svg

# Memory usage testing
python -m pytest --memray

# Load testing for API endpoints
locust -f tests/load_test.py
```

## 🛡️ Security Testing

### API Security
- ✅ API key validation
- ✅ Rate limiting
- ✅ Input sanitization
- ✅ Authentication checks

### Data Security
- ✅ Sensitive data masking
- ✅ Secure configuration storage
- ✅ Connection encryption

## 📝 Test Documentation

### Test Reports
```bash
# Generate detailed test report
python -m pytest --html=reports/test_report.html --self-contained-html

# Generate JUnit XML for CI
python -m pytest --junitxml=reports/junit.xml
```

### Metrics Collection
The test suite automatically collects:
- Test execution times
- Success/failure rates
- API response times
- Data quality metrics
- Coverage statistics

## 🎉 Conclusion

This testing framework ensures the nano-cap trading system operates reliably with real market data and API integrations. Regular testing validates system health and catches issues early.

For questions or issues with testing, check the logs or run the health check script for diagnostic information.