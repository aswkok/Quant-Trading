# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive quantitative trading system built around MACD (Moving Average Convergence Divergence) strategies for both stock and options trading using the Alpaca API. The system features real-time market data integration, automated trading execution, and sophisticated risk management.

## Common Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (copy .env.example to .env and configure)
cp .env.example .env
```

### Running the Trading System

**Main Stock Trading System:**
```bash
python main.py
```

**Integrated Real-Time MACD Trading:**
```bash
python integrated_macd_trader.py --symbol NVDA --warmup 30 --interval 60 --shares 100
```

**Options Trading System:**
```bash
cd MACD_option_trading
python main.py
```

**Real-Time Quote Monitoring:**
```bash
python enhanced_quote_monitor.py --symbol NVDA --interval 60
```

### Testing
```bash
# Run specific test files
python MACD_option_trading/test_market_hours.py
python MACD_option_trading/test_display.py

# Test with pytest (pytest is included in requirements.txt)
pytest
```

### Configuration
- Main trading mode is controlled in `main.py` via `TRADING_MODE = "PAPER"` (change to "LIVE" for live trading)
- Environment variables are loaded from `.env` file
- API credentials and trading parameters are configured via environment variables

## Architecture Overview

### Core System Components

**Stock Trading (`/`):**
- `main.py` - Primary entry point for stock trading with Alpaca API integration
- `strategies.py` - Technical analysis strategies (MACD, RSI, Bollinger Bands, Moving Average Crossover)
- `integrated_macd_trader.py` - End-to-end real-time MACD trading system
- `enhanced_quote_monitor.py` - Real-time quote monitoring with improved timestamp handling

**Options Trading (`MACD_option_trading/`):**
- `main.py` - Options trading main entry point
- `macd_options_strategy.py` - MACD strategy adapted for options
- `options_trader.py` - Options contract management
- `continuous_options_trader_service.py` - Background trading service

**Real-Time Data System:**
- `yahoo_quote_monitor.py` - Yahoo Finance data integration
- `quote_monitor_selector.py` - Data source selection wrapper (Alpaca/Yahoo)
- Multiple display systems: `real_time_display.py`, `web_display.py`, `console_display.py`

### Data Sources
- **Primary**: Alpaca API (real-time WebSocket streaming)
- **Fallback**: Yahoo Finance
- **Configuration**: Data source selection via environment variables

### Key Trading Features

**MACD Strategy Implementation:**
- Signal generation on crossover/crossunder events
- Position management: long/short transitions with specific rules
- Time-based throttling (15-minute minimum between position changes)
- Warm-up period for data collection before trading starts

**Risk Management:**
- Position sizing based on account value and risk tolerance
- Maximum position limits via `MAX_POSITIONS` environment variable
- Risk per trade controlled via `RISK_PER_TRADE` (default: 2%)
- Extended hours and overnight trading controls

**Options-Specific Features:**
- Three trading styles: Directional, Income, Combined
- Greeks tracking (Delta, Gamma, Theta, Vega)
- Strike selection based on delta targeting
- Automatic expiration management and rolling

### State Management
- Strategy states persisted in JSON files under `/state/` directories
- Position tracking for individual symbols (e.g., `NVDA_position.json`)
- Trading statistics and performance tracking
- Robust error recovery and system health monitoring

## Development Patterns

### Configuration Management
- All API credentials and sensitive data via environment variables
- Trading parameters configurable via `.env` file
- Paper/live trading mode switching in code constants

### Logging
- Comprehensive logging to both files and console
- Separate log files: `trading.log`, `integrated_trading.log`
- Structured logging with timestamps and levels

### Error Handling
- Robust error recovery for API failures
- Market hours detection with automatic schedule management
- Graceful fallback between data sources

### Market Hours Support
- Extended hours trading (4:00 AM - 9:30 AM ET, 4:00 PM - 8:00 PM ET)
- Overnight trading (8:00 PM - 4:00 AM ET)
- Automatic market schedule detection

## Key Environment Variables

Required in `.env` file:
- `ALPACA_API_KEY` - Alpaca API key
- `ALPACA_API_SECRET` - Alpaca API secret  
- `ALPACA_BASE_URL` - API endpoint (paper: `https://paper-api.alpaca.markets`, live: `https://api.alpaca.markets`)

Optional configuration:
- `RISK_PER_TRADE` - Risk percentage per trade (default: 2%)
- `MAX_POSITIONS` - Maximum concurrent positions (default: 5)
- `EXTENDED_HOURS` - Enable extended hours trading (default: True)
- `OVERNIGHT_TRADING` - Enable overnight trading (default: True)

## Testing Strategy

- Unit tests located in test files (`test_*.py`)
- Market hours testing via `test_market_hours.py`
- Display system testing via `test_display.py` and `test_safe_display.py`
- API connectivity testing via `test_api.py`

## Dependencies

Key dependencies from `requirements.txt`:
- `alpaca-py>=0.8.0` - Alpaca API client
- `pandas>=2.0.0`, `numpy>=1.24.0` - Data processing
- `yfinance>=0.2.0` - Yahoo Finance data
- `matplotlib>=3.7.0` - Plotting and visualization
- `python-dotenv>=1.0.0` - Environment variable management
- `pytest==7.4.0` - Testing framework

## Security Notes

- Never commit API keys or secrets to the repository
- Use `.env` file for all sensitive configuration
- Default configuration uses paper trading mode for safety
- All real money trading requires explicit mode switching