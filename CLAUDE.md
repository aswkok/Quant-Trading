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

**Enhanced MACD Trading (Advanced):**
```bash
# Full Enhanced MACD strategy (requires day trading buying power for shorts)
python integrated_macd_trader.py --symbol NVDA --strategy enhanced_macd --warmup 30 --interval 60 --shares 100

# Long-only Enhanced MACD (avoids PDT restrictions - RECOMMENDED)
python integrated_macd_trader.py --symbol NVDA --strategy enhanced_macd --long-only --warmup 30 --interval 60 --shares 100
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

### Enhanced MACD Strategy Usage

**Create and Use Enhanced MACD Strategy:**
```python
from strategies import EnhancedMACDStrategy, StrategyFactory

# Standard Enhanced MACD with full Cases A, B, C
strategy = EnhancedMACDStrategy(
    slope_threshold=0.002,
    slope_lookback=5,
    histogram_lookback=4,
    shares_per_trade=200,
    long_only=False  # Allows short selling
)

# Long-only Enhanced MACD (avoids day trading buying power issues)
strategy = EnhancedMACDStrategy(
    slope_threshold=0.002,
    slope_lookback=5,
    histogram_lookback=4,
    shares_per_trade=200,
    long_only=True  # Enhanced entries: A1 (crossover), A3 (momentum), B3/B4 (exits)
)

# Via factory with long-only mode
strategy = StrategyFactory.get_strategy('enhanced_macd', 
                                       slope_threshold=0.003,
                                       long_only=True)

# Generate signals
signals = strategy.generate_signals(historical_data)

# Export comprehensive analysis with case indicators (🅰️🅱️🅲️)
strategy.save_signals_to_csv(signals, symbol='NVDA')
```

**Enhanced Quote Monitor with CSV Export:**
```python
from enhanced_quote_monitor import EnhancedQuoteMonitor

# Create monitor and export Enhanced MACD data
monitor = EnhancedQuoteMonitor('NVDA')
monitor.save_enhanced_macd_csv()  # Complete analysis export with case indicators
```

### Testing
```bash
# Run specific test files
python MACD_option_trading/test_market_hours.py
python MACD_option_trading/test_display.py

# Test with pytest (pytest is included in requirements.txt)
pytest
```

### Enhanced Long-Only Strategy Details

**Long-Only Mode provides three sophisticated entry mechanisms:**

#### **Entry Points:**
- **A.1 - Traditional**: Bullish MACD crossover (MACD > Signal)
- **🆕 A.3 - Momentum**: Early momentum strengthening detection
  - MACD still below Signal Line (bearish overall)
  - MACD slope increasing (momentum recovering)
  - Histogram compressing (bearish strength weakening)

#### **Example Trading Sequence:**
```
Scenario: Market showing early signs of recovery
10:00   MACD: -0.08, Signal: -0.05   → STAY_FLAT (still bearish)
10:30   MACD: -0.05, Signal: -0.03   → 🅰️ BUY (Case A.3 - momentum strengthening)
11:00   MACD: +0.01, Signal: +0.02   → HOLD (traditional crossover achieved)
```

### Day Trading Buying Power Troubleshooting

**Problem**: `"insufficient day trading buying power"` error with Alpaca API

**Solution 1: Use Enhanced Long-Only Mode (Recommended)**
```bash
# Enhanced long-only with momentum strengthening entries
python integrated_macd_trader.py --symbol NVDA --strategy enhanced_macd --long-only --shares 100 --warmup 30

# Long-only with custom parameters
python integrated_macd_trader.py --symbol NVDA --strategy enhanced_macd --long-only \
    --slope-threshold 0.0015 --slope-lookback 5 --histogram-lookback 5 \
    --shares 100 --warmup 30 --interval 60
```

**Solution 2: Reduce Position Size**
```bash
# Use smaller quantities
python integrated_macd_trader.py --symbol NVDA --strategy enhanced_macd --shares 10
```

**Solution 3: Check Account Status**
The system now provides automatic diagnostics when PDT errors occur, showing:
- Total buying power vs. day trading buying power
- Pattern Day Trader status
- Specific solutions for your account type

### Configuration
- Main trading mode is controlled in `main.py` via `TRADING_MODE = "PAPER"` (change to "LIVE" for live trading)
- Environment variables are loaded from `.env` file  
- API credentials and trading parameters are configured via environment variables

## Architecture Overview

### Core System Components

**Stock Trading (`/`):**
- `main.py` - Primary entry point for stock trading with Alpaca API integration
- `strategies.py` - Technical analysis strategies (MACD, Enhanced MACD, RSI, Bollinger Bands, Moving Average Crossover)
- `integrated_macd_trader.py` - End-to-end real-time MACD trading system
- `enhanced_quote_monitor.py` - Real-time quote monitoring with improved timestamp handling and CSV export

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
- **Classic MACD**: Signal generation on crossover/crossunder events
- **Enhanced MACD**: Advanced momentum analysis with slope and histogram detection
- Position management: long/short transitions with specific rules
- Time-based throttling (15-minute minimum between position changes)
- Warm-up period for data collection before trading starts

**Enhanced MACD Features:**
- **Complete Case Analysis**: Cases A (no position), B (long position), C (short position)
- **Momentum Detection**: MACD slope calculation for trend direction
- **Advanced Entry/Exit Logic**: Momentum weakening/strengthening before traditional crossovers
- **Failsafe Conditions**: Guaranteed exits on MACD crossover/crossunder signals
- **Enhanced Long-Only Mode**: Three sophisticated entry mechanisms
  - **A.1**: Traditional bullish crossover entries
  - **A.3**: Momentum strengthening entries (captures early trend shifts)
  - **B.3/B.4**: Momentum weakening and failsafe exits
- **Case Indicators**: Visual displays with 🅰️🅱️🅲️ case identification
- **PDT Compliance**: Long-only mode eliminates day trading buying power requirements
- **Configurable Parameters**: slope_threshold, slope_lookback, histogram_lookback, long_only
- **Comprehensive CSV Export**: All calculations, signals, and case indicators

**Risk Management:**
- **Pre-flight Checks**: Validates buying power before order placement
- **PDT Error Handling**: Automatic detection and solution suggestions
- **Long-Only Mode**: Eliminates day trading buying power requirements
- Position sizing based on account value and risk tolerance
- Maximum position limits via `MAX_POSITIONS` environment variable
- Risk per trade controlled via `RISK_PER_TRADE` (default: 2%)
- Extended hours and overnight trading controls
- **Enhanced Diagnostics**: Account status reporting on trade failures

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

### Data Export and Analysis
- **CSV Export**: Comprehensive data export for backtesting and analysis
- **Enhanced Quote Monitor**: `save_enhanced_macd_csv()` exports real-time quotes with strategy data
- **Strategy Signals Export**: `save_signals_to_csv()` exports complete strategy calculations
- **Exported Data**: OHLC, MACD indicators, slopes, histograms, positions, actions, trigger reasons
- **Metadata Inclusion**: Strategy parameters and configuration saved with data
- **Time Series Format**: Index-preserved CSV for proper temporal analysis

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
- `alpaca-py>=0.42.0` - Alpaca API client (updated for pydantic 2.x compatibility)
- `pydantic>=2.0.3,<3.0.0` - Data validation and serialization
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