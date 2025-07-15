# Alpaca Quantitative Trading System

A comprehensive, production-ready quantitative trading system for stocks and options using the Alpaca API with advanced MACD strategies and real-time market data integration.

## Overview

This system provides a complete algorithmic trading infrastructure featuring:

- **Real-time MACD-based trading** with live market data streams
- **Dual-mode architecture**: Stock trading and sophisticated options trading
- **Multiple data sources**: Alpaca API (primary) with Yahoo Finance fallback
- **Advanced risk management** with position sizing and portfolio limits
- **Extended hours trading** support (pre-market, after-hours, overnight)
- **Real-time monitoring** with terminal and web-based interfaces
- **State persistence** and robust error recovery
- **Comprehensive logging** and performance tracking

## Getting Started

### Prerequisites

- Python 3.8+
- Alpaca API account (sign up at [Alpaca](https://alpaca.markets/))
- API key and secret from Alpaca

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd "Quant Trading"
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```bash
   cp .env.example .env  # If .env.example exists
   # Or create .env file with your Alpaca API credentials
   ```
   
   Required environment variables in `.env`:
   ```env
   ALPACA_API_KEY=your_api_key_here
   ALPACA_API_SECRET=your_api_secret_here
   ALPACA_BASE_URL=https://paper-api.alpaca.markets  # For paper trading
   # ALPACA_BASE_URL=https://api.alpaca.markets      # For live trading
   ```

## Usage

### Stock Trading System

**Main Stock Trading (Multiple Strategies):**
```bash
python main.py
```
Executes various technical analysis strategies including MACD, RSI, Bollinger Bands, and Moving Average Crossover on default stocks (AAPL, MSFT, AMZN, GOOGL, META).

**Real-Time MACD Trading (Recommended):**
```bash
python integrated_macd_trader.py --symbol NVDA --warmup 30 --interval 60 --shares 100
```
The primary trading system that integrates real-time quote monitoring, MACD calculation, and trade execution into a single workflow.

**Command Line Options for Integrated MACD Trading:**
- `--symbol`: Stock symbol to trade (default: NVDA)
- `--interval`: Seconds between quote fetches (default: 60)
- `--shares`: Number of shares per trade (default: 100)
- `--fast-window`: Fast EMA window for MACD (default: 13)
- `--slow-window`: Slow EMA window for MACD (default: 21)
- `--signal-window`: Signal line window for MACD (default: 9)
- `--extended-hours`: Enable trading during pre-market and after-hours
- `--warmup`: Data collection period before trading in minutes (default: 60)

### Options Trading System

**Advanced Options Trading with MACD:**
```bash
cd MACD_option_trading
python main.py
```
Sophisticated options trading system with MACD-based strategies, Greeks tracking, and automated contract management.

### Real-Time Monitoring

**Enhanced Quote Monitoring:**
```bash
python enhanced_quote_monitor.py --symbol NVDA --interval 60
```
Advanced real-time monitoring with improved timestamp handling and reliable MACD calculations.

**Yahoo Finance Quote Monitoring:**
```bash
python yahoo_quote_monitor.py --symbol NVDA --interval 60
```
Alternative data source monitoring using Yahoo Finance API.

### Testing and Development

**Run Tests:**
```bash
# Market hours testing
python MACD_option_trading/test_market_hours.py

# Display system testing
python MACD_option_trading/test_display.py

# Run all tests with pytest
pytest
```

**Background Services:**
```bash
# Continuous options trading service
python MACD_option_trading/continuous_options_trader_service.py

# Web-based monitoring interface
python MACD_option_trading/web_display.py
```

### Customizing Strategies

Strategies can be customized by editing the `strategies.py` module or creating new strategy classes. The system supports multiple technical analysis approaches beyond MACD.

## System Architecture

### Core Components

**Stock Trading System (`/`):**
- `main.py` - Primary entry point with multiple strategy support
- `integrated_macd_trader.py` - Complete real-time MACD trading system
- `strategies.py` - Technical analysis strategies library
- `enhanced_quote_monitor.py` - Real-time quote monitoring with enhanced features

**Options Trading System (`MACD_option_trading/`):**
- `main.py` - Options trading main entry point
- `macd_options_strategy.py` - MACD strategy adapted for options
- `options_trader.py` - Options contract management and execution
- `continuous_options_trader_service.py` - Background trading service

**Real-Time Data Infrastructure:**
- `yahoo_quote_monitor.py` - Yahoo Finance data integration
- `quote_monitor_selector.py` - Intelligent data source selection
- `real_time_display.py` - Terminal-based monitoring interface
- `web_display.py` - Browser-based monitoring dashboard

### Key Features

**Multi-Source Data Integration:**
- Primary: Alpaca API with WebSocket streaming
- Fallback: Yahoo Finance for reliability
- Automatic source switching and error recovery

**Advanced MACD Implementation:**
- Real-time calculation with live bid/ask data
- Crossover/crossunder signal detection
- Position state management with transitions
- Time-based throttling to prevent overtrading

**Extended Trading Hours:**
- Pre-market: 4:00 AM - 9:30 AM ET
- Regular: 9:30 AM - 4:00 PM ET  
- After-hours: 4:00 PM - 8:00 PM ET
- Overnight: 8:00 PM - 4:00 AM ET

## MACD Trading Strategy Details

The MACD (Moving Average Convergence Divergence) strategy implemented in this system works as follows:

### Core MACD Components
- **MACD Line**: The difference between a fast EMA (default: 13-period) and a slow EMA (default: 21-period)
- **Signal Line**: An EMA of the MACD line (default: 9-period)
- **Histogram**: The difference between the MACD line and signal line

### Trading Rules
1. **Initial Position Rules**:
   - When no position exists:
     - If MACD is above the signal line: BUY shares
     - If MACD is below the signal line: SHORT shares

2. **Crossover Signals**:
   - When MACD crosses above the signal line ("crossover"):
     - If currently short: Cover the short position AND buy additional shares
     - If no position: Buy shares
   - When MACD crosses below the signal line ("crossunder"):
     - If currently long: Sell all shares AND short additional shares
     - If no position: Short shares

3. **Position Mismatch Handling**:
   - If long but MACD is below signal line for >15 minutes: Sell and short
   - If short but MACD is above signal line for >15 minutes: Cover and buy

4. **Safeguards**:
   - Warm-up period collects sufficient data before trading
   - Time-based throttling prevents excessive trading in choppy markets
   - Extended hours trading uses limit orders with adjusted prices based on spread

### Strategy Characteristics
- Trend-following: Works best in trending markets
- Full-cycle: Always in the market (either long or short)
- Momentum-based: Trades on price momentum rather than price levels
- Adaptive: Uses EMAs which give more weight to recent prices

## Configuration

### Environment Variables

Configure the system through environment variables in the `.env` file:

**Required:**
- `ALPACA_API_KEY`: Your Alpaca API key
- `ALPACA_API_SECRET`: Your Alpaca API secret
- `ALPACA_BASE_URL`: API endpoint URL
  - Paper trading: `https://paper-api.alpaca.markets`
  - Live trading: `https://api.alpaca.markets`

**Optional:**
- `RISK_PER_TRADE`: Maximum risk per trade as percentage (default: 2%)
- `MAX_POSITIONS`: Maximum concurrent positions (default: 5)
- `EXTENDED_HOURS`: Enable pre-market and after-hours trading (default: True)
- `OVERNIGHT_TRADING`: Enable overnight trading sessions (default: True)

### Trading Mode Configuration

The trading mode is controlled in the code (`main.py`):
```python
TRADING_MODE = "PAPER"  # Change to "LIVE" for live trading
```

### State Management

The system persists trading state in JSON files:
- Position files: `{SYMBOL}_position.json` (e.g., `NVDA_position.json`)
- Strategy states: Stored in `/state/` directories
- Configuration and logs: Separate files for tracking and debugging

## Risk Management

The system includes comprehensive built-in risk management:

**Position Management:**
- Dynamic position sizing based on account value and risk tolerance
- Maximum position limits to prevent over-exposure
- Automatic position transitions (long ↔ short) based on MACD signals
- Time-based throttling (15-minute minimum) to prevent overtrading

**Risk Controls:**
- Portfolio-level risk monitoring
- Per-trade risk limits via `RISK_PER_TRADE` setting
- Extended hours trading with adjusted limit orders
- Robust error handling and recovery mechanisms

**Market Hours Compliance:**
- Automatic market schedule detection
- Support for all trading sessions (regular, extended, overnight)
- Holiday and weekend trading restrictions

## Deployment Modes

### Paper Trading (Default)
```env
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```
Safe testing environment with simulated trading using real market data.

### Live Trading
```env
ALPACA_BASE_URL=https://api.alpaca.markets
```
**⚠️ Warning**: Real money trading. Ensure you:
1. Have completed Alpaca's live trading account setup
2. Understand the risks involved
3. Have thoroughly tested your strategies in paper mode
4. Set appropriate risk limits

## Dependencies

Key dependencies (see `requirements.txt` for complete list):

- **Trading & Data**: `alpaca-py>=0.8.0`, `yfinance>=0.2.0`
- **Analysis**: `pandas>=2.0.0`, `numpy>=1.24.0`, `scikit-learn==1.3.0`
- **Visualization**: `matplotlib>=3.7.0`
- **Infrastructure**: `python-dotenv>=1.0.0`, `websocket-client==1.6.1`
- **Testing**: `pytest==7.4.0`
- **Backtesting**: `backtrader==1.9.78.123`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Test thoroughly in paper trading mode
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## Support

For questions and support:
- Review the code documentation and comments
- Check the log files for debugging information
- Test in paper trading mode before live deployment

## Disclaimer

**⚠️ Important**: This software is for educational and informational purposes only. It is not financial advice. Trading stocks and options involves significant risk of financial loss. Past performance does not guarantee future results. Use at your own risk and only with funds you can afford to lose.

## License

[MIT License](LICENSE)