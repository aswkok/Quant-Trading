# Alpaca Quantitative Trading System

A comprehensive quantitative trading system for stocks and options using the Alpaca API.

## Overview

This system provides a framework for algorithmic trading of stocks and options using the Alpaca API. It includes:

- Stock trading with various technical analysis strategies
- Real-time MACD-based trading with live market data
- Options trading strategies (covered calls, cash-secured puts, iron condors)
- Risk management and position sizing
- Customizable strategy parameters
- Logging and performance tracking

## Getting Started

### Prerequisites

- Python 3.8+
- Alpaca API account (sign up at [Alpaca](https://alpaca.markets/))
- API key and secret from Alpaca

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd Stock
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Copy the example environment file and update it with your Alpaca API credentials:
   ```
   cp .env.example .env
   ```
   Then edit the `.env` file with your actual API credentials.

## Usage

### Basic Stock Trading

Run the main trading script:

```
python main.py
```

This will execute the default moving average crossover strategy on a set of default stocks (AAPL, MSFT, AMZN, GOOGL, META).

### Integrated MACD Trading System

Run the integrated MACD trading system that connects real-time quotes with trading execution:

```
python integrated_macd_trader.py --symbol NVDA --warmup 30 --interval 60 --shares 100
```

This system combines real-time quote monitoring, MACD calculation, and trade execution into a single workflow. Options:
- `--symbol`: Stock symbol to trade (default: NVDA)
- `--interval`: Seconds between quote fetches (default: 60)
- `--shares`: Number of shares per trade (default: 100)
- `--fast-window`: Fast EMA window for MACD (default: 13)
- `--slow-window`: Slow EMA window for MACD (default: 21)
- `--signal-window`: Signal line window for MACD (default: 9)
- `--extended-hours`: Enable trading during pre-market and after-hours
- `--warmup`: Data collection period before trading in minutes (default: 60)

### Enhanced Quote Monitoring

Monitor real-time bid and ask prices with improved timestamp handling:

```
python enhanced_quote_monitor.py --symbol NVDA --interval 60
```

This enhanced version ensures proper timestamp management and reliable MACD calculations. Options:
- `--symbol`: Stock symbol to monitor (default: NVDA)
- `--interval`: Seconds between quote fetches (default: 60)
- `--max-records`: Maximum number of records to keep (default: 200)

### Options Trading

Run the options trading script:

```
python options_trading.py
```

By default, this will execute a covered call strategy on the same set of default stocks.

### Real-Time Quote Monitoring (Original Version)

Monitor real-time bid and ask prices for a specific stock:

```
python quote_monitor.py --symbol NVDA --interval 60
```

This will fetch quotes every 60 seconds and display them in a formatted table. Options:
- `--symbol`: Stock symbol to monitor (default: AAPL)
- `--interval`: Seconds between quote fetches (default: 60)
- `--max-records`: Maximum number of records to keep (default: 100)

### Real-Time MACD Trading (Original Version)

Execute MACD-based trades using real-time quote data:

```
python realtime_macd_trader.py --symbol NVDA --interval 60
```

This will fetch quotes, calculate MACD indicators, and execute trades based on MACD signals. Options:
- `--symbol`: Stock symbol to trade (default: NVDA)
- `--interval`: Seconds between quote fetches (default: 60)
- `--shares`: Number of shares per trade (default: 100)
- `--fast-window`: Fast EMA window (default: 13)
- `--slow-window`: Slow EMA window (default: 21)
- `--signal-window`: Signal line window (default: 9)

### Customizing Strategies

You can modify the strategies used by editing the scripts or by creating your own strategy classes in the `strategies.py` module.

## System Components

### main.py

The main entry point for stock trading. It initializes the trading system, connects to the Alpaca API, and executes trading strategies.

### integrated_macd_trader.py

A complete end-to-end trading system that integrates real-time quote monitoring, MACD calculation, and trade execution:
- Fetches real-time bid/ask quotes from Alpaca API
- Calculates MACD indicators on the fly
- Executes trades based on MACD crossovers and position changes
- Includes a warm-up period for gathering sufficient data
- Manages trade state and position transitions
- Supports extended hours trading
- Prevents over-trading with time-based throttling

### enhanced_quote_monitor.py

An improved version of the quote monitor with better timestamp handling:
- Ensures each quote has a unique timestamp
- Properly tracks real-time data
- Improved MACD calculation stability
- Better display of MACD signals and position data
- More robust error handling

### options_trading.py

Specialized module for options trading strategies including covered calls, cash-secured puts, and iron condors.

### strategies.py

Contains various trading strategy implementations:
- Moving Average Crossover
- RSI (Relative Strength Index)
- Bollinger Bands
- MACD (Moving Average Convergence Divergence)
- Enhanced MACD with real-time quote support

### quote_monitor.py

A standalone tool for monitoring real-time bid and ask prices for specified stocks:
- Fetches real-time quotes from Alpaca API
- Maintains a rolling window of the most recent quotes (default: 100)
- Calculates and displays bid-ask spreads
- Computes MACD indicators in real-time based on mid-prices
- Saves quote data to CSV for later analysis

### realtime_macd_trader.py

An integrated real-time trading system that combines quote monitoring with MACD-based trading:
- Uses live bid/ask data instead of historical OHLC data
- Calculates MACD indicators in real-time
- Executes trades based on MACD crossovers and position changes
- Supports extended hours and overnight trading
- Adapts limit prices based on current bid-ask spreads

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

The system is configured through environment variables in the `.env` file:

- `ALPACA_API_KEY`: Your Alpaca API key
- `ALPACA_API_SECRET`: Your Alpaca API secret
- `ALPACA_BASE_URL`: API endpoint URL (paper or live trading)
- `RISK_PER_TRADE`: Maximum risk per trade as a percentage of portfolio (default: 2%)
- `MAX_POSITIONS`: Maximum number of concurrent positions (default: 5)
- `EXTENDED_HOURS`: Enable trading during pre-market (4:00 AM - 9:30 AM ET) and after-hours (4:00 PM - 8:00 PM ET) sessions (default: True)
- `OVERNIGHT_TRADING`: Enable trading during overnight sessions (8:00 PM - 4:00 AM ET) (default: True)

## Risk Management

The system includes built-in risk management features:
- Position sizing based on account value and risk tolerance
- Maximum position limits
- Stop-loss calculations
- Time-based trade throttling to prevent overtrading

## Paper vs. Live Trading

By default, the system uses Alpaca's paper trading environment. To switch to live trading:

1. Update the `ALPACA_BASE_URL` in your `.env` file to `https://api.alpaca.markets`
2. Ensure you have completed Alpaca's account setup for live trading

## Disclaimer

This software is for educational and informational purposes only. It is not financial advice. Trading stocks and options involves significant risk of loss. Use at your own risk.

## License

[MIT License](LICENSE)