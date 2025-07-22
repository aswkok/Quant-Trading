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

### Enhanced MACD Strategy (Advanced)

**Enhanced MACD Trading with Slope and Histogram Analysis:**
```bash
# Standard Enhanced MACD strategy
python integrated_macd_trader.py --symbol NVDA --strategy enhanced_macd --warmup 30 --interval 60

# Long-only mode (avoids day trading buying power issues)
python integrated_macd_trader.py --symbol NVDA --strategy enhanced_macd --long-only --warmup 30 --interval 60

# Custom Enhanced MACD parameters
python integrated_macd_trader.py --symbol NVDA --strategy enhanced_macd \
   --slope-threshold 0.0015 --slope-lookback 5 --histogram-lookback 5 \
   --warmup 30 --interval 60 --long-only
```

### Complete Enhanced MACD Strategy Implementation

The Enhanced MACD Strategy implements sophisticated momentum analysis with precise entry/exit rules:

#### **🅰️ Case A: No Stock Holding (Flat Position)**
1. **Buy when**: MACD Line crosses above Signal Line (Bullish crossover)
2. **Short when**: MACD Line crosses below Signal Line (Bearish crossunder) - *Disabled in long-only mode*
3. **Buy on Momentum Strengthening** *(Long-only mode)*: Advanced entry detection
   - ✅ MACD Line is below Signal Line (still bearish overall)
   - ✅ MACD Slope is increasing or near zero (momentum recovering)
   - ✅ Absolute Histogram is compressing (bearish strength weakening)

#### **🅱️ Case B: Holding Long Position**
3. **Sell + Short when momentum weakens**:
   - ✅ MACD Line is above Signal Line
   - ✅ MACD Slope is decreasing or near zero (< slope_threshold)
   - ✅ MACD Histogram is smaller than the average of the last 3 values
4. **Exit Position (Failsafe)**: When MACD Line falls below Signal Line (Bearish crossunder)

#### **🅲️ Case C: Holding Short Position** *(Disabled in long-only mode)*
5. **Buy to Cover + Buy when momentum strengthens**:
   - ✅ MACD Line is below Signal Line
   - ✅ MACD Slope is increasing or near zero (> -slope_threshold)
   - ✅ Absolute Histogram is smaller than the average of the last 3 absolute values
6. **Exit Position (Failsafe)**: When MACD Line rises above Signal Line (Bullish crossover)

### Enhanced Strategy Parameters:
- `--slope-threshold`: MACD slope sensitivity (default: 0.001)
- `--slope-lookback`: Periods for slope calculation (default: 3)
- `--histogram-lookback`: Periods for histogram averaging (default: 3)
- `--long-only`: Enable long-only mode (no short selling) to avoid PDT restrictions

### Day Trading Buying Power Solutions

If you encounter `"insufficient day trading buying power"` errors:

#### **Solution 1: Use Long-Only Mode (Recommended)**
```bash
python integrated_macd_trader.py --symbol NVDA --strategy enhanced_macd --long-only --shares 100
```
- Disables all short selling that requires day trading buying power
- Maintains full Enhanced MACD analysis and momentum detection
- Works with accounts that have regular buying power but no PDT status

#### **Solution 2: Reduce Position Size**
```bash
python integrated_macd_trader.py --symbol NVDA --strategy enhanced_macd --shares 10
```
- Use smaller quantities to stay within buying power limits
- Gradually increase as your account grows

#### **Solution 3: Check Account Status**
The system now provides detailed diagnostics when PDT errors occur:
- Shows total vs. day trading buying power
- Identifies Pattern Day Trader status
- Suggests specific solutions based on your account

### Data Export and Analysis

**CSV Export with Enhanced MACD Analysis:**
```python
from enhanced_quote_monitor import EnhancedQuoteMonitor
from strategies import EnhancedMACDStrategy

# Export real-time quotes with Enhanced MACD data
monitor = EnhancedQuoteMonitor('NVDA')
monitor.save_enhanced_macd_csv()  # Exports comprehensive trading data

# Export strategy signals for backtesting  
strategy = EnhancedMACDStrategy(long_only=True)  # Long-only version
signals = strategy.generate_signals(historical_data)
strategy.save_signals_to_csv(signals, symbol='NVDA')
```

**Exported Data Includes:**
- Real-time quotes (bid, ask, mid, spread)
- MACD calculations (EMAfast, EMAslow, MACD, Signal, Histogram)
- Enhanced analysis (MACD_slope, Histogram_avg, momentum indicators)
- Trading signals (crossover, crossunder, position changes)
- Actions and trigger reasons (BUY, SELL, MOMENTUM_WEAKENING, FAILSAFE_EXIT, etc.)
- Strategy parameters and metadata with case indicators (🅰️🅱️🅲️)

### Customizing Strategies

Strategies can be customized by editing the `strategies.py` module or creating new strategy classes. The system supports multiple technical analysis approaches:

**Available Strategies:**
- `macd`: Classic MACD crossover strategy
- `enhanced_macd`: Advanced MACD with slope and histogram analysis
- `rsi`: RSI-based mean reversion strategy
- `bollinger_bands`: Bollinger Bands breakout strategy
- `moving_average_crossover`: Simple moving average strategy

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

The system implements both Classic and Enhanced MACD strategies with sophisticated momentum analysis:

### Core MACD Components
- **MACD Line**: The difference between a fast EMA (default: 13-period) and a slow EMA (default: 21-period)
- **Signal Line**: An EMA of the MACD line (default: 9-period)
- **Histogram**: The difference between the MACD line and signal line

### Classic MACD Strategy

**Trading Rules:**
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

## Enhanced MACD Strategy Implementation

The Enhanced MACD Strategy provides sophisticated momentum analysis with failsafe exit conditions:

### Strategy Architecture

**Core Philosophy**: Combines traditional MACD crossover signals with advanced momentum detection to reduce whipsaws and improve entry/exit timing.

### Case-Based Trading Logic

#### **🅰️ Case A: No Stock Holding**
- **A.1 Buy**: MACD crosses above Signal Line → Enter long position
- **A.2 Short**: MACD crosses below Signal Line → Enter short position *(disabled in long-only mode)*

#### **🅱️ Case B: Holding Long Position** 
- **B.3 Sell + Short**: Advanced momentum weakening detection
  - MACD above Signal Line (still bullish overall)
  - MACD slope decreasing (momentum slowing)
  - Histogram below recent average (weakening strength)
- **B.4 Failsafe Exit**: MACD crosses below Signal Line → Guaranteed exit

#### **🅲️ Case C: Holding Short Position** *(Long-only mode: disabled)*
- **C.5 Cover + Buy**: Advanced momentum strengthening detection
  - MACD below Signal Line (still bearish overall)
  - MACD slope increasing (momentum recovering)
  - Histogram compression (weakening bearish strength)
- **C.6 Failsafe Exit**: MACD crosses above Signal Line → Guaranteed exit

### Long-Only Mode Features

When `--long-only` is enabled:
- **No short selling**: Eliminates day trading buying power requirements
- **Enhanced entry detection**: Three ways to enter long positions
- **Case A.1**: BUY on bullish crossover (traditional)
- **Case A.2**: STAY_FLAT instead of SHORT (no bearish entries)
- **🆕 Case A.3**: BUY on momentum strengthening (advanced entry)
- **Case B.3**: SELL instead of SELL_AND_SHORT (exit only)
- **Case B.4**: SELL instead of SELL_AND_SHORT (failsafe exit)
- **Case C**: Completely disabled (no short positions to manage)

#### **Advanced Long-Only Entry Logic**
The long-only mode captures **momentum strengthening signals** that would trigger short position exits (Case C.5) and converts them into long entry opportunities:

```
Market Scenario: Bearish trend with emerging strength
Time    MACD    Signal   Slope      Histogram    Action
10:00   -0.08   -0.05    -0.002     -0.03       STAY_FLAT (bearish)
10:15   -0.06   -0.04    -0.001     -0.025      STAY_FLAT (improving)
10:30   -0.05   -0.03    +0.0005    -0.02       🅰️ BUY (Case A.3)
10:45   -0.02   -0.01    +0.001     -0.01       HOLD LONG
11:00   +0.01   +0.02    +0.002     +0.005      HOLD LONG (now bullish)
```

**Result**: Enters long position **before** traditional bullish crossover, capturing more upward movement.

### Technical Indicators

**MACD Slope Calculation:**
```python
slope = (macd_recent[-1] - macd_recent[0]) / (lookback_period - 1)
momentum_weakening = slope < threshold and histogram < histogram_average
```

**Histogram Analysis:**
- Rolling average over configurable periods
- Absolute value analysis for short position momentum
- Compression detection for trend reversal signals

### Strategy Characteristics

#### **Standard Enhanced MACD:**
- **Trend-following**: Works best in trending markets
- **Full-cycle**: Always in the market (either long or short)
- **Momentum-based**: Trades on price momentum rather than price levels
- **Adaptive**: Uses EMAs which give more weight to recent prices
- **Enhanced precision**: Early momentum detection reduces whipsaws

#### **Long-Only Enhanced MACD:**
- **Selective entries**: Three sophisticated entry mechanisms
- **Risk-controlled**: Only long positions, no short squeeze risk
- **Momentum-optimized**: Captures early momentum shifts in both directions
- **PDT-compliant**: No day trading buying power requirements
- **Bull-market focused**: Maximizes upward moves, sits out downtrends

### Safeguards
- Warm-up period collects sufficient data before trading
- Time-based throttling prevents excessive trading in choppy markets
- Extended hours trading uses limit orders with adjusted prices based on spread
- Comprehensive logging and state persistence for robust operation

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