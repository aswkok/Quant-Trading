#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Quote Monitor for MACD Trading System

This is an enhanced version of the quote monitor that uses WebSockets for real-time
data streaming from Alpaca, providing more reliable and up-to-date quote information.
"""

import os
import time
import json
import logging
import argparse
import threading
import pandas as pd
import numpy as np
import websocket
from datetime import datetime, timezone
from dotenv import load_dotenv
from tabulate import tabulate
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedQuoteMonitor:
    """
    Enhanced monitor for real-time bid and ask prices using WebSocket streaming.
    
    This class connects to Alpaca's WebSocket API to receive real-time quotes,
    calculates MACD indicators, and provides trading signals.
    """
    
    def __init__(self, symbol, max_records=200, interval_seconds=60, fast_window=13, slow_window=21, signal_window=9):
        """
        Initialize the quote monitor with WebSocket support.
        
        Args:
            symbol: Stock symbol to monitor
            max_records: Maximum number of records to keep in memory
            interval_seconds: Interval between display updates (WebSocket data comes in real-time)
            fast_window: Window for the fast EMA in MACD calculation
            slow_window: Window for the slow EMA in MACD calculation
            signal_window: Window for the signal line in MACD calculation
        """
        # Load environment variables
        load_dotenv()
        
        # API credentials
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API credentials not found. Please set ALPACA_API_KEY and ALPACA_API_SECRET in .env file.")
        
        # Initialize REST clients (still useful for account info, etc.)
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        
        # Monitor settings
        self.symbol = symbol.upper()  # Ensure uppercase for Alpaca API
        self.max_records = max_records
        self.interval_seconds = interval_seconds
        
        # Data storage
        self.quotes_df = pd.DataFrame(columns=['timestamp', 'bid', 'ask', 'spread', 'spread_pct', 'mid'])
        
        # MACD parameters
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.signal_window = signal_window
        
        # MACD state tracking
        self.last_macd_position = None  # 'ABOVE' or 'BELOW'
        self.last_signal_time = None
        
        # WebSocket connection
        self.ws = None
        self.ws_thread = None
        self.connected = False
        self.last_quote = None
        
        # WebSocket URLs
        self.ws_url = "wss://stream.data.alpaca.markets/v2/iex"
        
        # Initialize WebSocket connection
        self._init_websocket()
        
        logger.info(f"Enhanced Quote Monitor initialized for {symbol} using WebSocket streaming")
        logger.info(f"API Key found, starts with: {self.api_key[:4]}...")
        logger.info(f"Display update interval: {interval_seconds} seconds")
        logger.info(f"Maximum records: {max_records}")
        logger.info(f"MACD Parameters: Fast={fast_window}, Slow={slow_window}, Signal={signal_window}")
    
    def _init_websocket(self):
        """
        Initialize the WebSocket connection to Alpaca's streaming API.
        """
        # Define WebSocket callbacks
        def on_open(ws):
            self.connected = True
            logger.info("WebSocket connection opened")
            # Authentication message
            auth_message = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret
            }
            ws.send(json.dumps(auth_message))
            
            # Subscribe to quotes for our symbol
            subscribe_message = {
                "action": "subscribe",
                "quotes": [self.symbol]
            }
            ws.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to quotes for {self.symbol}")
        
        def on_message(ws, message):
            data = json.loads(message)
            
            # Handle different message types
            if isinstance(data, list):
                for msg in data:
                    self._process_message(msg)
            else:
                self._process_message(data)
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            self.connected = False
            # Schedule reconnection
            self._schedule_reconnect()
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
            self.connected = False
            # Schedule reconnection
            self._schedule_reconnect()
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(self.ws_url,
                                        on_open=on_open,
                                        on_message=on_message,
                                        on_error=on_error,
                                        on_close=on_close)
        
        # Start WebSocket in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True  # Thread will exit when main program exits
        self.ws_thread.start()
    
    def _schedule_reconnect(self):
        """
        Schedule a reconnection attempt after a short delay.
        """
        if not self.connected:
            logger.info("Scheduling WebSocket reconnection in 5 seconds...")
            reconnect_thread = threading.Thread(target=self._delayed_reconnect)
            reconnect_thread.daemon = True
            reconnect_thread.start()
    
    def _delayed_reconnect(self):
        """
        Wait a few seconds and then attempt to reconnect the WebSocket.
        """
        time.sleep(5)  # Wait 5 seconds before reconnecting
        logger.info("Attempting to reconnect WebSocket...")
        
        # Close existing connection if it exists
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        
        # Initialize a new connection
        self._init_websocket()
    
    def _process_message(self, msg):
        """
        Process incoming WebSocket messages.
        
        Args:
            msg: Message from WebSocket
        """
        try:
            # Check if it's a quote message
            if msg.get('T') == 'q' and msg.get('S') == self.symbol:
                # Extract quote data with proper type conversion
                try:
                    # Handle different timestamp formats from Alpaca
                    t_value = msg.get('t', 0)
                    if isinstance(t_value, str):
                        # ISO format: '2025-07-22T12:22:26.802804135Z'
                        timestamp = datetime.fromisoformat(t_value.replace('Z', '+00:00'))
                    else:
                        # Nanosecond format: 1640995200000000000
                        timestamp = datetime.fromtimestamp(float(t_value) / 1e9, tz=timezone.utc)
                    
                    bid_price = float(msg.get('bp', 0))
                    ask_price = float(msg.get('ap', 0))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid quote data types - bp: {msg.get('bp')}, ap: {msg.get('ap')}, t: {msg.get('t')} - Error: {e}")
                    return
                
                # Validate prices are positive numbers
                if bid_price <= 0 or ask_price <= 0:
                    logger.debug(f"Invalid price data - bid: {bid_price}, ask: {ask_price}")
                    return
                
                # Calculate spread
                spread = ask_price - bid_price
                spread_pct = (spread / bid_price) * 100 if bid_price > 0 else 0
                
                # Store as the latest quote
                self.last_quote = (timestamp, bid_price, ask_price, spread, spread_pct)
                
                # Add to dataframe
                self.add_quote_to_dataframe(self.last_quote)
                
                # Log at debug level to avoid flooding logs
                logger.debug(f"Quote received: {self.symbol} - Bid: {bid_price}, Ask: {ask_price}")
            
            # Handle authentication response
            elif msg.get('T') == 'success' and msg.get('msg') == 'authenticated':
                logger.info("Successfully authenticated with Alpaca WebSocket API")
            
            # Handle subscription response
            elif msg.get('T') == 'subscription':
                logger.info(f"Subscription status: {msg}")
        
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
            logger.debug(f"Problematic message: {msg}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
    
    def get_latest_quote(self):
        """
        Get the latest quote from WebSocket data.
        
        Returns:
            tuple: (timestamp, bid_price, ask_price, spread, spread_percentage)
        """
        return self.last_quote
    
    def add_quote_to_dataframe(self, quote_data):
        """
        Add a quote to the dataframe and maintain max_records limit.
        
        Args:
            quote_data: Tuple of (timestamp, bid_price, ask_price, spread, spread_pct)
        """
        if quote_data is None:
            return
        
        timestamp, bid_price, ask_price, spread, spread_pct = quote_data
        
        # Calculate mid price (average of bid and ask)
        try:
            mid_price = (float(bid_price) + float(ask_price)) / 2
        except (ValueError, TypeError) as e:
            logger.warning(f"Error calculating mid price with bid: {bid_price}, ask: {ask_price} - Error: {e}")
            return
        
        # Create a new row
        new_row = pd.DataFrame({
            'timestamp': [timestamp],
            'bid': [bid_price],
            'ask': [ask_price],
            'spread': [spread],
            'spread_pct': [spread_pct],
            'mid': [mid_price]
        })
        
        # Log the new quote for debugging
        logger.debug(f"Adding new quote: {timestamp}, bid=${bid_price:.2f}, ask=${ask_price:.2f}, mid=${mid_price:.2f}")
        
        # Append to dataframe - use proper concatenation for future compatibility
        if self.quotes_df.empty:
            # Initialize DataFrame with proper column types
            self.quotes_df = new_row.copy()
        else:
            self.quotes_df = pd.concat([self.quotes_df, new_row], ignore_index=True)
        
        # Trim to max_records
        if len(self.quotes_df) > self.max_records:
            self.quotes_df = self.quotes_df.iloc[-self.max_records:]
            
        # Calculate MACD if we have enough data
        self.calculate_macd()
    
    def calculate_macd(self):
        """
        Calculate MACD based on the mid-prices in the quotes dataframe.
        
        This method adds the following columns to the dataframe:
        - EMAfast: Fast EMA of mid prices
        - EMAslow: Slow EMA of mid prices
        - MACD: MACD line (EMAfast - EMAslow)
        - Signal: Signal line (EMA of MACD)
        - Histogram: MACD - Signal
        - MACD_position: 'ABOVE' or 'BELOW' indicating MACD position relative to signal line
        - crossover: True when MACD crosses above signal line
        - crossunder: True when MACD crosses below signal line
        """
        # Check if we have enough data
        warmup_period = max(self.slow_window * 3, self.fast_window * 3) + self.signal_window
        if len(self.quotes_df) < warmup_period:
            logger.info(f"Not enough data for reliable MACD calculation. Have {len(self.quotes_df)} records, need {warmup_period}.")
            return
        
        # Calculate MACD components using standard method with mid prices
        self.quotes_df['EMAfast'] = self.quotes_df['mid'].ewm(span=self.fast_window, adjust=False).mean()
        self.quotes_df['EMAslow'] = self.quotes_df['mid'].ewm(span=self.slow_window, adjust=False).mean()
        self.quotes_df['MACD'] = self.quotes_df['EMAfast'] - self.quotes_df['EMAslow']
        self.quotes_df['Signal'] = self.quotes_df['MACD'].ewm(span=self.signal_window, adjust=False).mean()
        self.quotes_df['Histogram'] = self.quotes_df['MACD'] - self.quotes_df['Signal']
        
        # Determine MACD position (above or below signal line)
        self.quotes_df['MACD_position'] = np.where(self.quotes_df['MACD'] > self.quotes_df['Signal'], 'ABOVE', 'BELOW')
        
        # Initialize crossover/crossunder columns
        self.quotes_df['crossover'] = False
        self.quotes_df['crossunder'] = False
        
        # Need at least 2 rows to detect crossovers
        if len(self.quotes_df) >= 2:
            # Get the current and previous positions
            current_position = self.quotes_df.iloc[-1]['MACD_position']
            previous_position = self.quotes_df.iloc[-2]['MACD_position']
            
            # Detect crossover (MACD crosses above signal)
            if current_position == 'ABOVE' and previous_position == 'BELOW':
                self.quotes_df.iloc[-1, self.quotes_df.columns.get_loc('crossover')] = True
                logger.debug(f"MACD Crossover detected (BULLISH) at {self.quotes_df.iloc[-1]['timestamp']}")
            
            # Detect crossunder (MACD crosses below signal)
            elif current_position == 'BELOW' and previous_position == 'ABOVE':
                self.quotes_df.iloc[-1, self.quotes_df.columns.get_loc('crossunder')] = True
                logger.debug(f"MACD Crossunder detected (BEARISH) at {self.quotes_df.iloc[-1]['timestamp']}")
        
        # Update the last MACD position
        if len(self.quotes_df) > 0:
            latest = self.quotes_df.iloc[-1]
            self.last_macd_position = latest['MACD_position']
            
            # Log the current MACD position
            logger.debug(f"Current MACD: {latest['MACD']:.6f}, Signal: {latest['Signal']:.6f}, Position: {self.last_macd_position}")
    
    def get_macd_signal(self):
        """
        Get the current MACD trading signal based on the latest data.
        
        Returns:
            dict: A dictionary containing signal information:
                - signal: 1.0 for buy, -1.0 for sell, 0.0 for hold
                - position: Current position (1.0 for long, -1.0 for short, 0.0 for none)
                - macd_position: 'ABOVE' or 'BELOW'
                - crossover: True if MACD just crossed above signal line
                - crossunder: True if MACD just crossed below signal line
                - macd_value: Current MACD value
                - signal_value: Current signal line value
                - histogram: Current histogram value
        """
        if len(self.quotes_df) == 0 or 'MACD' not in self.quotes_df.columns:
            return {
                'signal': 0.0,
                'position': 0.0,
                'macd_position': None,
                'crossover': False,
                'crossunder': False,
                'macd_value': None,
                'signal_value': None,
                'histogram': None
            }
        
        # Get the latest values
        latest = self.quotes_df.iloc[-1]
        
        # Check for crossover/crossunder
        crossover = latest['crossover'] if 'crossover' in self.quotes_df.columns else False
        crossunder = latest['crossunder'] if 'crossunder' in self.quotes_df.columns else False
        
        # Determine signal
        signal = 0.0
        if crossover:
            signal = 1.0
            self.last_signal_time = latest['timestamp']
        elif crossunder:
            signal = -1.0
            self.last_signal_time = latest['timestamp']
        
        # Create result dictionary
        result = {
            'signal': signal,
            'position': 1.0 if latest['MACD_position'] == 'ABOVE' else -1.0 if latest['MACD_position'] == 'BELOW' else 0.0,
            'macd_position': latest['MACD_position'],
            'crossover': crossover,
            'crossunder': crossunder,
            'macd_value': latest['MACD'],
            'signal_value': latest['Signal'],
            'histogram': latest['Histogram'],
            'timestamp': latest['timestamp'],
            'mid_price': latest['mid']
        }
        
        # Log the signal if it's a buy or sell
        if signal != 0.0:
            action = "BUY" if signal > 0 else "SELL"
            logger.info(f"MACD Signal: {action} at ${latest['mid']:.2f}")
            logger.info(f"MACD: {latest['MACD']:.6f}, Signal: {latest['Signal']:.6f}, Histogram: {latest['Histogram']:.6f}")
        
        return result
        
    def display_quotes(self):
        """
        Display the latest quotes and MACD information in a formatted table.
        """
        if len(self.quotes_df) == 0:
            # Check if market is likely open
            now = datetime.now(timezone.utc)
            is_weekday = now.weekday() < 5  # Monday to Friday
            hour_et = (now.hour - 4) % 24  # Convert UTC to ET (approximation)
            
            if not is_weekday:
                logger.info("No quotes available - Market is closed (weekend).")
            elif hour_et < 4 or hour_et >= 20:  # Before 4 AM ET or after 8 PM ET
                logger.info("No quotes available - Market is closed (outside of extended hours).")
            elif hour_et < 9 or hour_et >= 16:  # Before 9:30 AM ET or after 4 PM ET
                logger.info("No quotes available - Regular market hours are closed, but extended hours may be active.")
            else:
                logger.info("No quotes available yet - WebSocket is connected but no data received.")
                
            # Print connection status
            logger.info(f"WebSocket connection status: {'Connected' if self.connected else 'Disconnected'}")
            return
        
        # Get the latest quotes
        latest_quotes = self.quotes_df.tail(10).copy()
        
        # Format the timestamp
        latest_quotes['time'] = latest_quotes['timestamp'].dt.strftime('%H:%M:%S')
        
        # Format the prices
        latest_quotes['bid'] = latest_quotes['bid'].map('${:.2f}'.format)
        latest_quotes['ask'] = latest_quotes['ask'].map('${:.2f}'.format)
        latest_quotes['mid'] = latest_quotes['mid'].map('${:.2f}'.format)
        latest_quotes['spread'] = latest_quotes['spread'].map('${:.4f}'.format)
        latest_quotes['spread_pct'] = latest_quotes['spread_pct'].map('{:.2f}%'.format)
        
        # Format MACD values if they exist
        if 'MACD' in latest_quotes.columns:
            latest_quotes['MACD'] = latest_quotes['MACD'].map('{:.6f}'.format)
            latest_quotes['Signal'] = latest_quotes['Signal'].map('{:.6f}'.format)
            latest_quotes['Histogram'] = latest_quotes['Histogram'].map('{:.6f}'.format)
            
            # Create a position indicator
            latest_quotes['Position'] = latest_quotes['MACD_position']
            
            # Add Enhanced MACD columns using the same strategy calculations used for trading
            enhanced_columns = ['MACD_slope', 'Histogram_avg', 'signal', 'action', 'trigger_reason']
            if any(col in self.quotes_df.columns for col in enhanced_columns):
                # Use existing Enhanced MACD data from strategy
                if 'MACD_slope' in self.quotes_df.columns:
                    latest_quotes['Slope'] = latest_quotes.get('MACD_slope', pd.Series()).apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
                if 'Histogram_avg' in self.quotes_df.columns:
                    latest_quotes['HistAvg'] = latest_quotes.get('Histogram_avg', pd.Series()).apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
                
                # Add momentum and action based on strategy calculations
                latest_quotes['Momentum'] = latest_quotes.apply(lambda row:
                    'WEAK' if row.get('trigger_reason') == 'MOMENTUM_WEAKENING' else
                    'STRONG' if row.get('trigger_reason') == 'MOMENTUM_STRENGTHENING' else
                    'BULLISH' if row.get('trigger_reason') == 'MACD_CROSSOVER' else
                    'BEARISH' if row.get('trigger_reason') == 'MACD_CROSSUNDER' else
                    'NEUTRAL', axis=1)
                
                # Create enhanced signal indicator using actual strategy actions with case indicators
                latest_quotes['Signal_Indicator'] = latest_quotes.apply(lambda row: 
                    '🅰️ BUY' if row.get('action') == 'BUY' and 'CROSSOVER' in str(row.get('trigger_reason', '')) else
                    '🅰️ BUY-MOMENTUM' if row.get('action') == 'BUY' and 'MOMENTUM_STRENGTHENING_LONG_ONLY' in str(row.get('trigger_reason', '')) else
                    '🅰️ SHORT' if row.get('action') == 'SHORT' and 'CROSSUNDER' in str(row.get('trigger_reason', '')) else
                    '🅱️ SELL+SHORT' if row.get('action') == 'SELL_AND_SHORT' and 'MOMENTUM_WEAKENING' in str(row.get('trigger_reason', '')) else
                    '🅱️ FAILSAFE-EXIT' if row.get('action') == 'SELL_AND_SHORT' and 'FAILSAFE_CROSSUNDER' in str(row.get('trigger_reason', '')) else
                    '🅲️ COVER+BUY' if row.get('action') == 'COVER_AND_BUY' and 'MOMENTUM_STRENGTHENING' in str(row.get('trigger_reason', '')) else
                    '🅲️ FAILSAFE-EXIT' if row.get('action') == 'COVER_AND_BUY' and 'FAILSAFE_CROSSOVER' in str(row.get('trigger_reason', '')) else
                    '🚀 BUY' if row.get('action') == 'BUY' else
                    '📉 SHORT' if row.get('action') == 'SHORT' else
                    '🔄 SELL+SHORT' if row.get('action') == 'SELL_AND_SHORT' else
                    '🔄 COVER+BUY' if row.get('action') == 'COVER_AND_BUY' else
                    '⚡ WEAK' if row.get('trigger_reason') == 'MOMENTUM_WEAKENING' else
                    '⚡ STRONG' if row.get('trigger_reason') == 'MOMENTUM_STRENGTHENING' else
                    '🚀 BUY' if row.get('crossover', False) else 
                    '📉 SELL' if row.get('crossunder', False) else 
                    '➖ HOLD', axis=1)
                
                # Display the table with Enhanced MACD information
                display_columns = ['time', 'bid', 'ask', 'mid', 'MACD', 'Signal', 'Histogram', 'Slope', 'HistAvg', 'Momentum', 'Signal_Indicator']
                available_columns = [col for col in display_columns if col in latest_quotes.columns]
                
                print(tabulate(
                    latest_quotes[available_columns].iloc[::-1],
                    headers='keys',
                    tablefmt='pretty',
                    showindex=False
                ))
            elif len(latest_quotes) >= 3:
                # Fallback: Use basic Enhanced MACD calculations for display only
                # (This path is less preferred as it may not match trading decisions)
                # Calculate MACD slope for display
                latest_quotes['MACD_slope'] = self._calculate_display_macd_slope(latest_quotes)
                latest_quotes['Slope'] = latest_quotes['MACD_slope'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
                
                # Calculate histogram averages for display
                latest_quotes['Hist_avg'] = self._calculate_display_histogram_avg(latest_quotes)
                latest_quotes['HistAvg'] = latest_quotes['Hist_avg'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
                
                # Add momentum analysis signals
                latest_quotes['Momentum'] = self._calculate_display_momentum(latest_quotes)
                
                # Create enhanced signal indicator
                latest_quotes['Signal_Indicator'] = latest_quotes.apply(lambda row: 
                    '🚀 BUY' if row.get('crossover', False) else 
                    '📉 SELL' if row.get('crossunder', False) else 
                    '⚡ WEAK' if row['Momentum'] == 'WEAK' else
                    '⚡ STRONG' if row['Momentum'] == 'STRONG' else
                    '➖ HOLD', axis=1)
                
                # Display the table with Enhanced MACD information
                display_columns = ['time', 'bid', 'ask', 'mid', 'MACD', 'Signal', 'Histogram', 'Slope', 'HistAvg', 'Momentum', 'Signal_Indicator']
                available_columns = [col for col in display_columns if col in latest_quotes.columns]
                
                print(tabulate(
                    latest_quotes[available_columns].iloc[::-1],
                    headers='keys',
                    tablefmt='pretty',
                    showindex=False
                ))
            else:
                # Create a basic signal indicator
                latest_quotes['Signal_Indicator'] = ''
                latest_quotes.loc[latest_quotes['crossover'] == True, 'Signal_Indicator'] = '↑ BUY'
                latest_quotes.loc[latest_quotes['crossunder'] == True, 'Signal_Indicator'] = '↓ SELL'
                
                # Display the table with basic MACD information
                print(tabulate(
                    latest_quotes[['time', 'bid', 'ask', 'mid', 'MACD', 'Signal', 'Histogram', 'Position', 'Signal_Indicator']].iloc[::-1],
                    headers='keys',
                    tablefmt='pretty',
                    showindex=False
                ))
            
            # Print connection status
            print(f"\nWebSocket connection status: {'Connected' if self.connected else 'Disconnected'}")
            print(f"Total quotes collected: {len(self.quotes_df)}")
            
            # Get the latest MACD signal
            macd_signal = self.get_macd_signal()
            
            # Display current signal
            if macd_signal['macd_position'] is not None:
                signal_str = "BUY" if macd_signal['signal'] > 0 else "SELL" if macd_signal['signal'] < 0 else "HOLD"
                position_str = "LONG" if macd_signal['position'] > 0 else "SHORT" if macd_signal['position'] < 0 else "NONE"
                
                print(f"\nCurrent MACD Signal: {signal_str}")
                print(f"Current Position: {position_str}")
                print(f"MACD Position: {macd_signal['macd_position']} signal line")
                if macd_signal['crossover']:
                    print(f"BULLISH SIGNAL: MACD just crossed ABOVE signal line!")
                if macd_signal['crossunder']:
                    print(f"BEARISH SIGNAL: MACD just crossed BELOW signal line!")
        else:
            # Display the table without MACD information
            print(tabulate(
                latest_quotes[['time', 'bid', 'ask', 'mid', 'spread', 'spread_pct']].iloc[::-1],
                headers='keys',
                tablefmt='pretty',
                showindex=False
            ))
            
            # Print connection status
            print(f"\nWebSocket connection status: {'Connected' if self.connected else 'Disconnected'}")
            print(f"Total quotes collected: {len(self.quotes_df)}")
            print("MACD calculation pending - waiting for more data...")
        
        print("=" * 80 + "\n")
    
    def save_to_csv(self, filename=None, include_enhanced_macd=False):
        """
        Save the current quotes to a CSV file with optional Enhanced MACD strategy data.
        
        Args:
            filename: Optional filename, defaults to symbol_quotes_YYYYMMDD.csv
            include_enhanced_macd: If True, includes Enhanced MACD strategy calculations
        """
        if self.quotes_df.empty:
            logger.info("No quotes to save.")
            return
        
        if filename is None:
            today = datetime.now().strftime('%Y%m%d')
            suffix = "_enhanced_macd" if include_enhanced_macd else "_quotes"
            filename = f"{self.symbol}{suffix}_{today}.csv"
        
        try:
            df_to_save = self.quotes_df.copy()
            
            # If Enhanced MACD data is requested and we have the necessary columns
            if include_enhanced_macd:
                logger.info("Adding Enhanced MACD strategy calculations to CSV export")
                
                # Import and apply Enhanced MACD strategy if not already applied
                if not all(col in df_to_save.columns for col in ['MACD_slope', 'Histogram_avg', 'trigger_reason']):
                    logger.info("Calculating Enhanced MACD strategy data for CSV export")
                    
                    # Import here to avoid circular imports
                    from strategies import EnhancedMACDStrategy
                    
                    # Create a temporary Enhanced MACD strategy
                    temp_strategy = EnhancedMACDStrategy(
                        fast_window=self.fast_window,
                        slow_window=self.slow_window,
                        signal_window=self.signal_window
                    )
                    
                    # Convert quotes_df to the format expected by strategy
                    strategy_data = df_to_save.copy()
                    strategy_data['close'] = df_to_save['mid']  # Use mid price as close
                    strategy_data['open'] = df_to_save['mid']
                    strategy_data['high'] = df_to_save['ask']
                    strategy_data['low'] = df_to_save['bid'] 
                    strategy_data['volume'] = 100000  # Placeholder volume
                    
                    # Generate Enhanced MACD signals
                    enhanced_signals = temp_strategy.generate_signals(strategy_data)
                    
                    # Add Enhanced MACD columns to the dataframe
                    enhanced_columns = [
                        'MACD_slope', 'Histogram_avg', 'Histogram_abs_avg',
                        'signal', 'position', 'position_type', 'shares', 
                        'action', 'trigger_reason'
                    ]
                    
                    for col in enhanced_columns:
                        if col in enhanced_signals.columns:
                            df_to_save[col] = enhanced_signals[col]
                
                # Reorder columns for better readability
                column_order = [
                    'timestamp', 'bid', 'ask', 'mid', 'spread', 'spread_pct',
                    'EMAfast', 'EMAslow', 'MACD', 'Signal', 'Histogram',
                    'MACD_slope', 'Histogram_avg', 'Histogram_abs_avg',
                    'MACD_position', 'crossover', 'crossunder',
                    'signal', 'position', 'position_type', 'shares',
                    'action', 'trigger_reason'
                ]
                
                # Only include columns that exist in the dataframe
                available_columns = [col for col in column_order if col in df_to_save.columns]
                remaining_columns = [col for col in df_to_save.columns if col not in available_columns]
                
                df_to_save = df_to_save[available_columns + remaining_columns]
            
            # Save to CSV
            df_to_save.to_csv(filename, index=False)
            
            # Log summary of saved data
            logger.info(f"Quotes saved to {filename}")
            logger.info(f"Total records: {len(df_to_save)}")
            logger.info(f"Columns saved: {len(df_to_save.columns)}")
            
            if include_enhanced_macd:
                # Provide summary of Enhanced MACD data
                if 'action' in df_to_save.columns:
                    buy_signals = (df_to_save['action'] == 'BUY').sum()
                    sell_signals = (df_to_save['action'] == 'SHORT').sum()
                    momentum_weak = (df_to_save['trigger_reason'] == 'MOMENTUM_WEAKENING').sum()
                    momentum_strong = (df_to_save['trigger_reason'] == 'MOMENTUM_STRENGTHENING').sum()
                    
                    logger.info(f"Enhanced MACD Summary:")
                    logger.info(f"  - BUY signals: {buy_signals}")
                    logger.info(f"  - SHORT signals: {sell_signals}")
                    logger.info(f"  - Momentum weakening signals: {momentum_weak}")
                    logger.info(f"  - Momentum strengthening signals: {momentum_strong}")
                
                if 'crossover' in df_to_save.columns:
                    crossovers = df_to_save['crossover'].sum()
                    crossunders = df_to_save['crossunder'].sum()
                    logger.info(f"  - MACD crossovers: {crossovers}")
                    logger.info(f"  - MACD crossunders: {crossunders}")
        
        except Exception as e:
            logger.error(f"Error saving quotes to CSV: {e}")
            
    def save_enhanced_macd_csv(self, filename=None):
        """
        Convenience method to save quotes with Enhanced MACD strategy data.
        
        Args:
            filename: Optional filename, defaults to symbol_enhanced_macd_YYYYMMDD.csv
        """
        self.save_to_csv(filename=filename, include_enhanced_macd=True)
    
    def stop(self):
        """
        Stop the WebSocket connection and clean up resources.
        """
        try:
            self.connected = False
            if self.ws:
                self.ws.close()
                logger.info(f"WebSocket connection closed for {self.symbol}")
        except Exception as e:
            logger.error(f"Error stopping WebSocket connection: {e}")
    
    def diagnose_connection(self):
        """
        Diagnose WebSocket connection and data reception issues.
        
        Returns:
            dict: Diagnosis information
        """
        diagnosis = {
            'connection_status': 'Connected' if self.connected else 'Disconnected',
            'quotes_received': len(self.quotes_df),
            'last_quote_time': None,
            'websocket_url': self.ws_url,
            'symbol': self.symbol,
            'recommendations': []
        }
        
        if len(self.quotes_df) > 0:
            diagnosis['last_quote_time'] = self.quotes_df.iloc[-1]['timestamp']
        
        # Add recommendations based on diagnosis
        if not self.connected:
            diagnosis['recommendations'].append("Check internet connection and Alpaca API credentials")
        
        if len(self.quotes_df) == 0:
            diagnosis['recommendations'].append("No data received - check if market is open and symbol is valid")
            
        from datetime import datetime, timezone
        import pytz
        
        # Check market hours
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        weekday = now_et.weekday()
        hour = now_et.hour
        
        if weekday >= 5:
            diagnosis['recommendations'].append("Market is closed (weekend)")
        elif not (4 <= hour <= 20):
            diagnosis['recommendations'].append("Outside extended trading hours (4 AM - 8 PM ET)")
        
        return diagnosis
    
    def _calculate_display_macd_slope(self, data, lookback=3):
        """Calculate MACD slope for display purposes."""
        slopes = pd.Series(index=data.index, dtype=float)
        
        for i in range(lookback-1, len(data)):
            if i >= lookback-1:
                recent_macd = data['MACD'].iloc[i-lookback+1:i+1].values
                if len(recent_macd) >= 2 and not pd.isna(recent_macd).any():
                    # Simple linear slope calculation
                    slope = (recent_macd[-1] - recent_macd[0]) / (len(recent_macd) - 1)
                    slopes.iloc[i] = slope
        
        return slopes
    
    def _calculate_display_histogram_avg(self, data, lookback=3):
        """Calculate histogram rolling average for display purposes."""
        return data['Histogram'].rolling(window=lookback, min_periods=1).mean()
    
    def _calculate_display_momentum(self, data, slope_threshold=0.001):
        """Calculate momentum signals for display purposes."""
        momentum = pd.Series(index=data.index, dtype=str)
        
        for i in range(len(data)):
            row = data.iloc[i]
            
            if pd.isna(row.get('MACD_slope')) or pd.isna(row.get('Hist_avg')):
                momentum.iloc[i] = 'N/A'
                continue
                
            macd_position = row.get('MACD_position', '')
            slope = row.get('MACD_slope', 0)
            histogram = row.get('Histogram', 0)
            hist_avg = row.get('Hist_avg', 0)
            
            if macd_position == 'ABOVE':
                # Long position analysis
                is_slope_weak = slope < slope_threshold
                is_histogram_weak = histogram < hist_avg
                if is_slope_weak and is_histogram_weak:
                    momentum.iloc[i] = 'WEAK'
                else:
                    momentum.iloc[i] = 'STRONG'
            elif macd_position == 'BELOW':
                # Short position analysis
                is_slope_strong = slope > -slope_threshold
                is_histogram_strong = abs(histogram) < abs(hist_avg)
                if is_slope_strong and is_histogram_strong:
                    momentum.iloc[i] = 'STRONG'
                else:
                    momentum.iloc[i] = 'WEAK'
            else:
                momentum.iloc[i] = 'NEUTRAL'
        
        return momentum

# This class is a drop-in replacement for the original QuoteMonitor
# Just import this and use EnhancedQuoteMonitor instead of QuoteMonitor
QuoteMonitor = EnhancedQuoteMonitor

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced real-time stock quote monitor using WebSockets")
    
    parser.add_argument(
        "--symbol", 
        type=str, 
        default="NVDA",
        help="Stock symbol to monitor (default: NVDA)"
    )
    
    parser.add_argument(
        "--interval", 
        type=int, 
        default=5,
        help="Interval between display updates in seconds (default: 5)"
    )
    
    parser.add_argument(
        "--max-records", 
        type=int, 
        default=200,
        help="Maximum number of records to keep in memory (default: 200)"
    )
    
    args = parser.parse_args()
    
    # Create the quote monitor with WebSocket support
    monitor = EnhancedQuoteMonitor(
        symbol=args.symbol,
        max_records=args.max_records,
        interval_seconds=args.interval
    )
    
    # Give WebSocket time to connect and authenticate
    logger.info("Waiting for WebSocket connection to establish...")
    time.sleep(3)
    
    # Run the monitor - WebSocket runs in background thread
    try:
        while True:
            # Display the current quotes (WebSocket updates happen in background)
            monitor.display_quotes()
            
            # Wait for the next display update
            logger.info(f"Waiting {args.interval} seconds until next display update...")
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        logger.info("Quote monitor stopped by user")
        
        # Close WebSocket connection
        if monitor.ws:
            monitor.ws.close()
            logger.info("WebSocket connection closed")
        
        # Save the quotes to CSV before exiting
        monitor.save_to_csv()