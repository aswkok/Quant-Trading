#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alpaca Close Price Monitor for MACD Trading System

This is a specialized version of the enhanced quote monitor that uses closing prices
from Alpaca's historical data API instead of real-time mid-prices (bid+ask)/2 for 
MACD calculations, allowing for comparison of different price sources for signal generation.
"""

import os
import time
import logging
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Import the base EnhancedQuoteMonitor to extend it
from enhanced_quote_monitor import EnhancedQuoteMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlpacaCloseQuoteMonitor(EnhancedQuoteMonitor):
    """
    Enhanced monitor for Alpaca that uses closing prices for MACD calculations.
    
    This class extends the EnhancedQuoteMonitor but fetches and uses historical 
    closing prices from Alpaca's data API for calculating MACD indicators instead 
    of real-time mid-prices from bid/ask WebSocket data.
    """
    
    def __init__(self, symbol, max_records=200, interval_seconds=60, fast_window=13, slow_window=21, signal_window=9):
        """
        Initialize the close price monitor with Alpaca historical data support.
        
        Args:
            symbol: Stock symbol to monitor
            max_records: Maximum number of records to keep in memory
            interval_seconds: Interval between data fetches in seconds
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
        
        # Initialize only the historical data client for close prices
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        
        # Monitor settings
        self.symbol = symbol.upper()
        self.max_records = max_records
        self.interval_seconds = interval_seconds
        
        # Data storage - modified to include close price
        self.quotes_df = pd.DataFrame(columns=['timestamp', 'bid', 'ask', 'spread', 'spread_pct', 'mid', 'close'])
        
        # MACD parameters
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.signal_window = signal_window
        
        # MACD state tracking
        self.last_macd_position = None
        self.last_signal_time = None
        
        # Background fetching
        self.fetching = False
        self.fetch_thread = None
        
        # Initialize with historical data
        self._fetch_initial_data()
        
        # Start background fetching thread
        self._start_background_fetching()
        
        logger.info(f"Alpaca Close Price Monitor initialized for {symbol}")
        logger.info(f"Data fetch interval: {interval_seconds} seconds")
        logger.info(f"Maximum records: {max_records}")
        logger.info(f"MACD Parameters: Fast={fast_window}, Slow={slow_window}, Signal={signal_window}")
        logger.info("Using CLOSE prices for MACD calculations")
    
    def _fetch_initial_data(self):
        """
        Fetch initial historical data to populate the dataframe.
        """
        try:
            # Calculate how much data we need for proper MACD calculation
            warmup_period = max(self.slow_window * 3, self.fast_window * 3) + self.signal_window
            days_needed = max(warmup_period // 390 + 5, 10)  # 390 minutes per trading day, add buffer
            
            # Get historical bars
            start_date = datetime.now(timezone.utc) - timedelta(days=days_needed)
            end_date = datetime.now(timezone.utc)
            
            request = StockBarsRequest(
                symbol_or_symbols=[self.symbol],
                timeframe=TimeFrame.Minute,
                start=start_date,
                end=end_date
            )
            
            bars = self.data_client.get_stock_bars(request)
            
            if self.symbol in bars.df.index.get_level_values(0):
                df = bars.df.loc[self.symbol]
                
                # Convert to our format
                for idx, row in df.tail(self.max_records).iterrows():
                    timestamp = idx.tz_convert('US/Eastern')
                    close_price = float(row['close'])
                    
                    # For historical data, we'll use close price for bid/ask as well
                    # This maintains compatibility with the parent class structure
                    new_row = pd.DataFrame({
                        'timestamp': [timestamp],
                        'bid': [close_price],  # Use close price
                        'ask': [close_price],  # Use close price
                        'spread': [0.0],       # No spread for close prices
                        'spread_pct': [0.0],   # No spread percentage
                        'mid': [close_price],  # Mid price same as close
                        'close': [close_price] # Actual close price
                    })
                    
                    self.quotes_df = pd.concat([self.quotes_df, new_row], ignore_index=True)
                
                logger.info(f"Loaded {len(self.quotes_df)} historical records for {self.symbol}")
                
                # Calculate initial MACD
                self.calculate_macd()
            else:
                logger.warning(f"No historical data found for {self.symbol}")
                
        except Exception as e:
            logger.error(f"Error fetching initial data: {e}")
    
    def _start_background_fetching(self):
        """
        Start the background thread for continuous data fetching.
        """
        self.fetching = True
        self.fetch_thread = threading.Thread(target=self._fetch_data_loop)
        self.fetch_thread.daemon = True
        self.fetch_thread.start()
        logger.info(f"Started background fetching thread for {self.symbol}")
    
    def _fetch_data_loop(self):
        """
        Background loop to fetch latest close prices periodically.
        """
        while self.fetching:
            try:
                self._fetch_latest_close()
                time.sleep(self.interval_seconds)
            except Exception as e:
                logger.error(f"Error in background fetching: {e}")
                time.sleep(self.interval_seconds)
    
    def _fetch_latest_close(self):
        """
        Fetch the latest close price from Alpaca historical data.
        """
        try:
            # Get the last few minutes of data to ensure we get the latest close
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(minutes=10)
            
            request = StockBarsRequest(
                symbol_or_symbols=[self.symbol],
                timeframe=TimeFrame.Minute,
                start=start_date,
                end=end_date
            )
            
            bars = self.data_client.get_stock_bars(request)
            
            if self.symbol in bars.df.index.get_level_values(0):
                df = bars.df.loc[self.symbol]
                
                if not df.empty:
                    # Get the latest bar
                    latest_bar = df.iloc[-1]
                    timestamp = df.index[-1].tz_convert('US/Eastern')
                    close_price = float(latest_bar['close'])
                    
                    # Check if this is a new timestamp
                    if len(self.quotes_df) == 0 or timestamp > self.quotes_df.iloc[-1]['timestamp']:
                        # Add new close price data
                        new_row = pd.DataFrame({
                            'timestamp': [timestamp],
                            'bid': [close_price],
                            'ask': [close_price],
                            'spread': [0.0],
                            'spread_pct': [0.0],
                            'mid': [close_price],
                            'close': [close_price]
                        })
                        
                        self.quotes_df = pd.concat([self.quotes_df, new_row], ignore_index=True)
                        
                        # Trim to max_records
                        if len(self.quotes_df) > self.max_records:
                            self.quotes_df = self.quotes_df.iloc[-self.max_records:]
                        
                        # Recalculate MACD
                        self.calculate_macd()
                        
                        logger.debug(f"Updated close price for {self.symbol}: ${close_price:.2f} at {timestamp}")
                        
        except Exception as e:
            logger.error(f"Error fetching latest close price: {e}")
    
    def calculate_macd(self):
        """
        Calculate MACD based on the close prices instead of mid-prices.
        
        This method overrides the parent class method to use close prices
        for MACD calculation instead of mid-prices.
        """
        # Check if we have enough data
        warmup_period = max(self.slow_window * 3, self.fast_window * 3) + self.signal_window
        if len(self.quotes_df) < warmup_period:
            logger.debug(f"Not enough data for reliable MACD calculation. Have {len(self.quotes_df)} records, need {warmup_period}.")
            return
        
        # Calculate MACD components using close prices instead of mid prices
        self.quotes_df['EMAfast'] = self.quotes_df['close'].ewm(span=self.fast_window, adjust=False).mean()
        self.quotes_df['EMAslow'] = self.quotes_df['close'].ewm(span=self.slow_window, adjust=False).mean()
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
    
    def get_macd_signal(self):
        """
        Get the latest MACD signal using close prices.
        
        Returns:
            dict: Dictionary containing MACD signal information with close price data
        """
        if len(self.quotes_df) == 0:
            return {
                'signal': 0.0,
                'macd': 0.0,
                'signal_line': 0.0,
                'histogram': 0.0,
                'timestamp': None,
                'close_price': 0.0
            }
        
        latest = self.quotes_df.iloc[-1]
        
        # Determine signal based on crossovers
        signal = 0.0
        if 'crossover' in latest and latest['crossover']:
            signal = 1.0  # Buy signal
        elif 'crossunder' in latest and latest['crossunder']:
            signal = -1.0  # Sell signal
        
        return {
            'signal': signal,
            'macd': latest.get('MACD', 0.0),
            'signal_line': latest.get('Signal', 0.0),
            'histogram': latest.get('Histogram', 0.0),
            'timestamp': latest['timestamp'],
            'close_price': latest['close']  # Return close price instead of mid price
        }
    
    def stop(self):
        """
        Stop the background fetching thread.
        """
        self.fetching = False
        if self.fetch_thread and self.fetch_thread.is_alive():
            self.fetch_thread.join(timeout=5)
        logger.info(f"Stopped Alpaca close price monitor for {self.symbol}")

# Alias for compatibility
AlpacaCloseMonitor = AlpacaCloseQuoteMonitor