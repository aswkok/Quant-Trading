#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading strategies module for the Alpaca-based quantitative trading system.
This module contains various trading strategies that can be used with the system.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Strategy:
    """Base strategy class that all strategies should inherit from."""
    
    def __init__(self, name="BaseStrategy"):
        """Initialize the strategy."""
        self.name = name
        logger.info(f"Initialized {self.name} strategy")
    
    def generate_signals(self, data):
        """
        Generate trading signals from the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals column added
        """
        raise NotImplementedError("Subclasses must implement generate_signals method")


class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover strategy.
    
    Generates buy signals when the short-term moving average crosses above
    the long-term moving average, and sell signals when it crosses below.
    """
    
    def __init__(self, short_window=20, long_window=50):
        """
        Initialize the strategy with the specified windows.
        
        Args:
            short_window: Window for the short-term moving average
            long_window: Window for the long-term moving average
        """
        super().__init__(name=f"MA_Crossover_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
        logger.info(f"Initialized MA Crossover strategy with windows {short_window}/{long_window}")
    
    def generate_signals(self, data):
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals column added (1 for buy, -1 for sell, 0 for hold)
        """
        # Make a copy of the data to avoid modifying the original
        signals = data.copy()
        
        # Create the short and long moving averages
        signals[f'SMA{self.short_window}'] = signals['close'].rolling(window=self.short_window).mean()
        signals[f'SMA{self.long_window}'] = signals['close'].rolling(window=self.long_window).mean()
        
        # Create signals
        signals['signal'] = 0.0
        
        # Generate signals
        signals['signal'] = np.where(
            signals[f'SMA{self.short_window}'] > signals[f'SMA{self.long_window}'], 1.0, 0.0
        )
        
        # Generate trading orders
        signals['position'] = signals['signal'].diff()
        
        return signals


class RSIStrategy(Strategy):
    """
    Relative Strength Index (RSI) strategy.
    
    Generates buy signals when RSI falls below the oversold level and then rises back above it,
    and sell signals when RSI rises above the overbought level and then falls back below it.
    """
    
    def __init__(self, window=14, oversold=30, overbought=70):
        """
        Initialize the strategy with the specified parameters.
        
        Args:
            window: Window for RSI calculation
            oversold: Oversold threshold
            overbought: Overbought threshold
        """
        super().__init__(name=f"RSI_{window}_{oversold}_{overbought}")
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
        logger.info(f"Initialized RSI strategy with window {window}, oversold {oversold}, overbought {overbought}")
    
    def _calculate_rsi(self, data):
        """
        Calculate the RSI for the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with RSI values
        """
        # Calculate price changes
        delta = data['close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data):
        """
        Generate trading signals based on RSI.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals column added (1 for buy, -1 for sell, 0 for hold)
        """
        # Make a copy of the data to avoid modifying the original
        signals = data.copy()
        
        # Calculate RSI
        signals['RSI'] = self._calculate_rsi(signals)
        
        # Create signals
        signals['signal'] = 0.0
        signals['oversold'] = signals['RSI'] < self.oversold
        signals['overbought'] = signals['RSI'] > self.overbought
        
        # Generate buy signals when RSI crosses above oversold level
        oversold_crossover = (signals['oversold'].shift(1) == True) & (signals['oversold'] == False)
        signals.loc[oversold_crossover, 'signal'] = 1.0
        
        # Generate sell signals when RSI crosses below overbought level
        overbought_crossover = (signals['overbought'].shift(1) == True) & (signals['overbought'] == False)
        signals.loc[overbought_crossover, 'signal'] = -1.0
        
        # Generate trading orders
        signals['position'] = signals['signal']
        
        return signals


class BollingerBandsStrategy(Strategy):
    """
    Bollinger Bands strategy.
    
    Generates buy signals when price touches the lower band and sell signals when it touches the upper band.
    """
    
    def __init__(self, window=20, num_std=2):
        """
        Initialize the strategy with the specified parameters.
        
        Args:
            window: Window for moving average calculation
            num_std: Number of standard deviations for the bands
        """
        super().__init__(name=f"BollingerBands_{window}_{num_std}")
        self.window = window
        self.num_std = num_std
        logger.info(f"Initialized Bollinger Bands strategy with window {window}, std {num_std}")
    
    def generate_signals(self, data):
        """
        Generate trading signals based on Bollinger Bands.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals column added (1 for buy, -1 for sell, 0 for hold)
        """
        # Make a copy of the data to avoid modifying the original
        signals = data.copy()
        
        # Calculate Bollinger Bands
        signals['SMA'] = signals['close'].rolling(window=self.window).mean()
        signals['STD'] = signals['close'].rolling(window=self.window).std()
        signals['UpperBand'] = signals['SMA'] + (signals['STD'] * self.num_std)
        signals['LowerBand'] = signals['SMA'] - (signals['STD'] * self.num_std)
        
        # Create signals
        signals['signal'] = 0.0
        
        # Generate buy signals when price touches the lower band
        signals.loc[signals['close'] <= signals['LowerBand'], 'signal'] = 1.0
        
        # Generate sell signals when price touches the upper band
        signals.loc[signals['close'] >= signals['UpperBand'], 'signal'] = -1.0
        
        # Generate trading orders
        signals['position'] = signals['signal']
        
        return signals


class MACDStrategy(Strategy):
    """
    Enhanced MACD-based Trading Strategy with real-time data support.
    
    Rules:
    1. Buy when the MACD line crosses above the Signal line (crossover).
    2. Sell when the MACD line crosses below the Signal line (crossunder).
    3. If a crossover turns into a crossunder:
       - Sell all previously bought shares
       - Short an additional set of stocks
    4. If a crossunder turns into a crossover:
       - Buy to cover all shorts
       - Buy an additional set of stocks
    5. Option to use real-time quote data from QuoteMonitor
    """
    
    def __init__(self, fast_window=13, slow_window=21, signal_window=9, shares_per_trade=100):
        """
        Initialize the strategy with the specified parameters.
        
        Args:
            fast_window: Window for the fast EMA
            slow_window: Window for the slow EMA
            signal_window: Window for the signal line
            shares_per_trade: Number of shares to trade per signal (default: 100)
        """
        super().__init__(name=f"MACD_{fast_window}_{slow_window}_{signal_window}")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.signal_window = signal_window
        self.shares_per_trade = shares_per_trade
        self.quote_monitor = None  # Will be set if using real-time data
        logger.info(f"Initialized MACD strategy with windows {fast_window}/{slow_window}/{signal_window}")
    
    def set_quote_monitor(self, quote_monitor):
        """
        Set the quote monitor for real-time data.
        
        Args:
            quote_monitor: QuoteMonitor instance for real-time data
        """
        self.quote_monitor = quote_monitor
        logger.info(f"QuoteMonitor set for real-time MACD strategy")
    
    def generate_signals(self, data, symbol=None, use_realtime=False):
        """
        Generate trading signals based on MACD crossovers.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: The stock symbol being analyzed (optional)
            use_realtime: Whether to use real-time data from quote_monitor (default: False)
            
        Returns:
            DataFrame with signals
        """
        # If using real-time data and we have a quote monitor
        if use_realtime and self.quote_monitor and not self.quote_monitor.quotes_df.empty:
            logger.info("Using real-time quote data for MACD calculation")
            signals = self.quote_monitor.quotes_df.copy()
            
            # Calculate MACD components using mid prices from quote monitor
            if 'mid' in signals.columns:
                signals['EMAfast'] = signals['mid'].ewm(span=self.fast_window, adjust=False).mean()
                signals['EMAslow'] = signals['mid'].ewm(span=self.slow_window, adjust=False).mean()
                signals['MACD'] = signals['EMAfast'] - signals['EMAslow']
                signals['Signal'] = signals['MACD'].ewm(span=self.signal_window, adjust=False).mean()
            else:
                logger.warning("No 'mid' column found in quote data, falling back to historical data")
                return self._generate_signals_historical(data, symbol)
        else:
            # Fall back to historical data
            return self._generate_signals_historical(data, symbol)
            
        # Rest of processing remains the same as historical method
        return self._process_macd_signals(signals, symbol)
    
    def _generate_signals_historical(self, data, symbol=None):
        """
        Generate signals using historical OHLCV data.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: The stock symbol being analyzed (optional)
            
        Returns:
            DataFrame with signals
        """
        # Create a copy of the data
        signals = data.copy()
        
        # Calculate required warmup period for accurate MACD
        warmup_period = max(self.slow_window * 3, self.fast_window * 3) + self.signal_window
        
        # Check if we have enough data for accurate MACD calculation
        if len(signals) < warmup_period:
            logger.warning(f"Not enough data for reliable MACD calculation. Have {len(signals)} bars, need at least {warmup_period} bars.")
        else:
            logger.info(f"Sufficient data ({len(signals)} bars) for reliable MACD calculation (minimum: {warmup_period} bars).")
        
        # Calculate MACD components using standard method
        signals['EMAfast'] = signals['close'].ewm(span=self.fast_window, adjust=False).mean()
        signals['EMAslow'] = signals['close'].ewm(span=self.slow_window, adjust=False).mean()
        signals['MACD'] = signals['EMAfast'] - signals['EMAslow']
        signals['Signal'] = signals['MACD'].ewm(span=self.signal_window, adjust=False).mean()
        
        # Process the signals
        return self._process_macd_signals(signals, symbol)
    
    def _process_macd_signals(self, signals, symbol=None):
        """
        Process MACD signals for trading decisions.
        
        Args:
            signals: DataFrame with MACD components calculated
            symbol: The stock symbol being analyzed (optional)
            
        Returns:
            DataFrame with trading signals
        """
        # Add debug logging to see the actual values
        if len(signals) > 0:
            latest = signals.iloc[-1]
            logger.info(f"MACD Calculation - Fast EMA: {latest.get('EMAfast', 0):.4f}, Slow EMA: {latest.get('EMAslow', 0):.4f}")
            logger.info(f"MACD Value: {latest.get('MACD', 0):.6f}, Signal Line: {latest.get('Signal', 0):.6f}")
            logger.info(f"MACD Position: {'ABOVE' if latest.get('MACD', 0) > latest.get('Signal', 0) else 'BELOW'} signal line")
            logger.info(f"MACD Histogram: {(latest.get('MACD', 0) - latest.get('Signal', 0)):.6f}")
        
        # Calculate crossovers
        signals['MACD_prev'] = signals['MACD'].shift(1)
        signals['Signal_prev'] = signals['Signal'].shift(1)
        
        # MACD crosses above Signal line (buy signal)
        macd_crossover = (signals['MACD'] > signals['Signal']) & (signals['MACD_prev'] <= signals['Signal_prev'])
        
        # MACD crosses below Signal line (sell signal)
        macd_crossunder = (signals['MACD'] < signals['Signal']) & (signals['MACD_prev'] >= signals['Signal_prev'])
        
        # Current MACD position (above or below signal line)
        macd_above_signal = signals['MACD'] > signals['Signal']
        
        # Initialize columns
        signals['signal'] = 0.0
        signals['position'] = 0.0
        signals['position_type'] = ''
        signals['shares'] = 0
        signals['action'] = ''
        signals['macd_position'] = ''
        
        # Process signals sequentially to track position transitions
        for i in range(1, len(signals)):
            current_date = signals.index[i]
            
            # Determine current MACD position
            if macd_above_signal.iloc[i]:
                current_macd_position = 'ABOVE'
            else:
                current_macd_position = 'BELOW'
            
            signals.loc[current_date, 'macd_position'] = current_macd_position
            
            # Get previous position type and MACD position
            if i > 1:
                prev_position_type = signals.loc[signals.index[i-1], 'position_type']
                prev_macd_position = signals.loc[signals.index[i-1], 'macd_position']
            else:
                prev_position_type = ''
                prev_macd_position = ''
            
            # Handle crossover (MACD crosses above Signal line)
            if macd_crossover.iloc[i]:
                signals.loc[current_date, 'signal'] = 1.0  # Buy signal
                signals.loc[current_date, 'position'] = 1.0  # Long position
                signals.loc[current_date, 'position_type'] = 'LONG'
                signals.loc[current_date, 'shares'] = self.shares_per_trade
                
                # Check for transition from SHORT to LONG
                if prev_position_type == 'SHORT':
                    # Rule 5: If a crossunder turns into a crossover:
                    # - Buy to cover all shorts
                    # - Buy an additional 100 stocks
                    signals.loc[current_date, 'action'] = 'COVER_AND_BUY'
                elif prev_position_type == '':
                    # Rule 1: When no stock holding, buy 100 stocks on crossover
                    signals.loc[current_date, 'action'] = 'BUY'
                else:
                    # Already in a LONG position, no action needed
                    signals.loc[current_date, 'action'] = ''
                    
            # Handle crossunder (MACD crosses below Signal line)
            elif macd_crossunder.iloc[i]:
                signals.loc[current_date, 'signal'] = -1.0  # Sell signal
                signals.loc[current_date, 'position'] = -1.0  # Short position
                signals.loc[current_date, 'position_type'] = 'SHORT'
                signals.loc[current_date, 'shares'] = self.shares_per_trade
                
                # Check for transition from LONG to SHORT
                if prev_position_type == 'LONG':
                    # Rule 4: If a crossover turns into a crossunder:
                    # - Sell all previously bought shares
                    # - Short an additional 100 stocks
                    signals.loc[current_date, 'action'] = 'SELL_AND_SHORT'
                elif prev_position_type == '':
                    # Rule 2: When no stock holding, sell 100 stocks on crossunder
                    signals.loc[current_date, 'action'] = 'SHORT'
                else:
                    # Already in a SHORT position, no action needed
                    signals.loc[current_date, 'action'] = ''
            
            # No new crossover or crossunder
            else:
                # Carry forward the previous position
                if i > 1:
                    signals.loc[current_date, 'signal'] = signals.loc[signals.index[i-1], 'signal']
                    signals.loc[current_date, 'position'] = signals.loc[signals.index[i-1], 'position']
                    signals.loc[current_date, 'position_type'] = prev_position_type
                    signals.loc[current_date, 'shares'] = signals.loc[signals.index[i-1], 'shares']
                    
                    # Rule 3: If the crossover/crossunder status remains the same, do nothing
                    signals.loc[current_date, 'action'] = ''
                    
                    # Check for position changes without crossover/crossunder
                    if prev_macd_position != current_macd_position:
                        # MACD moved above signal without a proper crossover (rare case)
                        if current_macd_position == 'ABOVE' and prev_position_type == 'SHORT':
                            signals.loc[current_date, 'signal'] = 1.0
                            signals.loc[current_date, 'position'] = 1.0
                            signals.loc[current_date, 'position_type'] = 'LONG'
                            signals.loc[current_date, 'shares'] = self.shares_per_trade
                            signals.loc[current_date, 'action'] = 'COVER_AND_BUY'
                            
                        # MACD moved below signal without a proper crossunder (rare case)
                        elif current_macd_position == 'BELOW' and prev_position_type == 'LONG':
                            signals.loc[current_date, 'signal'] = -1.0
                            signals.loc[current_date, 'position'] = -1.0
                            signals.loc[current_date, 'position_type'] = 'SHORT'
                            signals.loc[current_date, 'shares'] = self.shares_per_trade
                            signals.loc[current_date, 'action'] = 'SELL_AND_SHORT'
        
        return signals
    
    def get_macd_signal_from_monitor(self):
        """
        Get the current MACD signal from the quote monitor if available.
        
        Returns:
            dict: MACD signal information or None if no quote monitor is available
        """
        if self.quote_monitor:
            return self.quote_monitor.get_macd_signal()
        return None


class EnhancedMACDStrategy(Strategy):
    """
    Enhanced MACD-based Trading Strategy with slope and histogram momentum analysis.
    
    This strategy implements the complete Enhanced MACD approach with precise entry and exit rules:
    
    🅰️ Case A: No Stock Holding (Flat Position)
    1. Buy when: MACD Line crosses above Signal Line (Bullish crossover), respecting enhanced signals
    2. Short when: MACD Line crosses below Signal Line (Bearish crossunder), respecting enhanced signals
    
    🅱️ Case B: Holding Long Position
    3. Sell + Short when momentum weakens:
       • MACD Line is above Signal Line
       • MACD Slope is decreasing or near zero (< slope_threshold)
       • MACD Histogram is smaller than the average of the last 3 values
    4. Exit Position (Failsafe): MACD Line falls below Signal Line (Bearish crossunder)
    
    🅲️ Case C: Holding Short Position  
    5. Buy to Cover + Buy when momentum strengthens:
       • MACD Line is below Signal Line
       • MACD Slope is increasing or near zero (> -slope_threshold)
       • Absolute Histogram is smaller than the average of the last 3 absolute values
    6. Exit Position (Failsafe): MACD Line rises above Signal Line (Bullish crossover)
    """
    
    def __init__(self, fast_window=13, slow_window=21, signal_window=9, shares_per_trade=100, 
                 slope_threshold=0.001, slope_lookback=3, histogram_lookback=3, long_only=False):
        """
        Initialize the enhanced strategy with the specified parameters.
        
        Args:
            fast_window: Window for the fast EMA
            slow_window: Window for the slow EMA
            signal_window: Window for the signal line EMA
            shares_per_trade: Number of shares per trade
            slope_threshold: Threshold for MACD slope detection (default: 0.001)
            slope_lookback: Number of periods to look back for slope calculation (default: 3)
            histogram_lookback: Number of periods for histogram averaging (default: 3)
            long_only: If True, only allows BUY orders (no SHORT or SELL_AND_SHORT) (default: False)
        """
        strategy_name = f"EnhancedMACD_{fast_window}_{slow_window}_{signal_window}"
        if long_only:
            strategy_name += "_LongOnly"
        super().__init__(name=strategy_name)
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.signal_window = signal_window
        self.shares_per_trade = shares_per_trade
        self.slope_threshold = slope_threshold
        self.slope_lookback = slope_lookback
        self.histogram_lookback = histogram_lookback
        self.long_only = long_only
        self.quote_monitor = None
        
        logger.info(f"Initialized Enhanced MACD strategy with windows {fast_window}/{slow_window}/{signal_window}")
        logger.info(f"Slope threshold: {slope_threshold}, lookback: {slope_lookback}")
        logger.info(f"Histogram lookback: {histogram_lookback}")
        if long_only:
            logger.info("🔒 LONG-ONLY MODE: Short selling disabled for day trading buying power compatibility")
    
    def set_quote_monitor(self, quote_monitor):
        """
        Set the quote monitor for real-time data.
        
        Args:
            quote_monitor: QuoteMonitor instance for real-time data
        """
        self.quote_monitor = quote_monitor
        logger.info(f"QuoteMonitor set for real-time Enhanced MACD strategy")
    
    def _calculate_macd_slope(self, macd_series):
        """
        Calculate the slope of the MACD line over the lookback period.
        
        Args:
            macd_series: Pandas Series containing MACD values
            
        Returns:
            Pandas Series containing MACD slopes
        """
        slopes = pd.Series(index=macd_series.index, dtype=float)
        
        for i in range(self.slope_lookback, len(macd_series)):
            # Calculate slope using linear regression over the lookback period
            y_values = macd_series.iloc[i-self.slope_lookback+1:i+1].values
            x_values = np.arange(len(y_values))
            
            if len(y_values) >= 2:
                # Simple slope calculation: (y2 - y1) / (x2 - x1)
                slope = (y_values[-1] - y_values[0]) / (len(y_values) - 1)
                slopes.iloc[i] = slope
        
        return slopes
    
    def _calculate_histogram_average(self, histogram_series):
        """
        Calculate the rolling average of histogram values over the lookback period.
        
        Args:
            histogram_series: Pandas Series containing histogram values
            
        Returns:
            Pandas Series containing histogram averages
        """
        return histogram_series.rolling(window=self.histogram_lookback).mean()
    
    def _check_momentum_weakening_long(self, signals, i):
        """
        Check if momentum is weakening for a long position (sell + short conditions).
        
        Enhanced MACD Strategy Case B.3: Sell + Short when holding long position
        - MACD Line is above Signal Line
        - MACD Slope is decreasing or near zero (< slope_threshold)
        - MACD Histogram is smaller than the average of the last 3 values
        
        Args:
            signals: DataFrame with MACD data
            i: Current index
            
        Returns:
            bool: True if conditions are met for selling long and going short
        """
        current_idx = signals.index[i]
        
        # Condition 1: MACD Line is above Signal Line
        macd_above_signal = signals.loc[current_idx, 'MACD'] > signals.loc[current_idx, 'Signal']
        
        if not macd_above_signal:
            return False
        
        # Condition 2: MACD Slope is decreasing or near zero (< slope_threshold)
        current_slope = signals.loc[current_idx, 'MACD_slope']
        
        slope_condition = current_slope < self.slope_threshold
        
        # Condition 3: MACD Histogram is smaller than average of last histogram_lookback values
        current_histogram = signals.loc[current_idx, 'Histogram']
        avg_histogram = signals.loc[current_idx, 'Histogram_avg']
        
        histogram_condition = current_histogram < avg_histogram
        
        logger.debug(f"Long momentum weakening check - MACD above Signal: {macd_above_signal}, "
                    f"Slope: {current_slope:.6f} < {self.slope_threshold} = {slope_condition}, "
                    f"Histogram: {current_histogram:.6f} < {avg_histogram:.6f} = {histogram_condition}")
        
        return slope_condition and histogram_condition
    
    def _check_momentum_strengthening_short(self, signals, i):
        """
        Check if momentum is strengthening for a short position (cover + long conditions).
        
        Enhanced MACD Strategy Case C.5: Buy to Cover + Buy when holding short position
        - MACD Line is below Signal Line
        - MACD Slope is increasing or near zero (> -slope_threshold)
        - Absolute Histogram is smaller than the average of the last 3 absolute values
        
        Args:
            signals: DataFrame with MACD data
            i: Current index
            
        Returns:
            bool: True if conditions are met for covering short and going long
        """
        current_idx = signals.index[i]
        
        # Condition 1: MACD Line is below Signal Line
        macd_below_signal = signals.loc[current_idx, 'MACD'] < signals.loc[current_idx, 'Signal']
        
        if not macd_below_signal:
            return False
        
        # Condition 2: MACD Slope is increasing or near zero (> -slope_threshold)
        current_slope = signals.loc[current_idx, 'MACD_slope']
        
        slope_condition = current_slope > -self.slope_threshold
        
        # Condition 3: Absolute histogram is smaller than average of absolute values of last histogram_lookback
        current_histogram_abs = abs(signals.loc[current_idx, 'Histogram'])
        avg_histogram_abs = signals.loc[current_idx, 'Histogram_abs_avg']
        
        histogram_condition = current_histogram_abs < avg_histogram_abs
        
        logger.debug(f"Short momentum strengthening check - MACD below Signal: {macd_below_signal}, "
                    f"Slope: {current_slope:.6f} > {-self.slope_threshold} = {slope_condition}, "
                    f"Histogram abs: {current_histogram_abs:.6f} < {avg_histogram_abs:.6f} = {histogram_condition}")
        
        return slope_condition and histogram_condition
    
    def generate_signals(self, data, symbol=None, use_realtime=False):
        """
        Generate trading signals based on Enhanced MACD with slope and histogram analysis.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: The stock symbol being analyzed (optional)
            use_realtime: Whether to use real-time data from quote_monitor (default: False)
            
        Returns:
            DataFrame with enhanced trading signals
        """
        # Use real-time data if available and requested
        if use_realtime and self.quote_monitor and not self.quote_monitor.quotes_df.empty:
            logger.info("Using real-time quote data for Enhanced MACD calculation")
            signals = self.quote_monitor.quotes_df.copy()
            
            # Calculate MACD components using mid prices from quote monitor
            if 'mid' in signals.columns:
                signals['EMAfast'] = signals['mid'].ewm(span=self.fast_window, adjust=False).mean()
                signals['EMAslow'] = signals['mid'].ewm(span=self.slow_window, adjust=False).mean()
                signals['MACD'] = signals['EMAfast'] - signals['EMAslow']
                signals['Signal'] = signals['MACD'].ewm(span=self.signal_window, adjust=False).mean()
            else:
                logger.warning("No 'mid' column found in quote data, falling back to historical data")
                return self._generate_signals_historical(data, symbol)
        else:
            # Use historical data
            return self._generate_signals_historical(data, symbol)
        
        # Process the enhanced signals
        return self._process_enhanced_macd_signals(signals, symbol)
    
    def _generate_signals_historical(self, data, symbol=None):
        """
        Generate signals using historical OHLCV data.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: The stock symbol being analyzed (optional)
            
        Returns:
            DataFrame with signals
        """
        signals = data.copy()
        
        # Calculate required warmup period
        warmup_period = max(self.slow_window * 3, self.fast_window * 3) + self.signal_window + self.histogram_lookback
        
        if len(signals) < warmup_period:
            logger.warning(f"Not enough data for reliable Enhanced MACD calculation. Have {len(signals)} bars, need at least {warmup_period} bars.")
        else:
            logger.info(f"Sufficient data ({len(signals)} bars) for reliable Enhanced MACD calculation (minimum: {warmup_period} bars).")
        
        # Calculate MACD components
        signals['EMAfast'] = signals['close'].ewm(span=self.fast_window, adjust=False).mean()
        signals['EMAslow'] = signals['close'].ewm(span=self.slow_window, adjust=False).mean()
        signals['MACD'] = signals['EMAfast'] - signals['EMAslow']
        signals['Signal'] = signals['MACD'].ewm(span=self.signal_window, adjust=False).mean()
        
        return self._process_enhanced_macd_signals(signals, symbol)
    
    def _process_enhanced_macd_signals(self, signals, symbol=None):
        """
        Process Enhanced MACD signals with slope and histogram analysis.
        
        Args:
            signals: DataFrame with MACD components calculated
            symbol: The stock symbol being analyzed (optional)
            
        Returns:
            DataFrame with enhanced trading signals
        """
        # Calculate histogram
        signals['Histogram'] = signals['MACD'] - signals['Signal']
        
        # Calculate MACD slope
        signals['MACD_slope'] = self._calculate_macd_slope(signals['MACD'])
        
        # Calculate histogram averages
        signals['Histogram_avg'] = self._calculate_histogram_average(signals['Histogram'])
        signals['Histogram_abs_avg'] = self._calculate_histogram_average(signals['Histogram'].abs())
        
        # Calculate crossovers using the corrected logic
        signals['MACD_position'] = np.where(signals['MACD'] > signals['Signal'], 'ABOVE', 'BELOW')
        signals['crossover'] = False
        signals['crossunder'] = False
        
        # Detect crossovers properly
        for i in range(1, len(signals)):
            current_position = signals.iloc[i]['MACD_position']
            previous_position = signals.iloc[i-1]['MACD_position']
            
            if current_position == 'ABOVE' and previous_position == 'BELOW':
                signals.iloc[i, signals.columns.get_loc('crossover')] = True
            elif current_position == 'BELOW' and previous_position == 'ABOVE':
                signals.iloc[i, signals.columns.get_loc('crossunder')] = True
        
        # Initialize trading columns
        signals['signal'] = 0.0
        signals['position'] = 0.0
        signals['position_type'] = ''
        signals['shares'] = 0
        signals['action'] = ''
        signals['trigger_reason'] = ''
        
        # Process signals sequentially to track position transitions
        for i in range(max(self.slope_lookback, self.histogram_lookback), len(signals)):
            current_idx = signals.index[i]
            
            # Get previous position
            prev_position_type = signals.loc[signals.index[i-1], 'position_type'] if i > 0 else ''
            
            # 🅰️ Case A: No Stock Holding (Flat Position)
            if not prev_position_type:
                # A.1: Buy when MACD Line crosses above Signal Line (respecting enhanced signals)
                if signals.iloc[i]['crossover']:
                    signals.loc[current_idx, 'signal'] = 1.0
                    signals.loc[current_idx, 'position'] = 1.0
                    signals.loc[current_idx, 'position_type'] = 'LONG'
                    signals.loc[current_idx, 'shares'] = self.shares_per_trade
                    signals.loc[current_idx, 'action'] = 'BUY'
                    signals.loc[current_idx, 'trigger_reason'] = 'MACD_CROSSOVER'
                
                # A.2: Short when MACD Line crosses below Signal Line (respecting enhanced signals)  
                elif signals.iloc[i]['crossunder']:
                    if self.long_only:
                        # Long-only mode: Stay flat instead of shorting
                        signals.loc[current_idx, 'signal'] = 0.0
                        signals.loc[current_idx, 'position'] = 0.0
                        signals.loc[current_idx, 'position_type'] = ''
                        signals.loc[current_idx, 'shares'] = 0
                        signals.loc[current_idx, 'action'] = 'STAY_FLAT'
                        signals.loc[current_idx, 'trigger_reason'] = 'LONG_ONLY_NO_SHORT'
                    else:
                        signals.loc[current_idx, 'signal'] = -1.0
                        signals.loc[current_idx, 'position'] = -1.0
                        signals.loc[current_idx, 'position_type'] = 'SHORT'
                        signals.loc[current_idx, 'shares'] = self.shares_per_trade
                        signals.loc[current_idx, 'action'] = 'SHORT'
                        signals.loc[current_idx, 'trigger_reason'] = 'MACD_CROSSUNDER'
                
                # A.3: Long-only mode: Check for momentum strengthening (from Case C logic)
                elif self.long_only and self._check_momentum_strengthening_short(signals, i):
                    signals.loc[current_idx, 'signal'] = 1.0
                    signals.loc[current_idx, 'position'] = 1.0
                    signals.loc[current_idx, 'position_type'] = 'LONG'
                    signals.loc[current_idx, 'shares'] = self.shares_per_trade
                    signals.loc[current_idx, 'action'] = 'BUY'
                    signals.loc[current_idx, 'trigger_reason'] = 'MOMENTUM_STRENGTHENING_LONG_ONLY'
            
            # 🅱️ Case B: Holding Long Position
            elif prev_position_type == 'LONG':
                # B.3: Sell + Short when momentum weakens (priority condition)
                if self._check_momentum_weakening_long(signals, i):
                    if self.long_only:
                        # Long-only mode: Just sell, don't short
                        signals.loc[current_idx, 'signal'] = 0.0
                        signals.loc[current_idx, 'position'] = 0.0
                        signals.loc[current_idx, 'position_type'] = ''
                        signals.loc[current_idx, 'shares'] = 0
                        signals.loc[current_idx, 'action'] = 'SELL'
                        signals.loc[current_idx, 'trigger_reason'] = 'MOMENTUM_WEAKENING_LONG_ONLY'
                    else:
                        signals.loc[current_idx, 'signal'] = -1.0
                        signals.loc[current_idx, 'position'] = -1.0
                        signals.loc[current_idx, 'position_type'] = 'SHORT'
                        signals.loc[current_idx, 'shares'] = self.shares_per_trade
                        signals.loc[current_idx, 'action'] = 'SELL_AND_SHORT'
                        signals.loc[current_idx, 'trigger_reason'] = 'MOMENTUM_WEAKENING'
                
                # B.4: Exit Position (Failsafe) when MACD falls below Signal Line  
                elif signals.iloc[i]['crossunder']:
                    if self.long_only:
                        # Long-only mode: Just sell, don't short
                        signals.loc[current_idx, 'signal'] = 0.0
                        signals.loc[current_idx, 'position'] = 0.0
                        signals.loc[current_idx, 'position_type'] = ''
                        signals.loc[current_idx, 'shares'] = 0
                        signals.loc[current_idx, 'action'] = 'SELL'
                        signals.loc[current_idx, 'trigger_reason'] = 'FAILSAFE_CROSSUNDER_LONG_ONLY'
                    else:
                        signals.loc[current_idx, 'signal'] = -1.0
                        signals.loc[current_idx, 'position'] = -1.0
                        signals.loc[current_idx, 'position_type'] = 'SHORT'
                        signals.loc[current_idx, 'shares'] = self.shares_per_trade
                        signals.loc[current_idx, 'action'] = 'SELL_AND_SHORT'
                        signals.loc[current_idx, 'trigger_reason'] = 'FAILSAFE_CROSSUNDER'
                
                else:
                    # Hold long position
                    signals.loc[current_idx, 'signal'] = signals.loc[signals.index[i-1], 'signal']
                    signals.loc[current_idx, 'position'] = signals.loc[signals.index[i-1], 'position']
                    signals.loc[current_idx, 'position_type'] = prev_position_type
                    signals.loc[current_idx, 'shares'] = signals.loc[signals.index[i-1], 'shares']
                    signals.loc[current_idx, 'action'] = 'HOLD'
            
            # 🅲️ Case C: Holding Short Position
            elif prev_position_type == 'SHORT':
                # C.5: Buy to Cover + Buy when momentum strengthens (priority condition)
                if self._check_momentum_strengthening_short(signals, i):
                    signals.loc[current_idx, 'signal'] = 1.0
                    signals.loc[current_idx, 'position'] = 1.0
                    signals.loc[current_idx, 'position_type'] = 'LONG'
                    signals.loc[current_idx, 'shares'] = self.shares_per_trade
                    signals.loc[current_idx, 'action'] = 'COVER_AND_BUY'
                    signals.loc[current_idx, 'trigger_reason'] = 'MOMENTUM_STRENGTHENING'
                
                # C.6: Exit Position (Failsafe) when MACD rises above Signal Line
                elif signals.iloc[i]['crossover']:
                    signals.loc[current_idx, 'signal'] = 1.0
                    signals.loc[current_idx, 'position'] = 1.0
                    signals.loc[current_idx, 'position_type'] = 'LONG'
                    signals.loc[current_idx, 'shares'] = self.shares_per_trade
                    signals.loc[current_idx, 'action'] = 'COVER_AND_BUY'
                    signals.loc[current_idx, 'trigger_reason'] = 'FAILSAFE_CROSSOVER'
                
                else:
                    # Hold short position
                    signals.loc[current_idx, 'signal'] = signals.loc[signals.index[i-1], 'signal']
                    signals.loc[current_idx, 'position'] = signals.loc[signals.index[i-1], 'position']
                    signals.loc[current_idx, 'position_type'] = prev_position_type
                    signals.loc[current_idx, 'shares'] = signals.loc[signals.index[i-1], 'shares']
                    signals.loc[current_idx, 'action'] = 'HOLD'
        
        return signals
    
    def get_macd_signal_from_monitor(self):
        """
        Get the current Enhanced MACD signal from the quote monitor if available.
        
        Returns:
            dict: Enhanced MACD signal information or None if no quote monitor is available
        """
        if self.quote_monitor:
            return self.quote_monitor.get_macd_signal()
        return None
    
    def save_signals_to_csv(self, signals_df, filename=None, symbol="UNKNOWN"):
        """
        Save Enhanced MACD strategy signals and calculations to a CSV file.
        
        Args:
            signals_df: DataFrame with Enhanced MACD signals
            filename: Optional filename, defaults to enhanced_macd_signals_YYYYMMDD.csv
            symbol: Symbol name for filename generation
        """
        if signals_df is None or signals_df.empty:
            logger.info("No signals to save.")
            return
            
        if filename is None:
            today = datetime.now().strftime('%Y%m%d')
            filename = f"{symbol}_enhanced_macd_signals_{today}.csv"
        
        try:
            # Create a comprehensive export with all Enhanced MACD data
            export_df = signals_df.copy()
            
            # Add metadata columns for better documentation
            export_df['strategy_name'] = self.name
            export_df['fast_window'] = self.fast_window
            export_df['slow_window'] = self.slow_window
            export_df['signal_window'] = self.signal_window
            export_df['slope_threshold'] = self.slope_threshold
            export_df['slope_lookback'] = self.slope_lookback
            export_df['histogram_lookback'] = self.histogram_lookback
            export_df['shares_per_trade'] = self.shares_per_trade
            
            # Reorder columns for better readability
            column_order = [
                'strategy_name', 'fast_window', 'slow_window', 'signal_window',
                'slope_threshold', 'slope_lookback', 'histogram_lookback', 'shares_per_trade',
                'close', 'EMAfast', 'EMAslow', 'MACD', 'Signal', 'Histogram',
                'MACD_slope', 'Histogram_avg', 'Histogram_abs_avg',
                'MACD_position', 'crossover', 'crossunder',
                'signal', 'position', 'position_type', 'shares',
                'action', 'trigger_reason'
            ]
            
            # Only include columns that exist in the dataframe
            available_columns = [col for col in column_order if col in export_df.columns]
            remaining_columns = [col for col in export_df.columns if col not in available_columns]
            
            export_df = export_df[available_columns + remaining_columns]
            
            # Save to CSV
            export_df.to_csv(filename, index=True)  # Include index (dates) for time series data
            
            # Generate comprehensive summary
            logger.info(f"Enhanced MACD signals saved to {filename}")
            logger.info(f"Strategy: {self.name}")
            logger.info(f"Parameters: Fast={self.fast_window}, Slow={self.slow_window}, Signal={self.signal_window}")
            logger.info(f"Slope threshold: {self.slope_threshold}, Slope lookback: {self.slope_lookback}")
            logger.info(f"Histogram lookback: {self.histogram_lookback}")
            logger.info(f"Total records: {len(export_df)}")
            logger.info(f"Columns saved: {len(export_df.columns)}")
            
            # Detailed signal analysis
            if 'action' in export_df.columns:
                actions_summary = export_df['action'].value_counts()
                logger.info("Actions Summary:")
                for action, count in actions_summary.items():
                    if action and action != 'HOLD':  # Skip empty and HOLD actions
                        logger.info(f"  - {action}: {count}")
            
            if 'trigger_reason' in export_df.columns:
                triggers_summary = export_df['trigger_reason'].value_counts()
                logger.info("Enhanced MACD Trigger Reasons:")
                for trigger, count in triggers_summary.items():
                    if trigger:  # Skip empty triggers
                        case_indicator = ""
                        if 'CROSSOVER' in trigger and 'FAILSAFE' not in trigger:
                            case_indicator = " (🅰️ Case A.1 or 🅲️ Case C.6)"
                        elif 'CROSSUNDER' in trigger and 'FAILSAFE' not in trigger:
                            case_indicator = " (🅰️ Case A.2 or 🅱️ Case B.4)"
                        elif 'MOMENTUM_WEAKENING' in trigger:
                            case_indicator = " (🅱️ Case B.3)"
                        elif 'MOMENTUM_STRENGTHENING' in trigger:
                            case_indicator = " (🅲️ Case C.5)"
                        elif 'FAILSAFE_CROSSUNDER' in trigger:
                            case_indicator = " (🅱️ Case B.4 Failsafe)"
                        elif 'FAILSAFE_CROSSOVER' in trigger:
                            case_indicator = " (🅲️ Case C.6 Failsafe)"
                        logger.info(f"  - {trigger}: {count}{case_indicator}")
            
            if 'crossover' in export_df.columns and 'crossunder' in export_df.columns:
                crossovers = export_df['crossover'].sum()
                crossunders = export_df['crossunder'].sum()
                logger.info(f"MACD Crossovers: {crossovers}")
                logger.info(f"MACD Crossunders: {crossunders}")
            
            # Position analysis
            if 'position_type' in export_df.columns:
                positions = export_df['position_type'].value_counts()
                logger.info("Position Distribution:")
                for pos, count in positions.items():
                    if pos:  # Skip empty positions
                        logger.info(f"  - {pos}: {count}")
            
            # Performance metrics (basic)
            if 'signal' in export_df.columns and 'close' in export_df.columns:
                signal_changes = export_df[export_df['signal'].diff() != 0]
                if len(signal_changes) > 0:
                    logger.info(f"Total signal changes: {len(signal_changes)}")
                    
                    # Calculate basic return if we have price data
                    if len(signal_changes) >= 2:
                        first_price = signal_changes['close'].iloc[0]
                        last_price = signal_changes['close'].iloc[-1]
                        total_return = ((last_price - first_price) / first_price) * 100
                        logger.info(f"Period price change: {total_return:.2f}%")
            
            return filename
            
        except Exception as e:
            logger.error(f"Error saving Enhanced MACD signals to CSV: {e}")
            return None


# Default strategy configurations
DEFAULT_STRATEGY_CONFIGS = {
    'moving_average_crossover': {
        'short_window': 20,
        'long_window': 50
    },
    'rsi': {
        'window': 14,
        'oversold': 30,
        'overbought': 70
    },
    'bollinger_bands': {
        'window': 20,
        'num_std': 2
    },
    'macd': {
        'fast_window': 13,
        'slow_window': 21,
        'signal_window': 9
    },
    'enhanced_macd': {
        'fast_window': 13,
        'slow_window': 21,
        'signal_window': 9
    }
}

# Trading symbols configuration
# DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'PLTR']
DEFAULT_SYMBOLS = ['NVDA']

# Default active strategy
DEFAULT_STRATEGY = 'macd'

class StrategyFactory:
    """Factory class for creating strategy instances."""
    
    @staticmethod
    def get_strategy(strategy_name, **kwargs):
        """
        Get a strategy instance based on the strategy name.
        
        Args:
            strategy_name: Name of the strategy to create
            **kwargs: Additional parameters for the strategy
            
        Returns:
            Strategy instance
        """
        strategies = {
            'moving_average_crossover': MovingAverageCrossover,
            'rsi': RSIStrategy,
            'bollinger_bands': BollingerBandsStrategy,
            'macd': MACDStrategy,
            'enhanced_macd': EnhancedMACDStrategy
        }
        
        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Get default config for this strategy
        default_config = DEFAULT_STRATEGY_CONFIGS.get(strategy_name, {})
        
        # Override defaults with any provided kwargs
        config = {**default_config, **kwargs}
        
        return strategies[strategy_name](**config)


# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(100, 5, 100),
        'volume': np.random.normal(1000000, 200000, 100)
    }, index=dates)
    
    # Create a strategy
    strategy = StrategyFactory.get_strategy('moving_average_crossover', short_window=10, long_window=30)
    
    # Generate signals
    signals = strategy.generate_signals(data)
    
    # Print the signals
    print(signals[['close', 'SMA10', 'SMA30', 'signal', 'position']].tail())
