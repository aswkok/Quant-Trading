#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integrated MACD Trading System

This script integrates real-time quote monitoring with MACD-based trading decisions
and executes trades on Alpaca. It forms a continuous workflow from live data feeds
to strategy calculation to order execution.
"""

import os
import time
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from alpaca.trading.enums import OrderSide, TimeInForce

# Import our components
from quote_monitor_selector import QuoteMonitor  # This will import from either Alpaca or Yahoo Finance
from strategies import MACDStrategy, EnhancedMACDStrategy, StrategyFactory
from main import AlpacaTradingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("integrated_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntegratedMACDTrader:
    """
    Integrated trading system that connects real-time quote data to MACD strategy and Alpaca execution.
    """
    
    def __init__(self, symbol, interval_seconds=60, fast_window=13, slow_window=21, signal_window=9, 
                 shares_per_trade=100, extended_hours=True, warmup_period_minutes=60, strategy_name='macd',
                 slope_threshold=0.001, slope_lookback=3, histogram_lookback=3, long_only=False):
        """
        Initialize the integrated trading system.
        
        Args:
            symbol: Stock symbol to trade
            interval_seconds: Update interval in seconds
            fast_window: Fast EMA window for MACD
            slow_window: Slow EMA window for MACD  
            signal_window: Signal line EMA window for MACD
            shares_per_trade: Number of shares per trade
            extended_hours: Enable extended hours trading
            warmup_period_minutes: Minutes to collect data before trading
            strategy_name: Strategy to use ('macd' or 'enhanced_macd')
            slope_threshold: MACD slope threshold for Enhanced MACD (default: 0.001)
            slope_lookback: Slope calculation lookback period (default: 3)
            histogram_lookback: Histogram averaging period (default: 3)
            extended_hours: Whether to trade during extended hours
            warmup_period_minutes: Warm-up period in minutes before trading begins
        """
        # Load environment variables
        load_dotenv(override=True)
        
        # Store configuration
        self.symbol = symbol
        self.interval_seconds = interval_seconds
        self.shares_per_trade = shares_per_trade
        self.extended_hours = extended_hours
        self.warmup_period_minutes = warmup_period_minutes
        
        # Store MACD parameters
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.signal_window = signal_window
        
        # Store Enhanced MACD parameters
        self.strategy_name = strategy_name
        self.slope_threshold = slope_threshold
        self.slope_lookback = slope_lookback
        self.histogram_lookback = histogram_lookback
        
        # Initialize components
        logger.info(f"Initializing integrated trading system for {symbol}")
        
        # 1. Initialize the Quote Monitor (either Alpaca or Yahoo Finance based on environment variable)
        self.quote_monitor = QuoteMonitor(
            symbol=symbol,
            max_records=max(slow_window * 3, 500),  # Keep enough records for good MACD calculation
            interval_seconds=interval_seconds,
            fast_window=fast_window,
            slow_window=slow_window,
            signal_window=signal_window
        )
        
        # 2. Initialize the MACD Strategy
        strategy_params = {
            'fast_window': fast_window,
            'slow_window': slow_window,
            'signal_window': signal_window,
            'shares_per_trade': shares_per_trade
        }
        
        # Add Enhanced MACD specific parameters if using enhanced strategy
        if strategy_name == 'enhanced_macd':
            strategy_params.update({
                'slope_threshold': slope_threshold,
                'slope_lookback': slope_lookback,
                'histogram_lookback': histogram_lookback,
                'long_only': long_only
            })
        
        self.strategy = StrategyFactory.get_strategy(strategy_name, **strategy_params)
        logger.info(f"Initialized {strategy_name} strategy: {self.strategy.name}")
        
        # Set the quote monitor for the strategy
        self.strategy.set_quote_monitor(self.quote_monitor)
        
        # If using Enhanced MACD, connect the strategy to the quote monitor
        if strategy_name == 'enhanced_macd' and hasattr(self.quote_monitor, 'set_enhanced_strategy'):
            self.quote_monitor.set_enhanced_strategy(self.strategy)
        
        # 3. Initialize the Alpaca Trading System
        self.trading_system = AlpacaTradingSystem()
        
        # Set trading parameters
        self.trading_system.extended_hours = extended_hours
        
        # System state
        self.is_running = False
        self.start_time = None
        self.last_trade_time = None
        self.position_type = "NONE"  # NONE, LONG, SHORT
        self.position_shares = 0
        
        logger.info(f"Integrated trading system initialized for {symbol}")
        logger.info(f"MACD Parameters: Fast={fast_window}, Slow={slow_window}, Signal={signal_window}")
        logger.info(f"Shares per trade: {shares_per_trade}")
        logger.info(f"Extended hours trading: {extended_hours}")
        logger.info(f"Warm-up period: {warmup_period_minutes} minutes")
    
    def is_warmup_complete(self):
        """Check if the warm-up period is complete."""
        if not self.start_time:
            return False
            
        elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60
        return elapsed_minutes >= self.warmup_period_minutes
    
    def _log_enhanced_macd_details(self):
        """Log Enhanced MACD calculation details for debugging and monitoring."""
        if not self.quote_monitor.quotes_df.empty and len(self.quote_monitor.quotes_df) > self.slope_lookback:
            df = self.quote_monitor.quotes_df
            latest = df.iloc[-1]
            
            # Check if we have MACD data
            if 'MACD' in df.columns and 'Signal' in df.columns:
                # Get latest MACD values
                macd_value = latest['MACD']
                signal_value = latest['Signal']
                histogram = latest['Histogram'] if 'Histogram' in df.columns else macd_value - signal_value
                
                # Calculate Enhanced MACD metrics if we have enough data
                if len(df) >= self.slope_lookback:
                    # Calculate MACD slope (simplified version for logging)
                    recent_macd = df['MACD'].iloc[-self.slope_lookback:].values
                    if len(recent_macd) >= 2:
                        slope = (recent_macd[-1] - recent_macd[0]) / (len(recent_macd) - 1)
                    else:
                        slope = 0.0
                    
                    # Calculate histogram average
                    recent_histogram = df['Histogram'].iloc[-self.histogram_lookback:] if 'Histogram' in df.columns else []
                    if len(recent_histogram) >= self.histogram_lookback:
                        hist_avg = recent_histogram.mean()
                        hist_abs_avg = recent_histogram.abs().mean()
                    else:
                        hist_avg = histogram
                        hist_abs_avg = abs(histogram)
                    
                    # Current position check
                    macd_position = 'ABOVE' if macd_value > signal_value else 'BELOW'
                    
                    # Log Enhanced MACD details
                    logger.info(f"📊 Enhanced MACD Analysis for {self.symbol}:")
                    logger.info(f"   💹 Current Price: ${latest['mid']:.2f} (Bid: ${latest['bid']:.2f}, Ask: ${latest['ask']:.2f})")
                    logger.info(f"   📈 MACD: {macd_value:.6f} | Signal: {signal_value:.6f} | Position: {macd_position}")
                    logger.info(f"   📊 Histogram: {histogram:.6f}")
                    logger.info(f"   ⬆️ MACD Slope: {slope:.6f} (threshold: ±{self.slope_threshold})")
                    logger.info(f"   📊 Histogram Avg ({self.histogram_lookback}p): {hist_avg:.6f} | Abs Avg: {hist_abs_avg:.6f}")
                    
                    # Momentum analysis
                    if macd_position == 'ABOVE':
                        is_slope_weak = slope < self.slope_threshold
                        is_histogram_weak = histogram < hist_avg
                        momentum_weakening = is_slope_weak and is_histogram_weak
                        logger.info(f"   🔍 Long Position Analysis:")
                        logger.info(f"      - Slope Weakening: {'YES' if is_slope_weak else 'NO'} (slope < {self.slope_threshold})")
                        logger.info(f"      - Histogram Declining: {'YES' if is_histogram_weak else 'NO'} (current < avg)")
                        logger.info(f"      - Momentum Weakening: {'YES ⚠️' if momentum_weakening else 'NO ✅'}")
                    else:
                        is_slope_strong = slope > -self.slope_threshold
                        is_histogram_strong = abs(histogram) < hist_abs_avg
                        momentum_strengthening = is_slope_strong and is_histogram_strong
                        logger.info(f"   🔍 Short Position Analysis:")
                        logger.info(f"      - Slope Strengthening: {'YES' if is_slope_strong else 'NO'} (slope > -{self.slope_threshold})")
                        logger.info(f"      - Histogram Compressing: {'YES' if is_histogram_strong else 'NO'} (|current| < |avg|)")
                        logger.info(f"      - Momentum Strengthening: {'YES ⚠️' if momentum_strengthening else 'NO ✅'}")
                    
                    # Log Enhanced MACD signals and actions
                    if 'action' in df.columns and latest['action']:
                        action = latest['action']
                        trigger = latest.get('trigger_reason', '')
                        logger.info(f"   📢 ENHANCED MACD SIGNAL:")
                        logger.info(f"      Action: {action}")
                        logger.info(f"      Trigger: {trigger}")
                        
                        # Case-specific logging
                        if action == 'BUY' and 'CROSSOVER' in trigger:
                            logger.info(f"      🅰️ Case A.1: Buy on bullish crossover (no position)")
                        elif action == 'BUY' and 'MOMENTUM_STRENGTHENING_LONG_ONLY' in trigger:
                            logger.info(f"      🅰️ Case A.3: Buy on momentum strengthening (long-only mode)")
                        elif action == 'SHORT' and 'CROSSUNDER' in trigger:
                            logger.info(f"      🅰️ Case A.2: Short on bearish crossunder (no position)")
                        elif action == 'SELL_AND_SHORT' and 'MOMENTUM_WEAKENING' in trigger:
                            logger.info(f"      🅱️ Case B.3: Sell+Short on momentum weakening (long position)")
                        elif action == 'SELL_AND_SHORT' and 'FAILSAFE_CROSSUNDER' in trigger:
                            logger.info(f"      🅱️ Case B.4: Failsafe exit - MACD fell below Signal Line (long position)")
                        elif action == 'COVER_AND_BUY' and 'MOMENTUM_STRENGTHENING' in trigger:
                            logger.info(f"      🅲️ Case C.5: Cover+Buy on momentum strengthening (short position)")
                        elif action == 'COVER_AND_BUY' and 'FAILSAFE_CROSSOVER' in trigger:
                            logger.info(f"      🅲️ Case C.6: Failsafe exit - MACD rose above Signal Line (short position)")
                    
                    # Log basic crossover/crossunder status if no enhanced action
                    elif 'crossover' in df.columns and latest['crossover']:
                        logger.info(f"   🚀 BULLISH CROSSOVER DETECTED!")
                    elif 'crossunder' in df.columns and latest['crossunder']:
                        logger.info(f"   📉 BEARISH CROSSUNDER DETECTED!")
    
    def _apply_enhanced_macd_to_monitor(self):
        """Apply Enhanced MACD strategy calculations to the quote monitor data."""
        if not self.quote_monitor.quotes_df.empty and len(self.quote_monitor.quotes_df) >= 30:
            try:
                # Convert quote monitor data to format expected by Enhanced MACD strategy
                strategy_data = self.quote_monitor.quotes_df.copy()
                
                # Use appropriate price column based on monitor type
                price_column = 'close' if 'close' in strategy_data.columns else 'mid'
                strategy_data['close'] = strategy_data[price_column]
                strategy_data['open'] = strategy_data[price_column]  
                strategy_data['high'] = strategy_data.get('ask', strategy_data[price_column])
                strategy_data['low'] = strategy_data.get('bid', strategy_data[price_column])
                strategy_data['volume'] = 100000  # Placeholder volume
                
                # Generate Enhanced MACD signals using the SAME strategy instance used for trading
                enhanced_signals = self.strategy.generate_signals(strategy_data)
                
                # Apply Enhanced MACD columns to the quote monitor dataframe
                enhanced_columns = [
                    'MACD_slope', 'Histogram_avg', 'Histogram_abs_avg',
                    'signal', 'position', 'position_type', 'shares', 
                    'action', 'trigger_reason'
                ]
                
                for col in enhanced_columns:
                    if col in enhanced_signals.columns:
                        # Only update if we have valid data and the indices align
                        if not enhanced_signals[col].isna().all() and len(enhanced_signals[col]) == len(self.quote_monitor.quotes_df):
                            # Align indices to ensure proper data matching
                            if enhanced_signals.index.equals(self.quote_monitor.quotes_df.index):
                                self.quote_monitor.quotes_df[col] = enhanced_signals[col]
                            else:
                                # If indices don't match, align them properly
                                self.quote_monitor.quotes_df[col] = enhanced_signals[col].reindex(self.quote_monitor.quotes_df.index)
                        else:
                            logger.debug(f"Skipping column {col}: invalid data or length mismatch")
                
                logger.debug(f"Applied Enhanced MACD calculations to {len(enhanced_columns)} columns in quote monitor")
                
            except Exception as e:
                logger.warning(f"Could not apply Enhanced MACD calculations to quote monitor: {e}")
                import traceback
                logger.debug(f"Full error: {traceback.format_exc()}")
    
    def _process_enhanced_macd_signals(self, current_qty, current_side, macd_signal):
        """
        Process Enhanced MACD signals by checking the action column from the strategy.
        This respects the Enhanced MACD rules for momentum weakening/strengthening.
        
        Returns:
            tuple: (action, qty)
        """
        action = None
        qty = 0
        
        # Get Enhanced MACD action from the quote monitor dataframe if available
        enhanced_action = None
        if not self.quote_monitor.quotes_df.empty and 'action' in self.quote_monitor.quotes_df.columns:
            latest_action = self.quote_monitor.quotes_df.iloc[-1].get('action', '')
            if latest_action and latest_action != '':
                enhanced_action = latest_action
        
        # Extract basic signal information for fallback
        macd_position = macd_signal['macd_position']
        crossover = macd_signal['crossover']
        crossunder = macd_signal['crossunder']
        
        logger.info(f"Enhanced MACD Analysis: action={enhanced_action}, position={current_side}, MACD={macd_position}")
        
        # Priority 1: Enhanced MACD Signals (from strategy analysis)
        if enhanced_action == 'BUY':
            action = "BUY"
            qty = self.shares_per_trade
            # Check trigger reason for more specific logging
            trigger = self.quote_monitor.quotes_df.iloc[-1].get('trigger_reason', '')
            if 'MOMENTUM_STRENGTHENING_LONG_ONLY' in trigger:
                logger.info(f"Enhanced MACD BUY signal detected - momentum strengthening (long-only Case A.3)")
            else:
                logger.info(f"Enhanced MACD BUY signal detected")
            
        elif enhanced_action == 'SHORT':
            action = "SHORT"
            qty = self.shares_per_trade
            logger.info(f"Enhanced MACD SHORT signal detected")
            
        elif enhanced_action == 'SELL':
            # Long-only mode: Just sell current position, don't short
            if current_side == 'long':
                action = "SELL"
                qty = current_qty
                logger.info(f"Enhanced MACD SELL signal detected (long-only mode)")
            
        elif enhanced_action == 'SELL_AND_SHORT':
            # This is for momentum weakening while holding long
            if current_side == 'long':
                action = "SELL_AND_SHORT"
                qty = current_qty + self.shares_per_trade
                logger.info(f"Enhanced MACD MOMENTUM WEAKENING detected - selling long position and shorting")
            elif current_side == 'none':
                # If no position but Enhanced MACD says SELL_AND_SHORT, just short
                action = "SHORT"
                qty = self.shares_per_trade
                logger.info(f"Enhanced MACD MOMENTUM WEAKENING detected - taking short position")
                
        elif enhanced_action == 'COVER_AND_BUY':
            # This is for momentum strengthening while holding short
            if current_side == 'short':
                action = "COVER_AND_BUY"
                qty = abs(current_qty) + self.shares_per_trade
                logger.info(f"Enhanced MACD MOMENTUM STRENGTHENING detected - covering short position and buying")
            elif current_side == 'none':
                # If no position but Enhanced MACD says COVER_AND_BUY, just buy
                action = "BUY"
                qty = self.shares_per_trade
                logger.info(f"Enhanced MACD MOMENTUM STRENGTHENING detected - taking long position")
        
        # Priority 2: Fallback to Enhanced MACD Case A (No Stock Holding) with crossovers
        elif current_side == 'none':
            if crossover:
                action = "BUY"
                qty = self.shares_per_trade
                logger.info(f"Enhanced MACD Case A: No position + bullish crossover - BUY")
            elif crossunder:
                action = "SHORT" 
                qty = self.shares_per_trade
                logger.info(f"Enhanced MACD Case A: No position + bearish crossunder - SHORT")
            elif macd_position == "ABOVE":
                # No crossover but MACD above signal - initial buy
                action = "BUY"
                qty = self.shares_per_trade
                logger.info(f"Enhanced MACD Case A: No position + MACD above signal - initial BUY")
            elif macd_position == "BELOW":
                # No crossunder but MACD below signal - initial short
                action = "SHORT"
                qty = self.shares_per_trade
                logger.info(f"Enhanced MACD Case A: No position + MACD below signal - initial SHORT")
        
        # Enhanced MACD Failsafe Actions based on trigger reasons
        elif enhanced_action == 'SELL_AND_SHORT' and 'FAILSAFE_CROSSUNDER' in str(self.quote_monitor.quotes_df.iloc[-1].get('trigger_reason', '')):
            # Case B.4 Failsafe exit for long position
            if current_side == 'long':
                action = "SELL_AND_SHORT"
                qty = current_qty + self.shares_per_trade
                logger.info(f"Enhanced MACD Case B.4 Failsafe: MACD fell below Signal Line while long - SELL_AND_SHORT")
                
        elif enhanced_action == 'COVER_AND_BUY' and 'FAILSAFE_CROSSOVER' in str(self.quote_monitor.quotes_df.iloc[-1].get('trigger_reason', '')):
            # Case C.6 Failsafe exit for short position
            if current_side == 'short':
                action = "COVER_AND_BUY"
                qty = abs(current_qty) + self.shares_per_trade
                logger.info(f"Enhanced MACD Case C.6 Failsafe: MACD rose above Signal Line while short - COVER_AND_BUY")
        
        # Priority 3: Traditional failsafe exits for Enhanced MACD (backup logic)
        elif current_side == 'long' and crossunder:
            action = "SELL_AND_SHORT"
            qty = current_qty + self.shares_per_trade
            logger.info(f"Enhanced MACD Case B.4 (backup): Bearish crossunder while long - SELL_AND_SHORT")
            
        elif current_side == 'short' and crossover:
            action = "COVER_AND_BUY"
            qty = abs(current_qty) + self.shares_per_trade
            logger.info(f"Enhanced MACD Case C.6 (backup): Bullish crossover while short - COVER_AND_BUY")
        
        return action, qty
    
    def _process_basic_macd_signals(self, current_qty, current_side, macd_signal):
        """
        Process basic MACD signals using traditional crossover logic.
        Used when strategy_name != 'enhanced_macd'.
        
        Returns:
            tuple: (action, qty)
        """
        action = None
        qty = 0
        
        # Extract signal information
        macd_position = macd_signal['macd_position']
        crossover = macd_signal['crossover']
        crossunder = macd_signal['crossunder']
        
        # No position yet
        if current_side == 'none':
            if macd_position == "ABOVE":
                action = "BUY"
                qty = self.shares_per_trade
                logger.info(f"Basic MACD: No position + MACD above signal - BUY")
            elif macd_position == "BELOW":
                action = "SHORT"
                qty = self.shares_per_trade
                logger.info(f"Basic MACD: No position + MACD below signal - SHORT")
        
        # Currently long
        elif current_side == 'long':
            if crossunder:
                action = "SELL_AND_SHORT"
                qty = current_qty + self.shares_per_trade
                logger.info(f"Basic MACD: Bearish crossunder while long - SELL_AND_SHORT")
            elif macd_position == "BELOW" and self.last_trade_time:
                elapsed_minutes = (datetime.now() - self.last_trade_time).total_seconds() / 60 if self.last_trade_time else 0
                if elapsed_minutes > 15:
                    action = "SELL_AND_SHORT"
                    qty = current_qty + self.shares_per_trade
                    logger.info(f"Basic MACD: MACD below signal while long - SELL_AND_SHORT")
        
        # Currently short
        elif current_side == 'short':
            if crossover:
                action = "COVER_AND_BUY"
                qty = abs(current_qty) + self.shares_per_trade
                logger.info(f"Basic MACD: Bullish crossover while short - COVER_AND_BUY")
            elif macd_position == "ABOVE" and self.last_trade_time:
                elapsed_minutes = (datetime.now() - self.last_trade_time).total_seconds() / 60 if self.last_trade_time else 0
                if elapsed_minutes > 15:
                    action = "COVER_AND_BUY"
                    qty = abs(current_qty) + self.shares_per_trade
                    logger.info(f"Basic MACD: MACD above signal while short - COVER_AND_BUY")
        
        return action, qty
    
    def update_quotes(self):
        """
        Check if we have new quote data from the WebSocket stream.
        With WebSockets, quotes are automatically added to the dataframe
        as they arrive, so we just need to check if we have data.
        """
        # WebSocket already adds quotes to dataframe automatically
        # Just check if we have any quotes
        if not self.quote_monitor.quotes_df.empty:
            return True
        return False
    
    def get_current_position(self):
        """Get the current position from Alpaca."""
        try:
            positions = self.trading_system.trading_client.get_all_positions()
            current_position = next((p for p in positions if p.symbol == self.symbol), None)
            
            if current_position:
                qty = float(current_position.qty)
                side = current_position.side
                
                # Convert to our format
                if side == 'long':
                    self.position_type = 'LONG'
                    self.position_shares = qty
                elif side == 'short':
                    self.position_type = 'SHORT'
                    self.position_shares = -qty
                
                logger.info(f"Current position for {self.symbol}: {abs(qty)} shares, Side: {side}")
                return qty, side
            else:
                self.position_type = 'NONE'
                self.position_shares = 0
                logger.info(f"No current position for {self.symbol}")
                return 0, 'none'
        except Exception as e:
            logger.error(f"Error getting current position: {e}")
            return 0, 'none'
    
    def process_macd_signal(self):
        """
        Process the latest MACD signal and execute trades if necessary.
        Properly handles Enhanced MACD signals when using enhanced_macd strategy.
        
        Returns:
            bool: True if a trade was executed, False otherwise
        """
        # Get the latest MACD signal directly from the quote monitor
        macd_signal = self.quote_monitor.get_macd_signal()
        
        # If we don't have a valid signal yet, do nothing
        if macd_signal['macd_position'] is None:
            logger.info("Not enough data for MACD calculation yet")
            return False
        
        # Get the current position from Alpaca
        current_qty, current_side = self.get_current_position()
        
        # Initialize action and qty
        action = None
        qty = 0
        
        # For Enhanced MACD strategy, check Enhanced MACD signals first
        if self.strategy_name == 'enhanced_macd':
            action, qty = self._process_enhanced_macd_signals(current_qty, current_side, macd_signal)
        else:
            # Fall back to basic MACD logic for classic strategy
            action, qty = self._process_basic_macd_signals(current_qty, current_side, macd_signal)
        
        # Execute the trade if we have an action and warm-up is complete
        if action and qty > 0 and self.is_warmup_complete():
            logger.info(f"Executing trade: {action} {qty} shares of {self.symbol}")
            
            # Import OrderSide from alpaca.trading.enums to avoid the attribute error
            from alpaca.trading.enums import OrderSide
            
            if action == "BUY":
                # Simple buy
                order_id = self.trading_system.place_market_order(
                    self.symbol, qty, OrderSide.BUY, 
                    extended_hours=self.extended_hours
                )
                
            elif action == "SHORT":
                # Simple short
                order_id = self.trading_system.place_market_order(
                    self.symbol, qty, OrderSide.SELL, 
                    extended_hours=self.extended_hours
                )
                
            elif action == "SELL":
                # Long-only mode: Just sell current position
                order_id = self.trading_system.place_market_order(
                    self.symbol, qty, OrderSide.SELL,
                    extended_hours=self.extended_hours
                )
                
            elif action == "COVER_AND_BUY":
                # Cover short and buy long
                # First, cover the short position
                cover_qty = abs(current_qty)
                if cover_qty > 0:
                    cover_order_id = self.trading_system.place_market_order(
                        self.symbol, cover_qty, OrderSide.BUY,
                        extended_hours=self.extended_hours
                    )
                    logger.info(f"Covered {cover_qty} shares of {self.symbol}")
                    
                    # Wait for the cover order to complete
                    time.sleep(2)
                
                # Then buy additional shares
                buy_order_id = self.trading_system.place_market_order(
                    self.symbol, self.shares_per_trade, OrderSide.BUY,
                    extended_hours=self.extended_hours
                )
                
            elif action == "SELL_AND_SHORT":
                # Sell long and short
                # First, sell the long position
                if current_qty > 0:
                    sell_order_id = self.trading_system.place_market_order(
                        self.symbol, current_qty, OrderSide.SELL,
                        extended_hours=self.extended_hours
                    )
                    logger.info(f"Sold {current_qty} shares of {self.symbol}")
                    
                    # Wait for the sell order to complete
                    time.sleep(2)
                
                # Then short additional shares
                short_order_id = self.trading_system.place_market_order(
                    self.symbol, self.shares_per_trade, OrderSide.SELL,
                    extended_hours=self.extended_hours
                )
            
            # Update the last trade time
            self.last_trade_time = datetime.now()
            
            # Update trade state
            self.trading_system.save_strategy_state(self.symbol, self.strategy.name, {
                'position_type': 'LONG' if action in ["BUY", "COVER_AND_BUY"] else 'SHORT',
                'shares': qty if action in ["BUY", "COVER_AND_BUY"] else -qty,
                'last_action': action,
                'last_signal_time': datetime.now().isoformat()
            })
            
            return True
            
        elif action and qty > 0 and not self.is_warmup_complete():
            logger.info(f"Warm-up period not complete - skipping trade: {action} {qty} shares of {self.symbol}")
            
        return False
    
    def run(self):
        """Run the integrated trading system."""
        logger.info(f"Starting integrated MACD trading system for {self.symbol}")
        logger.info(f"Warm-up period: {self.warmup_period_minutes} minutes")
        logger.info(f"Press Ctrl+C to stop trading")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            while self.is_running:
                # Update the quote data
                self.update_quotes()
                
                # Apply Enhanced MACD calculations if using enhanced strategy and we have enough data
                if (self.strategy_name == 'enhanced_macd' and 
                    not self.quote_monitor.quotes_df.empty and 
                    len(self.quote_monitor.quotes_df) >= 30):
                    self._apply_enhanced_macd_to_monitor()
                
                # Display the current status with Enhanced MACD data
                if len(self.quote_monitor.quotes_df) > 0:
                    self.quote_monitor.display_quotes()
                    
                    # Log warmup status
                    if not self.is_warmup_complete():
                        elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60
                        remaining_minutes = max(0, self.warmup_period_minutes - elapsed_minutes)
                        logger.info(f"Warm-up in progress: {elapsed_minutes:.1f}/{self.warmup_period_minutes} minutes elapsed, {remaining_minutes:.1f} minutes remaining")
                    else:
                        # Process MACD signals only if warm-up is complete
                        logger.info(f"Warm-up complete, processing trading signals...")
                        
                        # Log Enhanced MACD details if using enhanced strategy
                        if self.strategy_name == 'enhanced_macd':
                            self._log_enhanced_macd_details()
                        
                        self.process_macd_signal()
                
                # Wait for the next update
                logger.info(f"Waiting {self.interval_seconds} seconds until next update...")
                logger.info("\n" + "-" * 80 + "\n")
                time.sleep(self.interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Integrated trading system stopped by user")
            self.is_running = False
            
            # Save the quote data to CSV
            self.quote_monitor.save_to_csv()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Integrated MACD Trading System")
    
    parser.add_argument(
        "--symbol", 
        type=str, 
        default="NVDA",
        help="Stock symbol to trade (default: NVDA)"
    )
    
    parser.add_argument(
        "--interval", 
        type=int, 
        default=60,
        help="Update interval in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--shares", 
        type=int, 
        default=100,
        help="Number of shares per trade (default: 100)"
    )
    
    parser.add_argument(
        "--fast-window", 
        type=int, 
        default=13,
        help="Fast EMA window for MACD (default: 13)"
    )
    
    parser.add_argument(
        "--slow-window", 
        type=int, 
        default=21,
        help="Slow EMA window for MACD (default: 21)"
    )
    
    parser.add_argument(
        "--signal-window", 
        type=int, 
        default=9,
        help="Signal line window for MACD (default: 9)"
    )
    
    parser.add_argument(
        "--extended-hours",
        action="store_true",
        help="Enable trading during extended hours"
    )
    
    parser.add_argument(
        "--warmup",
        type=int,
        default=60,
        help="Warm-up period in minutes before trading begins (default: 60)"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        default="macd",
        choices=["macd", "enhanced_macd"],
        help="MACD strategy to use: 'macd' (classic) or 'enhanced_macd' (advanced with slope/histogram analysis) (default: macd)"
    )
    
    parser.add_argument(
        "--slope-threshold",
        type=float,
        default=0.001,
        help="MACD slope threshold for Enhanced MACD strategy (default: 0.001)"
    )
    
    parser.add_argument(
        "--slope-lookback",
        type=int,
        default=3,
        help="Slope calculation lookback period for Enhanced MACD (default: 3)"
    )
    
    parser.add_argument(
        "--histogram-lookback",
        type=int,
        default=3,
        help="Histogram averaging period for Enhanced MACD (default: 3)"
    )
    
    parser.add_argument(
        "--long-only",
        action="store_true",
        help="Enable long-only mode (no short selling) to avoid day trading buying power issues"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and run the integrated trading system
    trader = IntegratedMACDTrader(
        symbol=args.symbol,
        interval_seconds=args.interval,
        fast_window=args.fast_window,
        slow_window=args.slow_window,
        signal_window=args.signal_window,
        shares_per_trade=args.shares,
        extended_hours=args.extended_hours,
        warmup_period_minutes=args.warmup,
        strategy_name=args.strategy,
        slope_threshold=args.slope_threshold,
        slope_lookback=args.slope_lookback,
        histogram_lookback=args.histogram_lookback,
        long_only=args.long_only
    )
    
    # Run the trader
    trader.run()