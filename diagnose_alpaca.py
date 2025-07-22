#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alpaca Data Connection Diagnostic Tool

This script helps diagnose issues with Alpaca data reception and WebSocket connections.
"""

import os
import sys
import time
import logging
from datetime import datetime
import pytz
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run comprehensive Alpaca diagnostics."""
    
    print("=" * 60)
    print("🔍 ALPACA DATA CONNECTION DIAGNOSTIC TOOL")
    print("=" * 60)
    print()
    
    # Load environment variables
    load_dotenv()
    
    # 1. Check Environment Configuration
    print("📋 STEP 1: Environment Configuration")
    print("-" * 40)
    
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')
    base_url = os.getenv('ALPACA_BASE_URL')
    data_source = os.getenv('DATA_SOURCE', 'ALPACA').upper()
    alpaca_price_type = os.getenv('ALPACA_PRICE_TYPE', 'MID').upper()
    
    print(f"ALPACA_API_KEY: {'✅ SET (' + api_key[:10] + '...)' if api_key else '❌ NOT SET'}")
    print(f"ALPACA_API_SECRET: {'✅ SET' if api_secret else '❌ NOT SET'}")
    print(f"ALPACA_BASE_URL: {base_url if base_url else '⚠️  NOT SET (will use default)'}")
    print(f"DATA_SOURCE: {data_source}")
    print(f"ALPACA_PRICE_TYPE: {alpaca_price_type}")
    
    if not api_key or not api_secret:
        print("\n❌ CRITICAL: Missing API credentials!")
        print("Please set ALPACA_API_KEY and ALPACA_API_SECRET in your .env file")
        return False
    
    print()
    
    # 2. Market Hours Check
    print("🕐 STEP 2: Market Hours Check")
    print("-" * 40)
    
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    
    print(f"Current Time (ET): {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Day of Week: {now_et.strftime('%A')}")
    
    weekday = now_et.weekday()  # 0=Monday, 6=Sunday
    hour = now_et.hour
    
    if weekday >= 5:  # Saturday or Sunday
        status = "❌ WEEKEND - Market Closed"
    elif 9 <= hour <= 16:  # Regular market hours
        status = "✅ REGULAR HOURS - Market Open"
    elif 4 <= hour < 9 or 16 < hour <= 20:  # Extended hours
        status = "⚠️  EXTENDED HOURS - Limited Data"
    else:
        status = "❌ MARKET CLOSED"
    
    print(f"Market Status: {status}")
    print()
    
    # 3. Test Historical API
    print("📊 STEP 3: Testing Historical Data API")
    print("-" * 40)
    
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from datetime import timedelta
        
        client = StockHistoricalDataClient(api_key, api_secret)
        
        # Test with recent data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        request = StockBarsRequest(
            symbol_or_symbols=['NVDA'],
            timeframe=TimeFrame.Minute,
            start=start_time,
            end=end_time,
            feed='iex'
        )
        
        bars = client.get_stock_bars(request)
        
        if 'NVDA' in bars.df.index.get_level_values(0):
            df = bars.df.loc['NVDA']
            latest_bar = df.iloc[-1] if len(df) > 0 else None
            
            print(f"✅ Historical API Working")
            print(f"   Bars Retrieved: {len(df)}")
            if latest_bar is not None:
                print(f"   Latest Bar Time: {df.index[-1]}")
                print(f"   Latest Price: ${latest_bar['close']:.2f}")
        else:
            print("⚠️  No historical data found for NVDA")
            
    except Exception as e:
        print(f"❌ Historical API Error: {e}")
        return False
    
    print()
    
    # 4. Test WebSocket Connection
    print("🔌 STEP 4: Testing WebSocket Connection")
    print("-" * 40)
    
    try:
        from enhanced_quote_monitor import EnhancedQuoteMonitor
        
        # Create monitor with short test period
        print("Creating Enhanced Quote Monitor...")
        monitor = EnhancedQuoteMonitor('NVDA', max_records=10, interval_seconds=5)
        
        # Wait for connection and data
        print("Waiting 10 seconds for WebSocket data...")
        time.sleep(10)
        
        # Check results
        diagnosis = monitor.diagnose_connection()
        
        print(f"Connection Status: {diagnosis['connection_status']}")
        print(f"Quotes Received: {diagnosis['quotes_received']}")
        print(f"Last Quote Time: {diagnosis['last_quote_time']}")
        
        if diagnosis['quotes_received'] > 0:
            print("✅ WebSocket Connection Working!")
            
            # Show latest quote
            latest = monitor.quotes_df.iloc[-1]
            print(f"   Latest Quote: Bid ${latest['bid']:.2f}, Ask ${latest['ask']:.2f}")
        else:
            print("⚠️  No data received via WebSocket")
            
        # Print recommendations
        if diagnosis['recommendations']:
            print("\n🔧 Recommendations:")
            for rec in diagnosis['recommendations']:
                print(f"   • {rec}")
        
        monitor.stop()
        
    except Exception as e:
        print(f"❌ WebSocket Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # 5. Test Alternative Data Source
    if data_source == 'ALPACA':
        print("🔄 STEP 5: Testing Yahoo Finance Fallback")
        print("-" * 40)
        
        try:
            # Temporarily switch to Yahoo Finance
            os.environ['DATA_SOURCE'] = 'YAHOO'
            
            from yahoo_quote_monitor import YahooQuoteMonitor
            
            print("Testing Yahoo Finance data source...")
            yahoo_monitor = YahooQuoteMonitor('NVDA', max_records=5, interval_seconds=5)
            
            # Fetch some data
            time.sleep(3)
            
            if len(yahoo_monitor.quotes_df) > 0:
                print("✅ Yahoo Finance working as backup")
                latest = yahoo_monitor.quotes_df.iloc[-1]
                print(f"   Latest Yahoo Quote: ${latest['mid']:.2f}")
            else:
                print("⚠️  Yahoo Finance also not receiving data")
            
            # Restore original setting
            os.environ['DATA_SOURCE'] = data_source
            
        except Exception as e:
            print(f"❌ Yahoo Finance Test Error: {e}")
    
    print()
    print("=" * 60)
    print("✅ DIAGNOSTIC COMPLETE")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Diagnostic interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)