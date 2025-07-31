import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TradeLab:
    def __init__(self, symbols, mode='historical', start_date=None, end_date=None):
        self.symbols = symbols
        self.mode = mode
        self.interval = '5m'  # 5-minute intervals (300 seconds)
        self.current_step = 0
        self.data = {}
        self.timestamps = []
        self.portfolio_history = {sym: [] for sym in symbols}
        
        if mode == 'historical':
            self.start_date = start_date
            self.end_date = end_date
            self._load_historical_data()
        else:
            # Live mode would be implemented differently
            raise NotImplementedError("Live trading not implemented in this version")
    
    def _load_historical_data(self):
        """Load historical market data using yfinance"""
        print(f"📈 Loading historical data for {self.symbols} from {self.start_date} to {self.end_date}")
        
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                prepost=True
            )
            # Clean and prepare data
            data = data.dropna()
            data['Returns'] = data['Close'].pct_change()
            data['MA_10'] = data['Close'].rolling(window=10).mean()
            data['MA_50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            
            self.data[symbol] = data
            self.timestamps = data.index.tolist()
        
        # Initialize benchmark (S&P 500 equivalent)
        spy = yf.Ticker("SPY").history(
            start=self.start_date,
            end=self.end_date,
            interval=self.interval
        )['Close']
        self.benchmark = (spy / spy.iloc[0]).tolist()
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index (RSI)"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def get_current_state(self, symbol):
        """Get current market state for a symbol"""
        if self.current_step >= len(self.timestamps):
            return None
        
        current_data = self.data[symbol].iloc[self.current_step]
        return {
            'symbol': symbol,
            'timestamp': self.timestamps[self.current_step],
            'open': current_data['Open'],
            'high': current_data['High'],
            'low': current_data['Low'],
            'close': current_data['Close'],
            'volume': current_data['Volume'],
            'returns': current_data['Returns'],
            'ma_10': current_data['MA_10'],
            'ma_50': current_data['MA_50'],
            'rsi': current_data['RSI']
        }
    
    def execute_trade(self, symbol, action, shares, current_price):
        """Execute a trade and return transaction details"""
        # In training mode, we just simulate execution at current price
        # Real implementation would have slippage, fees, etc.
        cost = shares * current_price
        return {
            'symbol': symbol,
            'action': action,
            'shares': shares,
            'price': current_price,
            'cost': cost,
            'timestamp': self.timestamps[self.current_step]
        }
    
    def step(self):
        """Move to next time interval"""
        if self.current_step < len(self.timestamps) - 1:
            self.current_step += 1
            return True
        return False
    
    def get_benchmark_performance(self):
        """Get S&P 500 benchmark performance"""
        return self.benchmark[:self.current_step+1]
    
    def get_current_time(self):
        """Get current simulation time"""
        return self.timestamps[self.current_step]