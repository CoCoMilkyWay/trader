# implement avwap(rolling vwap) and deviationa multiplier (close-avwap)/atr indicator as follows, 
# feel free to assume ma/ema(of close), rsi... and others are already implemented and updated
# as arraies just as OHLCV
# leave param configurable
# do not store intermediate values as array, only the final indicators

import array
class macd:
    def __init__(self,
                 closes: array.array,
                 fast_ema: array.array,    
                 slow_ema: array.array,    
                 signal_period: int = 9     # Explicitly type as int
                 ):
        # Initialize arrays
        self.histogram = array.array('d', [])
        self.macd = array.array('d', [])     
        self.signal = array.array('d', [])    
        
        # Store inputs
        self.closes = closes
        self.fast_ema = fast_ema    
        self.slow_ema = slow_ema    
        self.signal_period = int(signal_period)  # Ensure it's an int
        
        # Initialize previous values
        self.previous_histogram = 0.0
        self.previous_macd = 0.0
        self.previous_signal = 0.0
        self.data_points = 0
        
    def update(self):
        """
        Updates the MACD indicator by calculating new values for:
        1. MACD line (fast EMA - slow EMA)
        2. Signal line (EMA of MACD)
        3. MACD histogram (MACD - Signal)
        Key Insights:
        - MACD line measures momentum by comparing recent price changes (fast EMA) 
          to longer-term trend (slow EMA)
        - Signal line smooths MACD using EMA formula: 
          EMA = Price * α + Previous_EMA * (1-α), where α = 2/(period+1)
        - Histogram shows momentum strength and potential reversals:
          - Positive: MACD above signal = bullish momentum
          - Negative: MACD below signal = bearish momentum  
          - Zero crossing = potential trend change
        """
        self.data_points += 1
        
        # Calculate MACD line
        macd_value = float(self.fast_ema[-1] - self.slow_ema[-1])
        self.macd.append(macd_value)
        self.previous_macd = macd_value
        
        # Calculate signal line as EMA of MACD
        signal_period_int = int(self.signal_period)  # Ensure integer
        alpha = 2.0 / (signal_period_int + 1)
        
        if not self.signal:  # First value
            signal_value = float(macd_value)
        else:
            signal_value = float(macd_value * alpha + self.signal[-1] * (1 - alpha))
            
        self.signal.append(signal_value)
        self.previous_signal = signal_value
        
        # Calculate histogram
        histogram = float(macd_value - signal_value)
        self.previous_histogram = histogram
        
        self.histogram.append(histogram)
        
        # Maintain fixed length
        LEN = 100
        if len(self.histogram) > 2*LEN:
            del self.histogram[:-LEN]
            del self.macd[:-LEN]
            del self.signal[:-LEN]