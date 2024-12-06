class ChandelierIndicator:
    def __init__(self, length=22, atr_period=22, mult=3):
        self.length = length
        self.atr_period = atr_period
        self.mult = mult
        
        # Price histories
        self.highs = []
        self.lows = []
        self.closes = []
        self.tr = []
        
        # State variables
        self.prev_shortvs = None
        self.prev_longvs = None
        self.prev_direction = 0
        self.prev_close = None
    
    def update(self, high:float, low:float, close:float):
        # Update price histories
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        # Keep only needed history
        max_period = max(self.length, self.atr_period)
        if len(self.highs) > max_period:
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)
        
        if len(self.highs) < 2:
            return False, False
            
        # Calculate TR
        tr = max(
            high - low,
            abs(high - self.prev_close) if self.prev_close else 0,
            abs(low - self.prev_close) if self.prev_close else 0
        )
        self.tr.append(tr)
        if len(self.tr) > self.atr_period:
            self.tr.pop(0)
            
        if len(self.highs) < max_period:
            return False, False
            
        # Calculate ATR
        atr = sum(self.tr) / len(self.tr)
        
        # Calculate stops
        highest = max(self.highs[-self.length:])
        lowest = min(self.lows[-self.length:])
        
        short_stop = lowest + self.mult * atr
        long_stop = highest - self.mult * atr
        
        # Update stops
        shortvs = short_stop if self.prev_shortvs is None else \
                 short_stop if close > self.prev_shortvs else \
                 min(short_stop, self.prev_shortvs)
                 
        longvs = long_stop if self.prev_longvs is None else \
                long_stop if close < self.prev_longvs else \
                max(long_stop, self.prev_longvs)
        
        # Calculate switches
        long_switch = (close >= self.prev_shortvs and 
                      self.prev_close < self.prev_shortvs) if self.prev_shortvs else False
        short_switch = (close <= self.prev_longvs and 
                       self.prev_close > self.prev_longvs) if self.prev_longvs else False
        
        # Calculate direction
        direction = (1 if self.prev_direction <= 0 and long_switch else
                    -1 if self.prev_direction >= 0 and short_switch else
                    self.prev_direction)
        
        # Update state
        self.prev_shortvs = shortvs
        self.prev_longvs = longvs
        self.prev_direction = direction
        self.prev_close = close
        
        return direction

# # Example usage
# indicator = ChandelierIndicator()
# 
# # Feed bar data one at a time
# bar = {'high': 100, 'low': 98, 'close': 99}
# result = indicator.update(bar)
# print(result)