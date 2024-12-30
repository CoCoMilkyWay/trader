import array
class ema:
    def __init__(self,
                 # opens  :array.array,
                 # highs  :array.array,
                 # lows   :array.array,
                 closes :array.array,
                 # volumes:array.array,
                 period,
                 smoothing=2,
                 ):
        self.ema = array.array('d', [])
        
        # self.opens   = opens
        self.closes  = closes
        # self.highs   = highs
        # self.lows    = lows
        # self.volumes = volumes
        
        self._period = period
        self._smoothing = smoothing
        self._multiplier = smoothing / (period + 1)
        
        # Store only necessary data points
        self.previous_ema = None
        self.data_points = 0
        
    def update(self):
        close_price = self.closes[-1]  # Close price is at index 3
        self.data_points += 1
        
        # Handle initial SMA calculation period
        if self.data_points <= self._period:
            if self.previous_ema is None:
                ema = close_price
            else:
                # Use weighted approach for initial value
                ema = (self.previous_ema * (self.data_points - 1) + close_price) / self.data_points
        else:
            # Calculate EMA
            ema = (close_price - self.previous_ema) * self._multiplier + self.previous_ema
        self.previous_ema = ema
        
        self.ema.append(ema)
        LEN = 100
        if len(self.ema) > 2*LEN:
            del self.ema[:-LEN]
        return