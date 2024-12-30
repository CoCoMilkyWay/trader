import array

class rsi:
    def __init__(self,
                 closes: array.array,
                 period: int = 14):
        self.rsi = array.array('d', [])
        self.closes = closes
        self._period = period
        
        # Store only necessary data for calculations
        self.prev_avg_gain = 0.0
        self.prev_avg_loss = 0.0
        self.data_points = 0
        self.prev_close = None
        
    def update(self):
        close_price = self.closes[-1]
        self.data_points += 1
        
        # Need at least 2 data points to calculate price change
        if self.prev_close is None:
            self.prev_close = close_price
            self.rsi.append(0)  # No RSI value for first point
            return
            
        # Calculate current gain/loss
        change = close_price - self.prev_close
        current_gain = max(change, 0)
        current_loss = abs(min(change, 0))
        
        # Initial period: calculate simple averages
        if self.data_points <= self._period:
            # Use weighted approach for initial values
            self.prev_avg_gain = (self.prev_avg_gain * (self.data_points - 1) + current_gain) / self.data_points
            self.prev_avg_loss = (self.prev_avg_loss * (self.data_points - 1) + current_loss) / self.data_points
        else:
            # Calculate smoothed averages
            self.prev_avg_gain = (self.prev_avg_gain * (self._period - 1) + current_gain) / self._period
            self.prev_avg_loss = (self.prev_avg_loss * (self._period - 1) + current_loss) / self._period
        
        # Calculate RSI
        if self.prev_avg_loss == 0:
            rsi_value = 100
        else:
            rs = self.prev_avg_gain / self.prev_avg_loss
            rsi_value = 100 - (100 / (1 + rs))
            
        self.rsi.append(rsi_value)
        self.prev_close = close_price
        LEN = 100
        if len(self.rsi) > 2*LEN:
            del self.rsi[:-LEN]
        return