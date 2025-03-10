import array
from math import sqrt

class rvi:
    """Relative Vigor Index measures the strength of a trend by comparing the closing price to the trading range over a specific period"""
    def __init__(self,
                 closes: array.array,
                 highs: array.array,
                 lows: array.array,
                 period: int = 10,      # Standard deviation and smoothing period
                 std_window: int = 7    # Window for standard deviation calculation
                 ):
        # Initialize arrays
        self.rvi = array.array('d', [])
        
        # Store inputs
        self.closes = closes
        self.highs = highs
        self.lows = lows
        self.period = int(period)
        self.std_window = int(std_window)
        
        # Initialize previous values
        self.previous_upward = 0.0
        self.previous_downward = 0.0
        self.previous_close = 0.0
        self.data_points = 0
        
        # Store recent values for std calculation
        self.recent_prices = []
        
    def _calculate_std(self, prices):
        """Calculate standard deviation for the given price window"""
        if len(prices) < 2:
            return 0.0
            
        mean = sum(prices) / len(prices)
        squared_diff_sum = sum((x - mean) ** 2 for x in prices)
        return sqrt(squared_diff_sum / (len(prices) - 1))
        
    def update(self):
        # measure volatility direction/momentum, not magnitude
        self.data_points += 1
        current_close = float(self.closes[-1])
        
        # Update recent prices for std calculation
        self.recent_prices.append(current_close)
        if len(self.recent_prices) > self.std_window:
            self.recent_prices.pop(0)
            
        # Calculate standard deviation
        current_std = self._calculate_std(self.recent_prices)
        
        # Determine if volatility is upward or downward based on close direction
        if self.data_points > 1:
            if current_close > self.previous_close:
                upward = current_std
                downward = 0.0
            else:
                upward = 0.0
                downward = current_std
        else:
            upward = downward = 0.0
            
        # Calculate smoothed upward and downward volatility
        alpha = 1.0 / self.period
        
        if self.data_points == 1:
            smoothed_upward = upward
            smoothed_downward = downward
        else:
            smoothed_upward = upward * alpha + self.previous_upward * (1 - alpha)
            smoothed_downward = downward * alpha + self.previous_downward * (1 - alpha)
            
        # Calculate RVI
        if (smoothed_upward + smoothed_downward) != 0:
            rvi_value = float(100 * smoothed_upward / (smoothed_upward + smoothed_downward))
        else:
            rvi_value = 50.0  # Default to neutral when no volatility
            
        self.rvi.append(rvi_value)
        
        # Update previous values
        self.previous_upward = smoothed_upward
        self.previous_downward = smoothed_downward
        self.previous_close = current_close
        
        # Maintain fixed length
        LEN = 100
        if len(self.rvi) > 2*LEN:
            del self.rvi[:-LEN]
            
        return rvi_value