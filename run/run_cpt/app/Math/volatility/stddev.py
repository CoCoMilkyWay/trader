import array
import math
class stddev:
    def __init__(self,
                 closes: array.array,
                 ma: array.array,    # Pre-calculated MA for the same period
                 period: int = 20,
                 ):
        self.stddev = array.array('d', [])
        
        self.closes = closes
        self.period = period
        self.ma = ma    # Moving average array
        
        # Store only necessary data points
        self.previous_stddev = None
        self.data_points = 0
        
    def update(self):
        """
        Calculate Standard Deviation:
        1. Get the mean (MA) for the period
        2. Calculate sum of squared differences from mean
        3. Take square root of average squared difference
        """
        self.data_points += 1
        
        # Need enough data points for calculation
        if self.data_points < self.period:
            self.stddev.append(0)
            return
            
        # Get price values for the period
        period_prices = self.closes[-self.period:]
        
        # Get mean value for the period
        mean = self.ma[-1]
        
        # Calculate sum of squared differences
        squared_diff_sum = sum((price - mean) ** 2 for price in period_prices)
        
        # Calculate standard deviation
        stddev_value = math.sqrt(squared_diff_sum / self.period)
        self.previous_stddev = stddev_value
        
        self.stddev.append(stddev_value)
        return