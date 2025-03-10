import array
class stoch_rsi:
    def __init__(self,
                 closes: array.array,
                 rsi: array.array,          # Pre-calculated RSI values
                 k_period: int = 3,         # Period for %K line
                 d_period: int = 3,         # Period for %D line (signal)
                 ):
        self.stoch_rsi = array.array('d', [])    # %K line
        self.histogram = array.array('d', [])    # %D line
        
        self.closes = closes
        self.rsi = rsi
        self.k_period = k_period
        self.d_period = d_period
        
        # Store only necessary data points
        self.previous_k = None
        self.previous_d = None
        self.data_points = 0
        
    def update(self):
        """
        Calculate Stochastic RSI:
        1. StochRSI = (RSI - Lowest RSI) / (Highest RSI - Lowest RSI)
        2. Signal = SMA of StochRSI
        """
        self.data_points += 1
        
        # Need enough data points for calculation
        if self.data_points < self.k_period:
            self.stoch_rsi.append(0)
            self.histogram.append(0)
            return
        
        # Get RSI values for the period
        period_rsi = self.rsi[-self.k_period:]
        
        # Find highest and lowest RSI values in the period
        highest_rsi = max(period_rsi)
        lowest_rsi = min(period_rsi)
        
        # Calculate Stochastic RSI (K value)
        if highest_rsi - lowest_rsi == 0:
            k_value = 0  # or could be 100, depending on preference
        else:
            k_value = (self.rsi[-1] - lowest_rsi) / (highest_rsi - lowest_rsi) * 100
            
        self.stoch_rsi.append(k_value)
        self.previous_k = k_value
        
        # Calculate Signal line (D value) - Simple moving average of K
        if self.data_points >= self.k_period + self.d_period:
            d_period_values = self.stoch_rsi[-self.d_period:]
            d_value = sum(d_period_values) / self.d_period
        else:
            d_value = k_value  # Use K value for initial periods
            
        self.histogram.append(k_value-d_value)
        self.previous_d = d_value
        LEN = 100
        if len(self.stoch_rsi) > 2*LEN:
            del self.stoch_rsi[:-LEN]
            del self.histogram[:-LEN]
        return