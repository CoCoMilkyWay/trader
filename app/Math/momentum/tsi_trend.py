import array

class tsi_trend:
    def __init__(self,
                 closes: array.array,
                 period_ma: array.array,    # Pre-calculated MA for the period
                 period: int = 14           # Correlation period
                 ):
        self.tsi = array.array('d', [])    # TSI values
        
        self.closes = closes
        self.period_ma = period_ma         # Moving average of prices
        self.period = period
        
        # Store only necessary data points
        self.previous_tsi = None
        self.data_points = 0
        
    def update(self):
        """
        Calculate TSI using Pearson correlation between price and bar index:
        - Uses pre-calculated MA for price mean
        - Creates index series [0,1,2,...] for the period
        - Calculates correlation between price and index
        """
        self.data_points += 1
        
        # Need enough data for the period
        if self.data_points < self.period:
            self.tsi.append(0)
            return
            
        # Get price data for the period
        period_prices = self.closes[-self.period:]
        
        # Create bar index [0,1,2,...,period-1]
        bar_index = array.array('d', range(self.period))
        
        # Get price mean from MA
        price_mean = self.period_ma[-1]
        index_mean = (self.period - 1) / 2  # Mean of [0,1,2,...,period-1]
        
        # Calculate correlation components
        numerator = 0
        price_variance = 0
        index_variance = 0
        
        for i in range(self.period):
            price_diff = period_prices[i] - price_mean
            index_diff = bar_index[i] - index_mean
            
            numerator += price_diff * index_diff
            price_variance += price_diff * price_diff
            index_variance += index_diff * index_diff
        
        # Calculate correlation
        if price_variance == 0 or index_variance == 0:
            tsi_value = 0
        else:
            tsi_value = numerator / (price_variance * index_variance)**0.5
            
        self.tsi.append(tsi_value)
        self.previous_tsi = tsi_value
        LEN = 100
        if len(self.tsi) > 2*LEN:
            del self.tsi[:-LEN]
        return