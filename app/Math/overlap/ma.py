import array
class ma:
    def __init__(self,
                 closes: array.array,
                 period: int = 20
                 ):
        self.ma = array.array('d', [])
        
        self.closes = closes
        self.period = period
        
        # Store only necessary data points
        self.previous_ma = None
        self.data_points = 0
        
    def update(self):
        """
        Calculate Simple Moving Average:
        MA = sum(prices) / period
        """
        self.data_points += 1
        
        # Handle initial periods
        if self.data_points < self.period:
            # Use available data points
            ma_value = sum(self.closes[-self.data_points:]) / self.data_points
        else:
            # Use full period
            ma_value = sum(self.closes[-self.period:]) / self.period
            
        self.ma.append(ma_value)
        self.previous_ma = ma_value
        LEN = 100
        if len(self.ma) > 2*LEN:
            del self.ma[:-LEN]
        return