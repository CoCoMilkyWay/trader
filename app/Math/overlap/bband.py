import array

class bband:
    def __init__(self,
                 ma: array.array,      # Pre-calculated Moving Average
                 stddev: array.array,   # Pre-calculated Standard Deviation
                 num_std: float = 2.0   # Default number of standard deviations is 2
                 ):
        self.ma = ma
        self.stddev = stddev
        self.num_std = num_std
        
        # Initialize arrays for final values
        self.upper_band = array.array('d', [])
        # self.middle_band = array.array('d', [])
        self.lower_band = array.array('d', [])
        self.data_points = 0
        
    def update(self):
        """
        Calculate Bollinger Bands using pre-calculated MA and StdDev:
        Middle Band = MA
        Upper Band = MA + (K × σ)
        Lower Band = MA - (K × σ)
        """
        self.data_points += 1
        
        # Get latest MA and StdDev values
        current_ma = self.ma[-1]
        current_stddev = self.stddev[-1]
        
        # Calculate bands
        # middle = current_ma
        upper = current_ma + (self.num_std * current_stddev)
        lower = current_ma - (self.num_std * current_stddev)
        
        # self.middle_band.append(middle)
        self.upper_band.append(upper)
        self.lower_band.append(lower)
        LEN = 100
        if len(self.upper_band) > 2*LEN:
            del self.upper_band[:-LEN]
            del self.lower_band[:-LEN]
        return