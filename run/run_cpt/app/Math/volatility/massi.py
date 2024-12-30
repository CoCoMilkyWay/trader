import array

class massi:
    def __init__(self, 
                 highs: array.array,
                 lows: array.array,
                 ema_period: int = 9,     # Period for the first EMA
                 sum_period: int = 25     # Period for summing the ratio
                 ):
        # Initialize arrays
        self.mass_index = array.array('d', [])
        
        # Store inputs
        self.highs = highs
        self.lows = lows
        self.ema_period = int(ema_period)
        self.sum_period = int(sum_period)
        
        # Initialize previous values
        self.previous_single_ema = 0.0
        self.previous_double_ema = 0.0
        self.previous_ratios = []  # Store recent ratios for summation
        self.previous_mass_index = 0.0
        self.data_points = 0
        
    def update(self):
        self.data_points += 1
        
        # Calculate current price range
        current_range = float(self.highs[-1] - self.lows[-1])
        
        # Calculate first EMA (single EMA)
        alpha = 2.0 / (self.ema_period + 1)
        if self.data_points == 1:
            single_ema = current_range
        else:
            single_ema = current_range * alpha + self.previous_single_ema * (1 - alpha)
        
        # Calculate second EMA (double EMA)
        if self.data_points == 1:
            double_ema = single_ema
        else:
            double_ema = single_ema * alpha + self.previous_double_ema * (1 - alpha)
            
        # Calculate ratio
        if double_ema != 0:  # Prevent division by zero
            ema_ratio = single_ema / double_ema
        else:
            ema_ratio = 1.0
            
        # Update previous ratios list
        self.previous_ratios.append(ema_ratio)
        if len(self.previous_ratios) > self.sum_period:
            self.previous_ratios.pop(0)
            
        # Calculate Mass Index
        mass_index_value = float(sum(self.previous_ratios))
        self.mass_index.append(mass_index_value)
        
        # Update previous values
        self.previous_single_ema = single_ema
        self.previous_double_ema = double_ema
        self.previous_mass_index = mass_index_value
        
        # Maintain fixed length
        LEN = 100
        if len(self.mass_index) > 2*LEN:
            del self.mass_index[:-LEN]
        return