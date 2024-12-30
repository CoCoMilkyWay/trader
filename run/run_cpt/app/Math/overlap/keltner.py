import array

class keltner:
    def __init__(self,
                 ema: array.array,     # Pre-calculated EMA
                 atr: array.array,     # Pre-calculated ATR
                 multiplier: float = 2.0  # Default ATR multiplier is 2
                 ):
        self.ema = ema
        self.atr = atr
        self.multiplier = multiplier
        
        # Initialize arrays for final values
        self.upper_band = array.array('d', [])  # EMA + (multiplier * ATR)
        # self.middle_band = array.array('d', []) # EMA
        self.lower_band = array.array('d', [])  # EMA - (multiplier * ATR)
        self.data_points = 0
        
    def update(self):
        """
        Calculate Keltner Channels:
        Middle Band = EMA
        Upper Band = EMA + (multiplier * ATR)
        Lower Band = EMA - (multiplier * ATR)
        """
        self.data_points += 1
        
        # Get latest values
        current_ema = self.ema[-1]
        current_atr = self.atr[-1]
        
        # Calculate bands
        middle = current_ema
        upper = middle + (self.multiplier * current_atr)
        lower = middle - (self.multiplier * current_atr)
        
        # self.middle_band.append(middle)
        self.upper_band.append(upper)
        self.lower_band.append(lower)
        LEN = 100
        if len(self.upper_band) > 2*LEN:
            del self.upper_band[:-LEN]
            del self.lower_band[:-LEN]
        return