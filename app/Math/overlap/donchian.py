import array

class donchian:
    def __init__(self,
                 highs: array.array,
                 lows: array.array,
                 period: int = 20    # Default period is 20
                 ):
        self.highs = highs
        self.lows = lows
        self.period = period
        
        # Initialize arrays for final values
        self.upper_band = array.array('d', [])  # Highest high
        self.middle_band = array.array('d', [])  # Average of upper and lower
        self.lower_band = array.array('d', [])  # Lowest low
        self.data_points = 0
        
    def update(self):
        """
        Calculate Donchian Channels:
        Upper Band = Highest high over period
        Lower Band = Lowest low over period
        Middle Band = (Upper Band + Lower Band) / 2
        """
        self.data_points += 1
        
        # Need enough data for the period
        if self.data_points <= self.period:
            current_high = self.highs[-1]
            current_low = self.lows[-1]
            middle = (current_high + current_low) / 2.0
            
            self.upper_band.append(current_high)
            self.lower_band.append(current_low)
            self.middle_band.append(middle)
            return
        
        # Get highest high and lowest low over period
        period_highs = self.highs[-self.period:]
        period_lows = self.lows[-self.period:]
        
        upper = max(period_highs)
        lower = min(period_lows)
        middle = (upper + lower) / 2.0
        
        self.upper_band.append(upper)
        self.lower_band.append(lower)
        self.middle_band.append(middle)
        LEN = 100
        if len(self.upper_band) > 2*LEN:
            del self.upper_band[:-LEN]
            del self.middle_band[:-LEN]
            del self.lower_band[:-LEN]
        return