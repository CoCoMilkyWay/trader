import array

class williams_r:
    def __init__(self,
                 highs: array.array,
                 lows: array.array,
                 closes: array.array,
                 period: int = 14     # Default lookback period
                 ):
        # Initialize indicator array
        self.wpr = array.array('d', [])
        
        # Store inputs
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.period = int(period)
        
        # Initialize previous value
        self.previous_wpr = 0.0

    def update(self):
        """
        Updates Williams %R indicator by calculating:
        (Highest High - Close) / (Highest High - Lowest Low) * -100
        
        Key Properties:
        - Oscillates between 0 and -100
        - -80 to -100: Oversold region
        - -20 to 0: Overbought region
        """
        # Get the period data
        period_highs = self.highs[-self.period:]
        period_lows = self.lows[-self.period:]
        
        # Calculate highest high and lowest low
        highest_high = max(period_highs)
        lowest_low = min(period_lows)
        
        # Get current close
        close = self.closes[-1]
        
        # Avoid division by zero
        denominator = highest_high - lowest_low
        if denominator == 0:
            wpr_value = 0.0
        else:
            # Calculate Williams %R value
            wpr_value = ((highest_high - close) / denominator) * -100.0
            
        self.wpr.append(float(wpr_value))
        self.previous_wpr = wpr_value
        
        # Maintain fixed length
        LEN = 200
        if len(self.wpr) > 2*LEN:
            del self.wpr[:-LEN]