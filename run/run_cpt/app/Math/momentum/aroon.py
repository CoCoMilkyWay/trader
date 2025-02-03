import array

class aroon:
    def __init__(self,
                 highs: array.array,
                 lows: array.array,
                 period: int = 25    # Standard period is 25
                 ):
        # Initialize arrays for indicator values
        self.aroon_up = array.array('d', [])
        self.aroon_down = array.array('d', [])
        
        # Store inputs
        self.highs = highs
        self.lows = lows
        self.period = int(period)  # Ensure it's an int
        
        # Initialize previous values
        self.previous_aroon_up = 0.0
        self.previous_aroon_down = 0.0
        self.data_points = 0
        
    def update(self):
        """
        Updates the Aroon indicator by calculating:
        1. Aroon Up = ((period - days since highest high) / period) * 100
        2. Aroon Down = ((period - days since lowest low) / period) * 100
        
        Key Insights:
        - Values range from 0 to 100
        - High Aroon Up (>70) indicates strong uptrend
        - High Aroon Down (>70) indicates strong downtrend
        - Crossovers can signal trend reversals
        - Values near 50 suggest consolidation/weak trend
        """
        self.data_points += 1
        
        if self.data_points < self.period:
            # Not enough data points yet
            self.aroon_up.append(0.0)
            self.aroon_down.append(0.0)
            return
            
        # Find highest high and lowest low positions in lookback period
        lookback_highs = self.highs[-self.period:]
        lookback_lows = self.lows[-self.period:]
        
        highest_price = max(lookback_highs)
        lowest_price = min(lookback_lows)
        
        # Find days since highest high and lowest low
        days_since_high = 0
        days_since_low = 0
        
        for i in range(len(lookback_highs) - 1, -1, -1):
            if lookback_highs[i] == highest_price:
                break
            days_since_high += 1
            
        for i in range(len(lookback_lows) - 1, -1, -1):
            if lookback_lows[i] == lowest_price:
                break
            days_since_low += 1
        
        # Calculate Aroon values
        aroon_up_value = ((self.period - days_since_high) / self.period) * 100.0
        aroon_down_value = ((self.period - days_since_low) / self.period) * 100.0
        
        self.aroon_up.append(float(aroon_up_value))
        self.aroon_down.append(float(aroon_down_value))
        
        self.previous_aroon_up = aroon_up_value
        self.previous_aroon_down = aroon_down_value
        
        # Maintain fixed length (similar to MACD implementation)
        LEN = 100
        if len(self.aroon_up) > 2*LEN:
            del self.aroon_up[:-LEN]
            del self.aroon_down[:-LEN]