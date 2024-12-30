import array
import math

class fisher:
    def __init__(self,
                 highs: array.array,
                 lows: array.array,
                 period: int = 9    # Default period is 9
                 ):
        self.highs = highs
        self.lows = lows
        self.period = period
        
        # Initialize arrays for final Fisher values
        self.fisher = array.array('d', [])
        # self.trigger_values = array.array('d', [])  # 1-period lag of fisher
        
        # Keep track of previous values for calculations
        self.prev_value1 = 0.0     # Previous smoothed value
        self.prev_fisher = 0.0     # Previous fisher transform value
        self.data_points = 0
        
    def update(self):
        """
        Calculate Fisher Transform:
        1. Calculate median price = (High + Low) / 2
        2. Track the highest high and lowest low over period
        3. Calculate percent position = scaled value between -1 and 1
        4. Apply smoothing
        5. Apply Fisher Transform
        """
        self.data_points += 1
        
        # Need enough data points for the lookback period
        if self.data_points <= self.period:
            self.fisher.append(0.0)
            # self.trigger_values.append(0.0)
            return
        
        # Get price range over period
        high_prices = self.highs[-self.period:]
        low_prices = self.lows[-self.period:]
        highest_high = max(high_prices)
        lowest_low = min(low_prices)
        
        # Calculate latest median price
        median_price = (self.highs[-1] + self.lows[-1]) / 2.0
        
        # Calculate price position as a value between -1 and 1
        price_range = highest_high - lowest_low
        if price_range != 0:
            value1 = 0.33 * 2 * ((median_price - lowest_low) / price_range - 0.5) + 0.67 * self.prev_value1
        else:
            value1 = 0
            
        # Bound value between -0.999 and 0.999
        value1 = max(min(0.999, value1), -0.999)
        
        # Apply Fisher Transform
        fisher = 0.5 * math.log((1 + value1) / (1 - value1))
        
        # Store previous values for next iteration
        self.prev_value1 = value1
        
        # Store 1-period lag as trigger line
        trigger = self.prev_fisher
        self.prev_fisher = fisher
        
        self.fisher.append(fisher)
        # self.trigger_values.append(trigger)
        LEN = 100
        if len(self.fisher) > 2*LEN:
            del self.fisher[:-LEN]
        return