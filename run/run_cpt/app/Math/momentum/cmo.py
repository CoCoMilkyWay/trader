import array

class cmo:
    def __init__(self,
                 closes: array.array,
                 period: int = 9    # Default period is 9
                 ):
        self.closes = closes
        self.period = period
        
        # Initialize array for CMO values
        self.cmo = array.array('d', [])
        
        # Keep track of recent gains and losses for calculation
        self.gains = []  # Track recent gains
        self.losses = [] # Track recent losses
        self.data_points = 0
        
    def update(self):
        """
        Calculate Chande Momentum Oscillator:
        CMO = 100 * ((Sum of Gains - Sum of Losses) / (Sum of Gains + Sum of Losses))
        
        Where:
        - Gain: Price increase from previous close
        - Loss: Price decrease from previous close (absolute value)
        - Period: Number of periods to sum gains and losses
        """
        self.data_points += 1
        
        # Need at least 2 points to calculate price change
        if self.data_points < 2:
            self.cmo.append(0.0)
            return
            
        # Calculate current gain or loss
        current_close = self.closes[-1]
        previous_close = self.closes[-2]
        price_change = current_close - previous_close
        
        if price_change > 0:
            self.gains.append(price_change)
            self.losses.append(0.0)
        else:
            self.gains.append(0.0)
            self.losses.append(abs(price_change))
            
        # Keep only the most recent period's worth of data
        if len(self.gains) > self.period:
            self.gains.pop(0)
            self.losses.pop(0)
            
        # Calculate CMO once we have enough data
        if self.data_points <= self.period + 1:
            self.cmo.append(0.0)
            return
            
        sum_gains = sum(self.gains)
        sum_losses = sum(self.losses)
        
        # Calculate CMO
        if sum_gains + sum_losses != 0:
            cmo = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
        else:
            cmo = 0.0
            
        self.cmo.append(cmo)
        LEN = 100
        if len(self.cmo) > 2*LEN:
            del self.cmo[:-LEN]
        return