import array

class roc:
    def __init__(self,
                 closes: array.array,
                 period: int = 14,  # Default period is 14
                 ):
        self.closes = closes
        self.period = period
        
        # Initialize array for ROC values
        self.roc = array.array('d', [])
        self.data_points = 0
        
    def update(self):
        """
        Calculate ROC using the formula:
        ROC = ((Current Close - Close n periods ago) / Close n periods ago) × 100
        
        ROC shows the percentage change in price from the close n periods ago
        """
        self.data_points += 1
        
        # Need at least period + 1 points to calculate ROC
        if self.data_points <= self.period:
            self.roc.append(0.0)
            return
            
        current_close = self.closes[-1]
        past_close = self.closes[-self.period - 1]
        
        # Calculate ROC
        if past_close != 0:
            roc = ((current_close - past_close) / past_close) * 100
        else:
            roc = 0.0
            
        self.roc.append(roc)
        LEN = 100
        if len(self.roc) > 2*LEN:
            del self.roc[:-LEN]
        return