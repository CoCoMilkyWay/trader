import array

class td_seq:
    def __init__(self,
                 closes: array.array,
                 ):
        # Store inputs
        self.closes = closes
        self.setup_index = 0  # Positive for price > price[4], negative for price < price[4]
        self.data_points = 0
        
    def update(self):
        """
        Update TD Sequential indicator with latest data.
        Returns:
            setup_index: Positive numbers when current close > close[4]
                        Negative numbers when current close < close[4]
                        Counts continue as long as condition holds
        """
        self.data_points += 1
        
        # Need at least 4 bars of data
        if self.data_points < 5:
            self.setup_index = 0
            return
            
        # Compare current close with close 4 bars ago
        if self.closes[-1] > self.closes[-4]:
            self.setup_index = self.setup_index + 1 if self.setup_index > 0 else 1
        elif self.closes[-1] < self.closes[-4]:
            self.setup_index = self.setup_index - 1 if self.setup_index < 0 else -1
        else:
            self.setup_index = 0
            
        return