import array
import math
class squeeze:
    def __init__(self,
                 closes: array.array,
                 bb_upper: array.array,    # Bollinger Band upper
                 bb_lower: array.array,    # Bollinger Band lower
                 kc_upper: array.array,    # Keltner Channel upper
                 kc_lower: array.array,    # Keltner Channel lower
                 linreg_length: int = 20   # Linear regression length
                 ):
        self.closes = closes
        self.bb_upper = bb_upper
        self.bb_lower = bb_lower
        self.kc_upper = kc_upper
        self.kc_lower = kc_lower
        self.linreg_length = linreg_length
        
        # Initialize arrays for output values
        self.momentum = array.array('d', [])        # Momentum value
        self.squeeze_rating = array.array('d', [])  # Rating value (0-3)
        self.data_points = 0
        
    def calculate_linreg(self, values: list) -> float:
        """Simple linear regression slope calculation"""
        n = len(values)
        if n < 2:
            return 0.0
            
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (y - y_mean) 
                       for i, y in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator
        
    def calculate_squeeze_rating(self, bb_width: float, kc_width: float) -> float:
        """
        Calculate squeeze rating (0-3):
        0 = No squeeze (BB width > KC width)
        1 = Light squeeze (BB width <= KC width but > 75% of KC width)
        2 = Moderate squeeze (BB width <= 75% but > 50% of KC width)
        3 = Tight squeeze (BB width <= 50% of KC width)
        """
        # if bb_width > kc_width:
        #     return 1
        
        if kc_width != 0: # when trade stops, KC goes to 0 quickly, causing exception
            bb_kc_ratio = bb_width / kc_width
        else:
            bb_kc_ratio = 10
        
        # if bb_kc_ratio > 0.75:
        #     return 1
        # elif bb_kc_ratio > 0.50:
        #     return 2
        # else:
        #     return 3
        
        # easier for NN to learn for more even distribution
        return math.log1p(bb_kc_ratio)
    
    def update(self):
        """
        Calculate Squeeze Momentum and Rating:
        1. Check if BBands are inside Keltner Channels (squeeze condition)
        2. Calculate squeeze rating based on relative band widths
        3. Calculate momentum using linear regression
        """
        self.data_points += 1
        
        # Get latest values
        bb_width = self.bb_upper[-1] - self.bb_lower[-1]
        kc_width = self.kc_upper[-1] - self.kc_lower[-1]
                
        # Calculate and append squeeze rating
        rating = self.calculate_squeeze_rating(bb_width, kc_width)
        self.squeeze_rating.append(rating)
        
        # Calculate momentum
        if self.data_points < self.linreg_length:
            self.momentum.append(0.0)
            return
            
        # Get recent closes for momentum calculation
        recent_closes = list(self.closes[-self.linreg_length:])
        momentum = self.calculate_linreg(recent_closes)
        self.momentum.append(momentum)
        LEN = 100
        if len(self.momentum) > 2*LEN:
            del self.momentum[:-LEN]
            del self.squeeze_rating[:-LEN]
        return