import array
import math

class gk:
    """
    Garman-Klass Volatility Estimator is more efficient because it considers intraday price movements (high and low prices)
    especially useful in low-frequency trading strategies, risk management, and options pricing
    """
    def __init__(self,
                 opens: array.array,
                 highs: array.array,
                 lows: array.array,
                 closes: array.array,
                 period: int = 10,     # Default lookback period
                 annualization: float = 252.0  # Trading days per year
                 ):
        # Initialize volatility array
        self.volatility = array.array('d', [])
        
        # Store inputs
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.period = int(period)
        self.annualization = float(annualization)
        
        # Initialize previous value
        self.previous_vol = 0.0
        self.data_points = 0

    def update(self):
        """
        Updates Garman-Klass volatility using OHLC prices
        Returns annualized volatility in percentage terms
        """
        self.data_points += 1
        
        if self.data_points < self.period:
            self.volatility.append(0.0)
            return

        # Get the period data
        period_opens = self.opens[-self.period:]
        period_highs = self.highs[-self.period:]
        period_lows = self.lows[-self.period:]
        period_closes = self.closes[-self.period:]
        
        # Calculate sum of variances over period
        sum_variance = 0.0
        for o, h, l, c in zip(period_opens, period_highs, period_lows, period_closes):
            # Log calculations
            log_h = math.log(h)
            log_l = math.log(l)
            log_o = math.log(o)
            log_c = math.log(c)
            
            # GK estimator components
            hl = 0.5 * (log_h - log_l) ** 2  # High-Low component
            co = (2 * math.log(2) - 1) * (log_c - log_o) ** 2  # Close-Open component
            
            # Daily variance
            sum_variance += hl - co
            
        # Calculate period variance
        period_variance = sum_variance / self.period
        
        # Convert to annualized volatility
        if period_variance > 0:
            vol_value = math.sqrt(period_variance * self.annualization) * 100
        else:
            vol_value = 0.0
            
        self.volatility.append(float(vol_value))
        self.previous_vol = vol_value
        
        # Maintain fixed length
        LEN = 100
        if len(self.volatility) > 2*LEN:
            del self.volatility[:-LEN]