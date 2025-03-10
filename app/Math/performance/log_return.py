import array
import math

class logreturn:
    def __init__(self,
                 closes: array.array,
                 # smooth_period: int = 0,     # 0 means no smoothing
                 ):
        # Initialize arrays
        self.log_returns = array.array('d', [])
        
        # Store inputs
        self.closes = closes
        # self.smooth_period = int(smooth_period)
        
        # Initialize previous values
        self.previous_close = 0.0
        self.previous_log_return = 0.0
        self.data_points = 0
        
        # Initialize smoothing parameters if needed
        # if self.smooth_period > 0:
        #     self.alpha = 2.0 / (self.smooth_period + 1)
        
    def update(self):
        """
        Update the log return indicator with the latest close price.
        The formula used is: log_return = ln(close_t / close_t-1)
        If smoothing is enabled, applies EMA smoothing to the log returns.
        """
        self.data_points += 1
        
        if self.data_points < 2:  # Need at least 2 points for the first return
            self.previous_close = float(self.closes[-1])
            self.log_returns.append(0.0)  # First point has no return
            return
            
        current_close = float(self.closes[-1])
        
        # Calculate log return
        try:
            log_return = float(math.log(current_close / self.previous_close))
        except (ValueError, ZeroDivisionError):
            # Handle edge cases (zero or negative prices)
            log_return = 0.0
            
        # # Apply smoothing if enabled
        # if self.smooth_period > 0:
        #     if len(self.log_returns) == 0:
        #         smoothed_return = log_return
        #     else:
        #         smoothed_return = float(log_return * self.alpha + 
        #                              self.log_returns[-1] * (1 - self.alpha))
        #     self.log_returns.append(smoothed_return)
        # else:
        self.log_returns.append(log_return)
            
        # Update previous values
        self.previous_close = current_close
        self.previous_log_return = self.log_returns[-1]
        
        # Maintain fixed length (same as MACD example)
        LEN = 100
        if len(self.log_returns) > 2*LEN:
            del self.log_returns[:-LEN]