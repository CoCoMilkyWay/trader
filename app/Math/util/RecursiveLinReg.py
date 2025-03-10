import collections

class RecursiveLinReg:
    """
    A class to compute the linear regression slope for a fixed-length window
    using recursive (constant time) updates.
    
    We assume that the independent variable is fixed: x = 0, 1, 2, ... , n-1.
    """
    def __init__(self, window_size: int):
        self.n = window_size             # Regression window length
        self.count = 0                   # Number of data points seen so far
        self.S_y = 0.0                   # Running sum of y-values
        self.S_xy = 0.0                  # Running weighted sum: sum(i * y_i)
        # Use a deque for O(1) pops from the left
        self.window = collections.deque(maxlen=window_size)
        
        # Precompute S_x and S_xx for x = 0, 1, ..., n-1.
        self.S_x = sum(range(window_size))  # = n*(n-1)/2
        self.S_xx = sum(i * i for i in range(window_size))
        self.D = window_size * self.S_xx - self.S_x**2
        if self.D == 0:
            raise ValueError("Denominator computed as zero; window_size must be > 1.")
    
    def update(self, new_value: float) -> float:
        """
        Update the regression sums with a new y-value and return the current slope.
        
        If there are not yet n data points, returns None.
        """
        if self.count < self.n:
            # Window not full yet: add new value at position "count".
            self.window.append(new_value)
            self.S_y += new_value
            self.S_xy += self.count * new_value
            self.count += 1
            if self.count < self.n:
                return 0.0  # Not enough data to compute a valid regression.
            else:
                # Exactly full now: compute the slope.
                return (self.n * self.S_xy - self.S_x * self.S_y) / self.D
        else:
            # Window is full: remove the oldest value and update sums recursively.
            y_old = self.window.popleft()
            old_S_y = self.S_y  # Save the previous S_y for the update of S_xy.
            
            # Update the running sum S_y:
            self.S_y = self.S_y - y_old + new_value
            
            # Update S_xy:
            # Derivation:
            #   New S_xy = sum_{i=0}^{n-2} i * y_{i+1} + (n-1)*new_value
            #             = (S_xy_old - (S_y_old - y_old)) + (n-1)*new_value
            self.S_xy = self.S_xy - (old_S_y - y_old) + (self.n - 1) * new_value
            
            # Add the new value to the window.
            self.window.append(new_value)
            
            # Return the updated slope.
            return (self.n * self.S_xy - self.S_x * self.S_y) / self.D
