class AverageFilter:
    def __init__(self, window_size=None):
        self.window_size = window_size
        self.values = [] if window_size is None else [0] * window_size
        self.sum = 0
        self.pos = 0
        self.is_filled = False

    def update(self, value):
        if self.window_size is None:
            # Running average (all data)
            self.values.append(value)
            self.sum += value
            return self.sum / len(self.values)
        
        # Rolling average with fixed window
        self.sum += value - self.values[self.pos]
        self.values[self.pos] = value
        self.pos = (self.pos + 1) % self.window_size
        self.is_filled = self.is_filled or self.pos == 0
        return self.sum / (self.window_size if self.is_filled else self.pos or 1)

    def get_average(self):
        if not self.values:
            return 0.0
        return self.sum / (len(self.values) if self.window_size is None else 
                         (self.window_size if self.is_filled else self.pos))

    def reset(self):
        self.values = [] if self.window_size is None else [0] * self.window_size
        self.sum = 0
        self.pos = 0
        self.is_filled = False