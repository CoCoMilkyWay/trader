import array

class kst:
    def __init__(self,
                 rcma1: array.array,    # ROC arrays from shortest to longest period
                 rcma2: array.array,
                 rcma3: array.array,
                 rcma4: array.array,
                 sma_periods: list[int] = [10, 15, 20, 30],      # Smoothing periods
                 weights: list[float] = [1, 2, 3, 4],            # Weight for each ROC
                 signal_period: int = 9                           # Signal line period
                 ):
        # Initialize array
        self.histogram = array.array('d', [])
        
        # Store ROC inputs
        self.rocs = [rcma1, rcma2, rcma3, rcma4]
        self.sma_periods = [int(p) for p in sma_periods]
        
        # Normalize weights
        weight_sum = sum(weights)
        self.weights = [float(w/weight_sum) for w in weights]
        self.signal_period = int(signal_period)
        
        # Initialize previous values
        self.previous_kst = 0.0
        self.previous_signal = 0.0
        self.previous_histogram = 0.0
        self.data_points = 0
        
        # Store recent values
        self.recent_rocs = [[] for _ in range(len(self.rocs))]
        self.recent_kst = []  # Store recent KST values for signal calculation
        
    def _calculate_sma(self, values: list, period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(values) < period:
            return 0.0
        return sum(values[-period:]) / period
        
    def update(self):
        """
        The KST (Know Sure Thing) oscillator combines 4 different ROC (Rate of Change) 
        calculations with different periods to form a momentum oscillator.
        
        Trading signals:
        1. Histogram crossing zero upward: Bullish signal
        2. Histogram crossing zero downward: Bearish signal
        3. Histogram divergence with price: Potential reversal signal
        """
        self.data_points += 1
        
        # Get current ROC values
        weighted_sum = 0.0
        for i, (roc_array, sma_period, weight) in enumerate(
            zip(self.rocs, self.sma_periods, self.weights)):
            
            # Get current ROC value
            current_roc = float(roc_array[-1])
            self.recent_rocs[i].append(current_roc)
            
            # Maintain ROC arrays at required length
            if len(self.recent_rocs[i]) > sma_period:
                self.recent_rocs[i].pop(0)
                
            # Calculate smoothed ROC
            smoothed_roc = self._calculate_sma(self.recent_rocs[i], sma_period)
            weighted_sum += smoothed_roc * weight
            
        # Calculate KST
        kst_value = float(weighted_sum)
        self.recent_kst.append(kst_value)
        
        # Maintain KST array at signal period length
        if len(self.recent_kst) > self.signal_period:
            self.recent_kst.pop(0)
            
        # Calculate signal line (SMA of KST)
        signal_value = self._calculate_sma(self.recent_kst, self.signal_period)
        
        # Calculate histogram
        histogram_value = float(kst_value - signal_value)
        self.histogram.append(histogram_value)
        
        # Update previous values
        self.previous_kst = kst_value
        self.previous_signal = signal_value
        self.previous_histogram = histogram_value
        
        # Maintain fixed length
        LEN = 100
        if len(self.histogram) > 2*LEN:
            del self.histogram[:-LEN]
            
        return