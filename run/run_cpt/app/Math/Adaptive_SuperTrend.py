"""
SuperTrend Signal Formation:
    When a new SuperTrend forms (direction changes), it represents a significant market structure change
    This is a natural point to reassess volatility regime
    SuperTrends don't form very frequently, so computation cost is reasonable
Lookback Period:
    Using a few hundred bars (e.g., 100-200)
    Roughly spans multiple SuperTrend cycles
    Provides enough data for meaningful clustering
    Recent enough to be relevant to current market conditions
"""

#　# Initialize
#　indicator = AdaptiveSuperTrend(atr_len=10, factor=3, lookback=100)
#　
#　# Process new bar
#　def on_bar(high, low, close):
#　    result = indicator.update(high, low, close)
#　    if result['signal']:
#　        print(f"Signal: {result['signal']}")
#　        print(f"Cluster: {'High' if result['cluster'] == 0 else 'Medium' if result['cluster'] == 1 else 'Low'}")
#　        print(f"SuperTrend: {result['value']:.2f}")

class AdaptiveSuperTrend:
    def __init__(self, atr_len=10, factor=3, lookback=200):
        self.atr_len = atr_len
        self.factor = factor
        self.lookback = lookback
        
        # ATR state
        self.prev_close = 0.0
        self.prev_atr = 0.0
        
        # SuperTrend state
        self.prev_supertrend = 0.0
        self.prev_direction = 0
        self.prev_upper = 0.0
        self.prev_lower = 0.0
        
        # Volatility state
        self.atr_history = []
        self.centroids = [0.0, 0.0, 0.0]
        self.current_cluster = 1  # Start with medium volatility
        
        # Init state flag
        self.is_initialized = False
        
    def _calculate_kmeans(self):
        """Run k-means on ATR history when SuperTrend changes"""
        if len(self.atr_history) < 3:
            return 1
            
        # Initialize centroids based on percentiles
        data = sorted(self.atr_history)
        high_idx = int(len(data) * 0.75)
        low_idx = int(len(data) * 0.25)
        self.centroids = [
            data[high_idx],      # 75th percentile
            data[len(data)//2],  # median
            data[low_idx]        # 25th percentile
        ]
        
        # Run k-means iterations
        for _ in range(5):
            clusters = [[], [], []]
            
            # Assign points to nearest centroid
            for atr in self.atr_history:
                distances = [abs(atr - c) for c in self.centroids]
                nearest = distances.index(min(distances))
                clusters[nearest].append(atr)
            
            # Update centroids
            for i in range(3):
                if clusters[i]:
                    self.centroids[i] = sum(clusters[i]) / len(clusters[i])
            
            # Keep centroids ordered (high to low)
            self.centroids.sort(reverse=True)
        
        # Determine current cluster
        distances = [abs(self.prev_atr - c) for c in self.centroids]
        return distances.index(min(distances))

    def update(self, high: float, low: float, close: float) -> dict:
        # Calculate TR and ATR
        if self.is_initialized:
            tr = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close)
            )
        else:
            tr = high - low
            
        self.prev_close = close

        if self.is_initialized:
            self.prev_atr = (self.prev_atr * (self.atr_len - 1) + tr) / self.atr_len
        else:
            self.prev_atr = tr
        
        # Update ATR history
        self.atr_history.append(self.prev_atr)
        if len(self.atr_history) > self.lookback:
            self.atr_history.pop(0)

        # Calculate basic bands
        mid = (high + low) / 2
        cluster_factors = [1.5, 1.0, 0.75]  # High, Medium, Low volatility factors
        adaptive_factor = self.factor * cluster_factors[self.current_cluster]
        
        upper = mid + (adaptive_factor * self.prev_atr)
        lower = mid - (adaptive_factor * self.prev_atr)

        # Initialize on first bar
        if not self.is_initialized:
            self.prev_supertrend = mid
            self.prev_direction = 1 if close > mid else -1
            self.prev_upper = upper
            self.prev_lower = lower
            self.is_initialized = True
            return {
                'value': mid,
                'direction': self.prev_direction,
                'signal': '',
                'cluster': self.current_cluster,
                'atr': self.prev_atr
            }

        # Update bands
        final_upper = upper if (upper < self.prev_upper or close > self.prev_upper) else self.prev_upper
        final_lower = lower if (lower > self.prev_lower or close < self.prev_lower) else self.prev_lower
        
        # Calculate direction and SuperTrend
        direction = 1 if close > self.prev_supertrend else -1
        supertrend = final_lower if direction == 1 else final_upper

        # Check for trend change and recalculate clusters if needed
        signal = ''
        if direction != self.prev_direction:
            self.current_cluster = self._calculate_kmeans()
            signal = 'buy' if direction == 1 else 'sell'

        # Update state
        self.prev_supertrend = supertrend
        self.prev_direction = direction
        self.prev_upper = final_upper
        self.prev_lower = final_lower

        return {
            'value': supertrend,
            'direction': direction,
            'signal': signal,
            'cluster': self.current_cluster,
            'atr': self.prev_atr
        }