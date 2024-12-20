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

# Price
#   ^                    Adaptive SuperTrend Components
#   |    
#   |    Upper Band         ATR Volatility Clusters
#   |    final_upper ----→ +-----------------+     High Vol (×1.5)
#   |                      |      Band       |     ••••••••
#   |           ×factor    |     Width      |     Med Vol (×1.0)
#   |    Price    ×ATR     |  adaptive_factor|     --------
#   |    Action            |      ×ATR      |     Low Vol (×0.75)
#   |                      |                |     ········
#   |              •••••••••••••••••••••••••••••••••         
#   |              |       |                |      |  ↑
#   |        Close •       |                |      |  | Band Width
#   |              |       |    Neutral     |      |  | = factor × 
#   |              |       |     Zone       |      |  |   adaptive_factor ×
#   |              • Mid   |                |      |  |   ATR
#   |            (H+L)/2   |                |      |  ↓
#   |              |       |                |      |
#   |              •••••••••••••••••••••••••••••••• 
#   |                      |                |
#   |    Lower Band       +-----------------+
#   |    final_lower --→
#   |              |<-------------------->|
#   |                   ATR Length (10)
#   |              
#   |              |<--------------------------------->|
#   |                      Lookback (200)
#   +-------------------------------------------------> Time
#                                                     
# 
# SuperTrend State Changes:
# ------------------------
# Direction = 1 (Bullish)   Direction = -1 (Bearish)
# SuperTrend = final_lower  SuperTrend = final_upper
# 
# Volatility Regime Detection:
# ---------------------------
# 1. Collect ATR history in lookback window
# 2. Cluster ATR values into 3 groups using k-means
# 3. Assign multipliers based on cluster:
#    High Vol   → factor × 1.5
#    Medium Vol → factor × 1.0
#    Low Vol    → factor × 0.75
# 
# Band Calculation:
# ----------------
# mid = (high + low) / 2
# band_offset = factor × cluster_factor × ATR
# upper = mid + band_offset
# lower = mid - band_offset
# 
# Final Band Logic:
# ----------------
# final_upper = min(upper, prev_upper) if close ≤ prev_upper
#              else upper
# 
# final_lower = max(lower, prev_lower) if close ≥ prev_lower
#              else lower
# 
# Signal Generation:
# ----------------
# Long Signal:  Direction changes from -1 to 1
#              (Price crosses above SuperTrend)
#              → Recalculate volatility cluster
# 
# Short Signal: Direction changes from 1 to -1
#              (Price crosses below SuperTrend)
#              → Recalculate volatility cluster

from typing import Tuple, List
from collections import deque
from config.cfg_cpt import cfg_cpt

class AdaptiveSuperTrend:
    __slots__ = ('atr_len', 'factor', 'lookback', 'prev_close', 'prev_atr',
                 'prev_supertrend', 'prev_direction', 'prev_upper', 'prev_lower',
                 'atr_history', 'centroids', 'current_cluster', 'is_initialized',
                 'his_ts', 'his_val', '_cluster_factors')

    def __init__(self, atr_len=50, factor=3):
        self.atr_len = atr_len
        self.factor = factor
        
        # ATR state
        self.prev_close = 0.0
        self.prev_atr = 0.0
        
        # SuperTrend state
        self.prev_supertrend = 0.0
        self.prev_direction = 0
        self.prev_upper = 0.0
        self.prev_lower = 0.0
        
        # Volatility state - only use deque for atr_history
        self.atr_history = deque(maxlen=self.atr_len)
        self.centroids = [0.0, 0.0, 0.0]  # Fixed size of 3
        self.current_cluster = 1  # Start with medium volatility
        
        # Pre-calculate factors (magical numbers :< try have some research on atr distribution)
        self._cluster_factors = (1.5, 1.0, 0.75)  # Fixed size tuple of 3
        
        # Init state flag
        self.is_initialized = False
        
        # Keep lists for debug to maintain original behavior
        if cfg_cpt.dump_ind:
            self.his_ts = []
            self.his_val = []

    def _calculate_kmeans(self):
        """Optimized k-means on ATR history"""
        if len(self.atr_history) < 3:
            return 1
            
        # Convert deque to sorted list once
        data = sorted(self.atr_history)
        n = len(data)
        
        # Quick percentile calculation
        self.centroids[0] = data[n * 3 // 4]  # 75th percentile
        self.centroids[1] = data[n // 2]      # median
        self.centroids[2] = data[n // 4]      # 25th percentile
        
        # Use lists for clusters to maintain original behavior
        clusters = ([], [], [])
        
        # Run k-means iterations
        for _ in range(5):
            # Clear previous clusters
            for cluster in clusters:
                cluster.clear()
            
            # Assign points to nearest centroid - optimized distance calculation
            for atr in data:
                d0 = abs(atr - self.centroids[0])
                d1 = abs(atr - self.centroids[1])
                d2 = abs(atr - self.centroids[2])
                
                if d0 <= d1 and d0 <= d2:
                    clusters[0].append(atr)
                elif d1 <= d0 and d1 <= d2:
                    clusters[1].append(atr)
                else:
                    clusters[2].append(atr)
            
            # Update centroids - use direct sum
            for i, cluster in enumerate(clusters):
                if cluster:  # Only update if cluster has points
                    self.centroids[i] = sum(cluster) / len(cluster)
            
            # Keep centroids ordered
            if self.centroids[1] > self.centroids[0]:
                self.centroids[0], self.centroids[1] = self.centroids[1], self.centroids[0]
            if self.centroids[2] > self.centroids[1]:
                self.centroids[1], self.centroids[2] = self.centroids[2], self.centroids[1]
                if self.centroids[1] > self.centroids[0]:
                    self.centroids[0], self.centroids[1] = self.centroids[1], self.centroids[0]
        
        # Find current cluster - optimized distance comparison
        d0 = abs(self.prev_atr - self.centroids[0])
        d1 = abs(self.prev_atr - self.centroids[1])
        d2 = abs(self.prev_atr - self.centroids[2])
        
        if d0 <= d1 and d0 <= d2:
            return 0
        elif d1 <= d0 and d1 <= d2:
            return 1
        return 2

    def update(self, high: float, low: float, close: float, ts: float) -> Tuple[bool, bool]:
        # Calculate TR and ATR
        if self.is_initialized:
            tr1 = high - low
            tr2 = abs(high - self.prev_close)
            tr3 = abs(low - self.prev_close)
            tr = tr1 if tr1 > tr2 else tr2
            tr = tr if tr > tr3 else tr3
        else:
            tr = high - low
            
        self.prev_close = close

        if self.is_initialized:
            self.prev_atr = (self.prev_atr * (self.atr_len - 1) + tr) / self.atr_len
        else:
            self.prev_atr = tr
        
        # Update ATR history - O(1) with deque
        self.atr_history.append(self.prev_atr)

        # Calculate basic bands
        mid = (high + low) * 0.5  # Multiply is faster than divide
        adaptive_factor = self.factor * self._cluster_factors[self.current_cluster]
        
        band_offset = adaptive_factor * self.prev_atr
        upper = mid + band_offset
        lower = mid - band_offset

        # Initialize on first bar
        if not self.is_initialized:
            self.prev_supertrend = mid
            self.prev_direction = 1 if close > mid else -1
            self.prev_upper = upper
            self.prev_lower = lower
            self.is_initialized = True
            
            return False, False

        # Update bands
        final_upper = upper if (upper < self.prev_upper or close > self.prev_upper) else self.prev_upper
        final_lower = lower if (lower > self.prev_lower or close < self.prev_lower) else self.prev_lower
        
        # Calculate direction and SuperTrend
        direction = 1 if close > self.prev_supertrend else -1
        supertrend = final_lower if direction == 1 else final_upper

        # Check for trend change
        signal = ''
        long_switch = False
        short_switch = False
        if direction != self.prev_direction:
            self.current_cluster = self._calculate_kmeans()
            if direction == 1:
                signal = 'buy'
                long_switch = True
            else:
                signal = 'sell'
                short_switch = True

        # Update state
        self.prev_supertrend = supertrend
        self.prev_direction = direction
        self.prev_upper = final_upper
        self.prev_lower = final_lower
        
        if cfg_cpt.dump_ind:
            self.his_ts.append(ts)
            self.his_val.append(supertrend)

        return long_switch, short_switch