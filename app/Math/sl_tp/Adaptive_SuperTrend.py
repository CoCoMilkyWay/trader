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
#    Low Vol    → factor × 0.677
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
from config.cfg_stk import cfg_stk

class AdaptiveSuperTrend:
    """
    A trailing-stop/trend-identify method, more stable than parabolic-SAR and Chandelier
    
    By experiments:
    
        more stable = factor is linearly proportional to holding period

        while other SL-TP method are more sensitive to hyperparameter tuning
        
    If the trend is clean, then this indicator is more likely to have higher winrate/return. e.g. bear trend if of higher quality in bull market
    """
    __slots__ = ('atr_len', 'factor', 'ema_multiplier', 'prev_close', 'prev_atr',
                 'prev_supertrend', 'prev_direction', 'prev_upper', 'prev_lower',
                 'atr_history', 'centroids', 'current_cluster', 'cluster_factors', 'is_initialized',
                 
                 # debug
                 'his_ts_upper', 'his_ts_lower', 'his_val_upper', 'his_val_lower',
                 'trade_start_price', 'trade_start_time', 'long_trades', 'short_trades',
                 'avg_return', 'avg_win_rate', 'avg_hold',
                 )
    
    def __init__(self, atr_len:int=50, factor:float=3):
        self.atr_len = atr_len
        self.factor = factor
        
        # EMA multiplier for ATR calculation
        self.ema_multiplier = 2.0 / (self.atr_len + 1)
        
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
        self.centroids = [0.0, 0.0, 0.0]  # Fixed size of 3 (high, medium, low) volatility
        self.cluster_factors = [1.5, 1, 0.667,]
        self.current_cluster = 1  # Start with medium volatility
        
        # Init state flag
        self.is_initialized = False
        
        # Keep lists for debug to maintain original behavior
        if cfg_stk.dump_ind:
            self.his_ts_upper = [[]]
            self.his_ts_lower = [[]]
            self.his_val_upper = [[]]
            self.his_val_lower = [[]]
            # trade tracking attributes
            self.trade_start_price = 0.0
            self.trade_start_time = 0
            self.long_trades = {'returns':0.0, 'wins': 0, 'total': 0, 'hold_time': 0}
            self.short_trades = {'returns':0.0, 'wins': 0, 'total': 0, 'hold_time': 0}
            self.avg_return = [0.0, 0.0] # long, short
            self.avg_win_rate = [0.0, 0.0] # long, short
            self.avg_hold = [0.0, 0.0] # long, short

    def _calculate_kmeans(self):
        """
        volatility clustering using k-means rather than quantile
        k-means clustering is a natural process, each cluster can have different number of elements
        """
        if len(self.atr_history) < 3:
            return 1
        
        # Quick percentile calculation
        if self.centroids == [0.0, 0.0, 0.0]:
            data = sorted(self.atr_history)
            n = len(data)
            self.centroids[0] = data[n * 3 // 4]  # 75th percentile
            self.centroids[1] = data[n // 2]      # median
            self.centroids[2] = data[n // 4]      # 25th percentile
        
        # Run k-means iterations
        for _ in range(5):
            clusters = ([], [], [])
            # Assign points to nearest centroid - optimized distance calculation
            for atr in self.atr_history:
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
            self.centroids.sort(reverse=True) # high to low
        
        # Find current cluster - optimized distance comparison
        d0 = abs(self.prev_atr - self.centroids[0])
        d1 = abs(self.prev_atr - self.centroids[1])
        d2 = abs(self.prev_atr - self.centroids[2])
        
        if d0 <= d1 and d0 <= d2:
            return 0 # 75th percentile
        elif d1 <= d0 and d1 <= d2:
            return 1 # median
        return 2 # 25th percentile

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
            self.prev_atr = (tr * self.ema_multiplier) + (self.prev_atr * (1 - self.ema_multiplier))

            self.prev_atr = (self.prev_atr * (self.atr_len - 1) + tr) / self.atr_len
        else:
            self.prev_atr = tr
        
        # Update ATR history - O(1) with deque
        self.atr_history.append(self.prev_atr)

        # Calculate basic bands
        mid = (high + low) * 0.5  # Multiply is faster than divide
        adaptive_factor = self.factor * self.cluster_factors[self.current_cluster]
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
            if cfg_stk.dump_ind:
                self.trade_start_price = close
                self.trade_start_time = ts
            return False, False

        # Update bands
        final_upper = upper if (upper < self.prev_upper or close > self.prev_upper) else self.prev_upper
        final_lower = lower if (lower > self.prev_lower or close < self.prev_lower) else self.prev_lower
        
        # Calculate direction and SuperTrend
        direction = 1 if close > self.prev_supertrend else -1
        supertrend = final_lower if direction == 1 else final_upper

        # Check for trend change
        # signal = ''
        long_switch = False
        short_switch = False
        if direction != self.prev_direction:
            self.current_cluster = self._calculate_kmeans()
            if self.centroids[1]!=0:
                self.cluster_factors = [self.centroids[0]/self.centroids[1], 1, self.centroids[2]/self.centroids[1]]
            long_switch = (direction == 1)
            short_switch = (direction == -1)
            
            if cfg_stk.dump_ind:
                # Calculate trade result
                if self.trade_start_price != 0:
                    trade_result = (close - self.trade_start_price) * self.prev_direction / self.trade_start_price
                    hold_duration = ts - self.trade_start_time
                    # Update statistics
                    trades = self.long_trades if self.prev_direction == 1 else self.short_trades
                    trades['total'] += 1
                    if trade_result > 0:
                        trades['wins'] += 1
                    trades['returns'] += trade_result
                    trades['hold_time'] += hold_duration # type: ignore
                    # Print stats
                    if True: # trades['total'] == 1:
                        idx = 0 if direction==1 else 1
                        self.avg_return[idx] = ((trades['returns'] / trades['total'] * 100))
                        self.avg_win_rate[idx] = ((trades['wins'] / trades['total'] * 100))
                        self.avg_hold[idx] = (trades['hold_time'] / trades['total'] / 3600)  # Convert seconds to hours
                        # trades['total'] = 0
                        # trades['wins'] = 0
                        # trades['hold_time'] = 0
                        # print(f"{'LONG' if self.prev_direction == 1 else 'SHORT'} Switch - Factor: {self.factor} Win Rate: {self.win_rate[idx]:.1f}%, Avg Hold Time: {self.avg_hold[idx]:.1f}hrs")
                
                # Start new trade
                self.trade_start_price = close
                self.trade_start_time = ts
                
                # Update history lists
                if direction == 1:
                    self.his_ts_lower.append([])
                    self.his_val_lower.append([])
                else:
                    self.his_ts_upper.append([])
                    self.his_val_upper.append([])
                    
        elif cfg_stk.dump_ind:
            # Update history for current direction
            if direction == 1:
                self.his_ts_lower[-1].append(ts)
                self.his_val_lower[-1].append(supertrend)
            else:
                self.his_ts_upper[-1].append(ts)
                self.his_val_upper[-1].append(supertrend)
                
        # Update state
        self.prev_supertrend = supertrend
        self.prev_direction = direction
        self.prev_upper = final_upper
        self.prev_lower = final_lower
        
        return long_switch, short_switch
    
    def get_stats(self, code:str):
        print(f"\nAdaptiveSuperTrend Statistics: {code}")
        print(f"Long trades:  Return: {self.avg_return[0]:6.2f}% | Win Rate: {self.avg_win_rate[0]:5.1f}% | Avg Hold: {self.avg_hold[0]:5.1f}h")
        print(f"Short trades: Return: {self.avg_return[1]:6.2f}% | Win Rate: {self.avg_win_rate[1]:5.1f}% | Avg Hold: {self.avg_hold[1]:5.1f}h")
