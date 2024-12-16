import numpy as np
from collections import deque
from typing import Tuple, Optional
# https://www.youtube.com/watch?v=AdINVvnJfX4&ab_channel=JustinDehorty

"""
                         [RSI] 
  ::..                     |                    ..:::  
   .....                   |                  ......
    .••••::.               |               :••••••. 
     .:•••••:.             |            :::••••••.  
       .•••••:...          |         .::.••••••.    
         .::•••••::..      |       :..••••••..      
            .:•••••••::.........::••••••:..         
              ..::::••••.•••••••.•••••••:.            
                ...:•••••••.•••••••••::.              
                  .:..••.••••••.••••..                
  |---------------.:•••••••••••••••••.---------------[ADX]          
  0             .:•:•••.••••••.•••••••.                
              .••••••••••••••••••••••••:.            
            .:••••••••••::..::.::••••••••:.          
          .::••••••::.     |       .::•••:::.       
         .:••••••..        |          :••••••••.     
       .:••••:...          |           ..•••••••:.   
     ..:••::..             |              :.•••••••.   
    .:•....                |               ...::.:••.  
   ...:..                  |                   :...:••.     
  :::.                     |                       ..::  
                          _|_ 0

       Figure: Neighborhood in Lorentzian Space 
 Observations:
 (1) In Lorentzian Space, the shortest distance between two points is not 
     necessarily a straight line, but rather, a geodesic curve.
 (2) The warping effect of Lorentzian distance reduces the overall influence  
     of outliers and noise.
 (3) Lorentzian Distance becomes increasingly different from Euclidean Distance 
     as the number of nearest neighbors used for comparison increases.

There are several problems with this traditional KNN approach in the context of real-time calculations involving time series data:
- It is computationally expensive to iterate through the entire dataset and calculate the distance between every historical bar and
  the current bar.
- Market time series data is often non-stationary, meaning that the statistical properties of the data change slightly over time.
- It is possible that the nearest neighbors are not the most informative ones, and the KNN algorithm may return poor results if the
  nearest neighbors are not representative of the majority of the data.

Previously, the user @capissimo attempted to address some of these issues in several of his PineScript-based KNN implementations by:
- Using a modified KNN algorithm based on consecutive furthest neighbors to find a set of approximate "nearest" neighbors.
- Using a sliding window approach to only calculate the distance between the current bar and the most recent n bars in the dataset.

Of these two approaches, the latter is inherently limited by the fact that it only considers the most recent bars in the overall dataset. 

The former approach has more potential to leverage historical price action, but is limited by:
- The possibility of a sudden "max" value throwing off the estimation
- The possibility of selecting a set of approximate neighbors that are not representative of the majority of the data by oversampling 
  values that are not chronologically distinct enough from one another
- The possibility of selecting too many "far" neighbors, which may result in a poor estimation of price action

To address these issues, a novel Approximate Nearest Neighbors (ANN) algorithm is used in this indicator.

In the below ANN algorithm:
1. The algorithm iterates through the dataset in chronological order, using the modulo operator to only perform calculations every 4 bars.
   This serves the dual purpose of reducing the computational overhead of the algorithm and ensuring a minimum chronological spacing 
   between the neighbors of at least 4 bars.
2. A list of the k-similar neighbors is simultaneously maintained in both a predictions array and corresponding distances array.
3. When the size of the predictions array exceeds the desired number of nearest neighbors specified in settings.neighborsCount, 
   the algorithm removes the first neighbor from the predictions array and the corresponding distance array.
4. The lastDistance variable is overriden to be a distance in the lower 25% of the array. This step helps to boost overall accuracy 
   by ensuring subsequent newly added distance values increase at a slower rate.
5. Lorentzian distance is used as a distance metric in order to minimize the effect of outliers and take into account the warping of 
   "price-time" due to proximity to significant economic events.

LORENTZIAN CLASSIFIER COMPLETE ALGORITHMIC FLOW
=============================================

1. DATA INGESTION AND HISTORY MANAGEMENT
--------------------------------------
- Maintain price history using deques (OHLCV)
- Fixed lookback period (default 2000 bars)
- Minimum required bars for calculation (20)

2. FEATURE CALCULATION & ENGINEERING
----------------------------------
A. RSI (Relative Strength Index) - Momentum
   - Primary RSI(14): medium-term momentum cycles
   - Secondary RSI(9): short-term momentum shifts
   - Calculation:
     1. Get price changes (deltas)
     2. Separate gains and losses
     3. Calculate average gain/loss
     4. RS = avg_gain / avg_loss
     5. RSI = 100 - (100 / (1 + RS))
   Purpose: Capture momentum state of market

B. Wave Trend (WT) - Cycle Detection
   - Calculation sequence:
     1. Get HLC3 = (High + Low + Close) / 3
     2. Calculate esa = EMA(HLC3, n1)
     3. Calculate d = EMA(abs(HLC3 - esa), n1)
     4. ci = (HLC3 - esa) / (0.015 * d)
     5. wt1 = EMA(ci, n2)
   Purpose: Identify market cycles and early reversals

C. CCI (Commodity Channel Index) - Price Deviations
   - Calculation steps:
     1. TP = (High + Low + Close) / 3
     2. SMA of TP
     3. Mean Deviation
     4. CCI = (TP - SMA) / (0.015 * Mean Deviation)
   Purpose: Measure price deviation from average

D. ADX (Average Directional Index) - Trend Strength
   - Calculation sequence:
     1. Calculate TR (True Range)
     2. +DM (Directional Movement)
     3. -DM (Directional Movement)
     4. Smooth with Wilder's method
     5. Calculate DI+, DI-
     6. ADX = SMA(abs(DI+ - DI-) / (DI+ + DI-))
   Purpose: Quantify trend strength

3. LORENTZIAN DISTANCE CALCULATION
--------------------------------
A. For each feature (f1 to f5):
   - distance += log(1 + |current_value - historical_value|)
   - Log transformation creates "gravity wells" around similar states
   - Automatically adjusts for different feature scales

B. Pattern Matching Process:
   - Sample every 4 bars (chronological spacing)
   - Compare current feature set with historical
   - Store distances and corresponding predictions
   - Maintain K nearest neighbors (default 8)
   - Update threshold using 75th percentile

4. KERNEL REGRESSION CALCULATION
------------------------------
A. Rational Quadratic Kernel (Primary):
   - RQ(x) = (1 + (x²)/(2α))^(-α)
   - α = relative_weighting (default 8.0)
   - Better handles non-linear relationships

B. Gaussian Kernel (Secondary):
   - G(x) = exp(-x²/(2α))
   - Used for crossover confirmation
   - More sensitive to local changes

C. Kernel Estimate Generation:
   - Weight = kernel(distance_to_current)
   - Estimate = Σ(weight * price) / Σ(weight)
   - Lookback window = 8 bars default
   - Regression level = 25 (sensitivity)

5. FILTER APPLICATION
-------------------
A. Volatility Filter:
   - Calculate 20-bar standard deviation
   - Compare with 100-bar average volatility
   - Must be <= 2.5x average to trade

B. Regime Filter:
   - Calculate 20-bar price slope
   - Must be > threshold (-0.1 default)
   - Identifies trending vs ranging

C. ADX Filter:
   - Must be > threshold (20 default)
   - Confirms trend strength

D. Moving Average Filters (Optional):
   - EMA and SMA checks
   - Price must be above for longs
   - Price must be below for shorts

6. SIGNAL GENERATION
------------------
A. Base Signal:
   - Sum of nearest neighbor predictions
   - Normalized by neighbor count
   - Creates initial direction

B. Signal Modification:
   - Apply kernel confirmation
   - Check filter conditions
   - Validate trend alignment
   
C. Signal Strength:
   - Based on prediction unanimity
   - Scaled by kernel slope
   - Range: 0.0 to 1.0

7. POSITION MANAGEMENT
--------------------
A. Entry Conditions:
   1. Base signal direction change
   2. All filters passed
   3. Kernel confirmation
   4. Minimum strength threshold

B. Exit Conditions:
   1. Fixed Time:
      - Exit after 4 bars if using time-based
   2. Dynamic (if enabled):
      - Kernel slope change
      - Early signal flip
      - Trend reversal

C. Early Signal Flip Detection:
   - Monitor kernel slope changes
   - Check prediction stability
   - Exit if signal flips before 4 bars

8. SIGNAL OUTPUT
--------------
Return tuple:
- Direction: -1 (short), 0 (neutral), 1 (long)
- Strength: 0.0 to 1.0 signal confidence
- Kernel: Current kernel regression estimate

9. TRADE MANAGEMENT
-----------------
A. Position Tracking:
   - Monitor bars held
   - Track current position
   - Handle signal transitions

B. Risk Management:
   - Early exit on signal flip
   - Kernel-based trend exit
   - Time-based position limit

The complete system creates a robust classification approach that:
- Identifies similar market states
- Confirms with multiple features
- Validates with filters
- Manages positions systematically
- Adapts to changing conditions
"""

class LorentzianClassifier:
    def __init__(self, 
                 lookback: int = 2000,
                 neighbors_count: int = 8,
                 feature_count: int = 5,
                 kernel_lookback: int = 8,
                 rq_kernel_width: float = 8.0,
                 regression_level: int = 25,
                 kernel_lag: int = 2,
                 use_kernel_smoothing: bool = False,
                 use_volatility_filter: bool = True,
                 use_regime_filter: bool = True,
                 use_adx_filter: bool = True,
                 regime_threshold: float = -0.1,
                 adx_threshold: int = 20,
                 use_ema_filter: bool = False,
                 use_sma_filter: bool = False,
                 ema_period: int = 200,
                 sma_period: int = 200):
        
        # Core parameters
        self.lookback = lookback
        self.neighbors_count = neighbors_count
        self.feature_count = min(max(2, feature_count), 5)
        
        # Kernel settings
        self.kernel_lookback = kernel_lookback
        self.rq_kernel_width = rq_kernel_width
        self.regression_level = regression_level
        self.kernel_lag = kernel_lag
        self.use_kernel_smoothing = use_kernel_smoothing
        
        # Filter settings
        self.use_volatility_filter = use_volatility_filter
        self.use_regime_filter = use_regime_filter
        self.use_adx_filter = use_adx_filter
        self.regime_threshold = regime_threshold
        self.adx_threshold = adx_threshold
        self.use_ema_filter = use_ema_filter
        self.use_sma_filter = use_sma_filter
        self.ema_period = ema_period
        self.sma_period = sma_period
        
        # Price & Volume History
        self.closes = deque(maxlen=lookback)
        self.highs = deque(maxlen=lookback)
        self.lows = deque(maxlen=lookback)
        self.volumes = deque(maxlen=lookback)
        
        # Feature Arrays
        self.f1_array = deque(maxlen=lookback)  # RSI
        self.f2_array = deque(maxlen=lookback)  # WT
        self.f3_array = deque(maxlen=lookback)  # CCI
        self.f4_array = deque(maxlen=lookback)  # ADX
        self.f5_array = deque(maxlen=lookback)  # RSI2
        
        # ML Components
        self.distances = deque(maxlen=neighbors_count)
        self.predictions = deque(maxlen=neighbors_count)
        self.last_distance = -1.0
        
        # Kernel Components
        self.kernel_estimates = deque(maxlen=lookback)
        self.yhat1 = deque(maxlen=lookback)  # Rational Quadratic
        self.yhat2 = deque(maxlen=lookback)  # Gaussian
        
        # State Management
        self.bars_held = 0
        self.current_signal = 0
        self.last_kernel_estimate = None
        
        # Moving Averages
        self.ema_values = deque(maxlen=lookback)
        self.sma_values = deque(maxlen=lookback)

    def _calculate_rsi(self, periods: int = 14, smooth: int = 1) -> Optional[float]:
        """RSI with smoothing"""
        if len(self.closes) < periods + 1:
            return None
            
        deltas = np.diff([c for c in self.closes])
        seed = deltas[-periods:]
        
        up = seed.copy()
        up[up < 0] = 0
        down = -seed.copy()
        down[down < 0] = 0
        
        rs = np.mean(up) / np.mean(down) if np.mean(down) != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        return rsi # type: ignore

    def _calculate_wt(self, n1: int = 10, n2: int = 11) -> Optional[float]:
        """Wave Trend"""
        if len(self.closes) < max(n1, n2):
            return None
            
        hlc3 = [(h + l + c) / 3 for h, l, c in zip(self.highs, self.lows, self.closes)]
        ema1 = np.mean(hlc3[-n1:])  # Simplified EMA
        
        d = np.mean([abs(hlc3[-i] - ema1) for i in range(1, n1+1)])
        ci = (hlc3[-1] - ema1) / (0.015 * d) if d != 0 else 0
        
        wt1 = np.mean([ci for _ in range(n2)])  # Simplified second smoothing
        return wt1 # type: ignore

    def _calculate_cci(self, periods: int = 20) -> Optional[float]:
        """Commodity Channel Index"""
        if len(self.closes) < periods:
            return None
            
        tp = [(h + l + c) / 3 for h, l, c in zip(self.highs, self.lows, self.closes)]
        sma = np.mean(tp[-periods:])
        mean_deviation = np.mean([abs(x - sma) for x in tp[-periods:]])
        
        cci = (tp[-1] - sma) / (0.015 * mean_deviation) if mean_deviation != 0 else 0
        return cci

    def _calculate_adx(self, periods: int = 14) -> Optional[float]:
        """Average Directional Index"""
        if len(self.closes) < periods * 2:
            return None
            
        # True Range
        tr = [max(h - l, abs(h - pc), abs(l - pc)) 
              for h, l, pc in zip(list(self.highs)[-periods:], 
                                list(self.lows)[-periods:], 
                                list(self.closes)[-periods-1:-1])]
        
        # Directional Movement
        plus_dm = [max(h - ph, 0) if (h - ph) > (pl - l) else 0 
                  for h, ph, l, pl in zip(list(self.highs)[-periods:],
                                        list(self.highs)[-periods-1:-1],
                                        list(self.lows)[-periods:],
                                        list(self.lows)[-periods-1:-1])]
        
        minus_dm = [max(pl - l, 0) if (pl - l) > (h - ph) else 0 
                   for h, ph, l, pl in zip(list(self.highs)[-periods:],
                                         list(self.highs)[-periods-1:-1],
                                         list(self.lows)[-periods:],
                                         list(self.lows)[-periods-1:-1])]
        
        tr_sum = np.sum(tr)
        plus_di = 100 * np.sum(plus_dm) / tr_sum if tr_sum != 0 else 0
        minus_di = 100 * np.sum(minus_dm) / tr_sum if tr_sum != 0 else 0
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) != 0 else 0
        adx = np.mean([dx for _ in range(periods)])  # Simplified ADX smoothing
        
        return adx # type: ignore

    def _rational_quadratic_kernel(self, distance: float) -> float:
        """Rational Quadratic Kernel"""
        return (1 + (distance * distance) / (2 * self.rq_kernel_width)) ** (-self.rq_kernel_width)

    def _gaussian_kernel(self, distance: float) -> float:
        """Gaussian Kernel"""
        return np.exp(-(distance * distance) / (2 * self.rq_kernel_width))

    def _calculate_kernel_estimate(self) -> Optional[float]:
        """Nadaraya-Watson Kernel Regression"""
        if len(self.closes) < self.kernel_lookback:
            return None
            
        prices = list(self.closes)[-self.kernel_lookback:]
        weights_rq = [self._rational_quadratic_kernel(i) for i in range(len(prices))]
        weights_g = [self._gaussian_kernel(i) for i in range(len(prices))]
        
        estimate_rq = np.sum([p * w for p, w in zip(prices, weights_rq)]) / np.sum(weights_rq)
        estimate_g = np.sum([p * w for p, w in zip(prices, weights_g)]) / np.sum(weights_g)
        
        self.yhat1.append(estimate_rq)
        self.yhat2.append(estimate_g)
        
        return estimate_rq

    def _get_lorentzian_distance(self, i: int) -> float:
        """Calculate Lorentzian distance between current and historical features"""
        if len(self.f1_array) <= i:
            return float('inf')
            
        distance = 0
        features = [
            (self.f1_array, self.f1_array[-1]),
            (self.f2_array, self.f2_array[-1]),
            (self.f3_array, self.f3_array[-1]),
            (self.f4_array, self.f4_array[-1]),
            (self.f5_array, self.f5_array[-1])
        ]
        
        for idx, (feature_array, current_value) in enumerate(features):
            if idx < self.feature_count and len(feature_array) > i:
                historical_value = list(feature_array)[i]
                distance += np.log1p(abs(current_value - historical_value))
                
        return distance

    def _check_filters(self) -> bool:
        """Check all filter conditions"""
        if len(self.closes) < 100:  # Minimum required history
            return False
            
        passes_filters = True
        
        # Volatility Filter
        if self.use_volatility_filter:
            std = np.std(list(self.closes)[-20:])
            avg_std = np.mean([np.std(list(self.closes)[i:i+20]) for i in range(-100, -20, 20)])
            passes_filters &= std <= avg_std * 2.5
        
        # Regime Filter
        if self.use_regime_filter:
            slope = np.polyfit(range(20), list(self.closes)[-20:], 1)[0]
            passes_filters &= slope > self.regime_threshold
        
        # ADX Filter
        if self.use_adx_filter:
            adx = self._calculate_adx()
            passes_filters &= adx is not None and adx > self.adx_threshold
        
        # Moving Average Filters
        if self.use_ema_filter:
            ema = np.mean(list(self.closes)[-self.ema_period:])  # Simplified EMA
            passes_filters &= self.closes[-1] > ema
            
        if self.use_sma_filter:
            sma = np.mean(list(self.closes)[-self.sma_period:])
            passes_filters &= self.closes[-1] > sma
        
        return passes_filters # type: ignore

    def update(self, high: float, low: float, close: float, volume: float) -> Tuple[int, float, float]:
        """
        Process new bar update
        Returns: (signal, signal_strength, kernel_estimate)
        signal: 1 for long, -1 for short, 0 for neutral
        """
        # Update price history
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        self.volumes.append(volume)
        
        if len(self.closes) < 20:  # Minimum required history
            return 0, 0.0, close
        
        # Calculate features
        rsi = self._calculate_rsi(14, 1)
        wt = self._calculate_wt(10, 11)
        cci = self._calculate_cci(20)
        adx = self._calculate_adx(20)
        rsi2 = self._calculate_rsi(9, 1)
        
        if None in (rsi, wt, cci, adx, rsi2):
            return 0, 0.0, close
            
        # Update feature arrays
        self.f1_array.append(rsi)
        self.f2_array.append(wt)
        self.f3_array.append(cci)
        self.f4_array.append(adx)
        self.f5_array.append(rsi2)
        
        # Calculate kernel estimate
        kernel_estimate = self._calculate_kernel_estimate()
        if kernel_estimate is None:
            kernel_estimate = close
            
        self.kernel_estimates.append(kernel_estimate)
        
        # Core ML Logic
        size = min(self.lookback-1, len(self.closes)-1)
        
        for i in range(0, size, 4):  # Process every 4th bar
            d = self._get_lorentzian_distance(i)
            
            if d >= self.last_distance:
                self.last_distance = d
                label = 1 if self.closes[min(i+4, len(self.closes)-1)] > self.closes[i] else -1
                
                self.distances.append(d)
                self.predictions.append(label)
                
                if len(self.predictions) > self.neighbors_count:
                    # Update last_distance to 75th percentile
                    sorted_distances = sorted(self.distances)
                    self.last_distance = sorted_distances[round(self.neighbors_count * 3/4)]
                    self.distances.popleft()
                    self.predictions.popleft()
        
        # Generate signal
        if not self._check_filters() or len(self.predictions) < self.neighbors_count:
            return 0, 0.0, kernel_estimate
            
        prediction_sum = sum(self.predictions)
        signal_strength = abs(prediction_sum) / self.neighbors_count
        
        # Kernel-based signal modification
        is_bullish_kernel = len(self.kernel_estimates) > 1 and self.kernel_estimates[-1] > self.kernel_estimates[-2]
        is_bearish_kernel = len(self.kernel_estimates) > 1 and self.kernel_estimates[-1] < self.kernel_estimates[-2]
        
        # Signal logic
        if prediction_sum > 0 and is_bullish_kernel and self.current_signal <= 0:
            self.current_signal = 1
            self.bars_held = 0
        elif prediction_sum < 0 and is_bearish_kernel and self.current_signal >= 0:
            self.current_signal = -1
            self.bars_held = 0
        else:
            self.bars_held += 1
           
        # Handle exit conditions
        if self.use_kernel_smoothing:
            # Kernel-based exits
            if (self.current_signal == 1 and is_bearish_kernel) or \
               (self.current_signal == -1 and is_bullish_kernel):
                self.current_signal = 0
                self.bars_held = 0
        else:
            # Time-based exits
            if self.bars_held >= 4:
                self.current_signal = 0
                self.bars_held = 0

        # Early signal flip detection
        if len(self.kernel_estimates) >= 3:
            prev_signal = 1 if self.kernel_estimates[-3] < self.kernel_estimates[-2] else -1
            curr_signal = 1 if self.kernel_estimates[-2] < self.kernel_estimates[-1] else -1

            if prev_signal != curr_signal and self.bars_held < 4:
                self.current_signal = 0
                self.bars_held = 0

        return self.current_signal, signal_strength, kernel_estimate

def example():
   # Example usage
   classifier = LorentzianClassifier(
       lookback=2000,
       neighbors_count=8,
       feature_count=5,
       kernel_lookback=8,
       rq_kernel_width=8.0
   )
   
   # Simulate some data
   for i in range(100):
       signal, strength, kernel = classifier.update(
           high=100 + i * 0.1 + np.random.random(),
           low=99 + i * 0.1 + np.random.random(),
           close=99.5 + i * 0.1 + np.random.random(),
           volume=1000 + np.random.random() * 100
       )
       if signal != 0:
           print(f"Bar {i}: Signal={signal}, Strength={strength:.2f}, Kernel={kernel:.2f}")

if __name__ == "__main__":
   example()