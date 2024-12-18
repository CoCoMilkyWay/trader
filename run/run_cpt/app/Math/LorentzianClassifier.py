import numpy as np
from collections import deque
from typing import Tuple
from dataclasses import dataclass

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
                 use_volatility_filter: bool = True,
                 use_regime_filter: bool = True,
                 use_adx_filter: bool = False,
                 regime_threshold: float = -0.1,
                 adx_threshold: int = 20,
                 use_ema_filter: bool = False,
                 use_sma_filter: bool = False,
                 ema_period: int = 200,
                 sma_period: int = 200,
                 use_kernel_filter: bool = True,
                 show_kernel_estimate: bool = True,
                 use_kernel_smoothing: bool = False,
                 kernel_lookback: int = 8,
                 relative_weighting: float = 8.0,
                 regression_level: int = 25,
                 lag: int = 2,
                ):
        
        # # TODO:
        lookback = 100
        
        self.lookback = lookback
        self.neighbors_count = neighbors_count
        self.feature_count = feature_count
        
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
        
        # Kernel settings
        self.use_kernel_filter = use_kernel_filter
        self.show_kernel_estimate = show_kernel_estimate
        self.use_kernel_smoothing = use_kernel_smoothing
        self.kernel_lookback = kernel_lookback
        self.relative_weighting = relative_weighting
        self.regression_level = regression_level
        self.lag = lag

        # Price history - maintain exactly like Pine
        self.closes = deque(maxlen=lookback)
        self.highs = deque(maxlen=lookback)
        self.lows = deque(maxlen=lookback)
        self.volumes = deque(maxlen=lookback)
        
        # Feature Arrays - match Pine exactly
        self.f1_array = deque(maxlen=lookback)  # RSI(14)
        self.f2_array = deque(maxlen=lookback)  # WT(10,11)
        self.f3_array = deque(maxlen=lookback)  # CCI(20) 
        self.f4_array = deque(maxlen=lookback)  # ADX(20)
        self.f5_array = deque(maxlen=lookback)  # RSI(9)
        
        # ML Components - maintain arrays like Pine
        self.distances = deque(maxlen=neighbors_count)
        self.predictions = deque(maxlen=neighbors_count)
        self.last_distance = -1.0
        
        # Kernel Components - match Pine's array handling
        self.kernel_estimates = deque(maxlen=lookback)
        self.yhat1 = deque(maxlen=lookback)  # Rational Quadratic
        self.yhat2 = deque(maxlen=lookback)  # Gaussian
        
        # State Management - exactly like Pine
        self.bars_held = 0
        self.current_signal = 0
        self.signal = 0
        
        # Moving Averages - keep Pine's array style
        self.ema_values = deque(maxlen=lookback)
        self.sma_values = deque(maxlen=lookback)

    def _calculate_rsi(self, periods: int = 14, smooth: int = 1) -> float:
        """RSI calculation matching Pine exactly"""
        if len(self.closes) < periods + 1:
            return 0.0
            
        # Match Pine's exact RSI calculation
        deltas = np.diff([c for c in self.closes])[-periods:]
        gains = np.array([max(d, 0) for d in deltas])
        losses = np.array([abs(min(d, 0)) for d in deltas])
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return float(rsi)

    def _calculate_wt(self, n1: int = 10, n2: int = 11) -> float:
        """Wave Trend calculation matching Pine exactly"""
        if len(self.closes) < max(n1, n2):
            return 0.0
            
        # Match Pine's HLC3 calculation
        hlc3 = [(h + l + c) / 3.0 for h, l, c in zip(self.highs, self.lows, self.closes)]
        
        # Use exact Pine EMA formula
        ema1 = hlc3[-n1]
        alpha = 2.0 / (n1 + 1.0)
        for i in range(-n1+1, 0):
            ema1 = alpha * hlc3[i] + (1.0 - alpha) * ema1
            
        # Match Pine's mean deviation
        d = np.mean([abs(hlc3[-i] - ema1) for i in range(1, n1+1)])
        
        # Exact ci calculation
        ci = (hlc3[-1] - ema1) / (0.015 * d) if d != 0 else 0
        
        # Match Pine's WT1
        wt1 = ci
        alpha2 = 2.0 / (n2 + 1.0)
        for i in range(n2-1):
            wt1 = alpha2 * wt1 + (1.0 - alpha2) * wt1
            
        return float(wt1)

    def _calculate_cci(self, periods: int = 20) -> float:
        """CCI calculation matching Pine exactly"""
        if len(self.closes) < periods:
            return 0.0
            
        # Match Pine's exact CCI calculation
        tp = [(h + l + c) / 3.0 for h, l, c in zip(self.highs, self.lows, self.closes)][-periods:]
        sma_tp = np.mean(tp)
        mean_deviation = np.mean([abs(x - sma_tp) for x in tp])
        
        cci = (tp[-1] - sma_tp) / (0.015 * mean_deviation) if mean_deviation != 0 else 0
        return float(cci)

    def _calculate_adx(self, periods: int = 14) -> float:
        """ADX calculation matching Pine exactly"""
        if len(self.closes) < periods * 2:
            return 0.0
            
        # Calculate True Range exactly like Pine
        tr = [max(h - l, abs(h - pc), abs(l - pc)) 
              for h, l, pc in zip(list(self.highs)[-periods:], 
                                list(self.lows)[-periods:], 
                                list(self.closes)[-periods-1:-1])]
        
        # Calculate +DM and -DM exactly like Pine
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
        
        # Match Pine's exact smoothing
        tr_sum = sum(tr)
        plus_di = 100.0 * sum(plus_dm) / tr_sum if tr_sum != 0 else 0
        minus_di = 100.0 * sum(minus_dm) / tr_sum if tr_sum != 0 else 0
        
        # Calculate DX and ADX exactly like Pine
        dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) != 0 else 0
        adx = float(np.mean([dx for _ in range(periods)]))
        
        return adx

    def _calculate_ma(self, ma_length: int, ma_type: str = 'ema') -> float:
        """Calculate EMA or SMA exactly as Pine"""
        if len(self.closes) < ma_length:
            return 0.0

        closes_list = list(self.closes)
        prices = closes_list[-ma_length:]

        if ma_type == 'ema':
            # Match Pine's exact EMA calculation
            ema = prices[0]
            alpha = 2.0 / (ma_length + 1.0)
            for price in prices[1:]:
                ema = alpha * price + (1.0 - alpha) * ema
            return float(ema)
        else:
            # Match Pine's exact SMA calculation
            return float(np.mean(prices))

    def _calculate_kernel_estimates(self) -> Tuple[float, float]:
        """Calculate both kernel estimates matching Pine"""
        h = self.kernel_lookback
        if len(self.closes) < h:
            return 0.0, 0.0
            
        # Match Pine's exact kernel calculations
        closes = np.array(list(self.closes)[-h:])
        distances = np.arange(float(len(closes)))
        
        # Use exact kernel functions
        rq_weights = self._rational_quadratic_kernel(distances)
        g_weights = self._gaussian_kernel(distances)
        
        # Match Pine's weighted averages
        yhat1 = np.sum(closes * rq_weights) / np.sum(rq_weights)
        yhat2 = np.sum(closes * g_weights) / np.sum(g_weights)
        
        return float(yhat1), float(yhat2)

    def _rational_quadratic_kernel(self, distances: np.ndarray) -> np.ndarray:
        """Rational Quadratic Kernel matching Pine"""
        # Use exact Pine formula
        return (1.0 + (distances ** 2) / (2.0 * self.relative_weighting)) ** (-self.relative_weighting)

    def _gaussian_kernel(self, distances: np.ndarray) -> np.ndarray:
        """Gaussian Kernel matching Pine"""
        # Use exact Pine formula
        return np.exp(-distances ** 2 / (2.0 * self.relative_weighting))

    def _check_filters(self) -> bool:
        """All filters matching Pine exactly"""
        if len(self.closes) < 100:
            return False
            
        passes_filters = True
        
        # Volatility Filter - match Pine's calculation
        if self.use_volatility_filter:
            recent_std = np.std(list(self.closes)[-20:])
            historical_std = np.mean([np.std(list(self.closes)[i:i+20]) 
                                    for i in range(-100, -20, 20)])
            passes_filters &= bool(recent_std <= historical_std * 2.5)
        
        # Regime Filter - exact Pine calculation
        if self.use_regime_filter:
            x = np.arange(20)
            y = list(self.closes)[-20:]
            slope = np.polyfit(x, y, 1)[0]
            passes_filters &= bool(slope > self.regime_threshold)
        
        # ADX Filter - match Pine
        if self.use_adx_filter:
            adx = self._calculate_adx()
            passes_filters &= bool(adx > self.adx_threshold)
            
        # MA Filters - exact Pine calculations
        if self.use_ema_filter:
            ema = self._calculate_ma(self.ema_period, 'ema')
            passes_filters &= bool(self.closes[-1] > ema)
            
        if self.use_sma_filter:
            sma = self._calculate_ma(self.sma_period, 'sma')
            passes_filters &= bool(self.closes[-1] > sma)
        
        return passes_filters

    def _get_lorentzian_distance(self, i: int) -> float:
        """Exact Lorentzian distance matching Pine"""
        if len(self.f1_array) <= i:
            return float('inf')
            
        # Use exact Pine log1p formula for each feature
        distance = (
            np.log1p(abs(self.f1_array[-1] - list(self.f1_array)[i])) +
            np.log1p(abs(self.f2_array[-1] - list(self.f2_array)[i])) +
            np.log1p(abs(self.f3_array[-1] - list(self.f3_array)[i])) +
            np.log1p(abs(self.f4_array[-1] - list(self.f4_array)[i])) +
            np.log1p(abs(self.f5_array[-1] - list(self.f5_array)[i]))
        )
        
        return float(distance)

    def update(self, high: float, low: float, close: float, volume: float) -> Tuple[bool, bool]:
        """Process new bar exactly as Pine"""
        # Update price history
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        self.volumes.append(volume)
        
        if len(self.closes) < 20:
            return False, False

        # Calculate features - match Pine exactly
        rsi14 = self._calculate_rsi(14, 1)
        wt = self._calculate_wt(10, 11)
        cci = self._calculate_cci(20)
        adx = self._calculate_adx(20)
        rsi9 = self._calculate_rsi(9, 1)
        
        # Update feature arrays in Pine's order
        self.f1_array.append(rsi14)
        self.f2_array.append(wt)
        self.f3_array.append(cci)
        self.f4_array.append(adx)
        self.f5_array.append(rsi9)
        
        # Calculate kernel estimates like Pine
        yhat1, yhat2 = self._calculate_kernel_estimates()
        self.yhat1.append(yhat1)
        self.yhat2.append(yhat2)
        self.kernel_estimates.append(yhat1)
        
        # Core ML Logic - exact Pine implementation
        size = min(self.lookback-1, len(self.closes)-1)
        
        for i in range(0, size, 4):  # Pine's 4-bar spacing
            d = self._get_lorentzian_distance(i)
            
            if d >= self.last_distance:
                self.last_distance = d
                # Match Pine's exact label calculation
                next_idx = min(i+4, len(self.closes)-1)
                label = 1 if self.closes[next_idx] > self.closes[i] else -1
                
                self.distances.append(d)
                self.predictions.append(label)
                
                if len(self.predictions) > self.neighbors_count:
                    # Match Pine's exact percentile calculation
                    sorted_dists = sorted(self.distances)
                    percentile_idx = round(self.neighbors_count * 0.75)  # 75th percentile
                    self.last_distance = sorted_dists[percentile_idx]
                    self.distances.popleft()
                    self.predictions.popleft()

        # Generate signal exactly like Pine
        if not self._check_filters() or len(self.predictions) < self.neighbors_count:
            return False, False
            
        prediction_sum = sum(self.predictions)
        
        # Kernel trend detection matching Pine
        is_bullish_kernel = len(self.kernel_estimates) > 1 and self.kernel_estimates[-1] > self.kernel_estimates[-2]
        is_bearish_kernel = len(self.kernel_estimates) > 1 and self.kernel_estimates[-1] < self.kernel_estimates[-2]
        
        # Signal logic matching Pine exactly
        prev_signal = self.signal
        if prediction_sum > 0 and is_bullish_kernel and self.current_signal <= 0:
            self.current_signal = 1
            self.bars_held = 0
            self.signal = 1
        elif prediction_sum < 0 and is_bearish_kernel and self.current_signal >= 0:
            self.current_signal = -1
            self.bars_held = 0
            self.signal = -1
        else:
            self.bars_held += 1

        # Kernel Regression Filters: Filters based on Nadaraya-Watson Kernel Regression using the Rational Quadratic Kernel
        is_bullish_smooth = len(self.yhat2) > 0 and len(self.yhat1) > 0 and self.yhat2[-1] >= self.yhat1[-1]
        is_bearish_smooth = len(self.yhat2) > 0 and len(self.yhat1) > 0 and self.yhat2[-1] <= self.yhat1[-1]
        
        # Exit conditions matching Pine
        if self.use_kernel_smoothing:
            if ((self.current_signal == 1 and is_bearish_smooth) or 
                (self.current_signal == -1 and is_bullish_smooth)):
                self.current_signal = 0
                self.bars_held = 0
        else:
            if self.bars_held >= 4:  # Pine's fixed 4-bar exit
                self.current_signal = 0
                self.bars_held = 0
        
        # Early signal flip detection matching Pine exactly
        if len(self.kernel_estimates) >= 3:
            prev_signal = 1 if self.kernel_estimates[-3] < self.kernel_estimates[-2] else -1
            curr_signal = 1 if self.kernel_estimates[-2] < self.kernel_estimates[-1] else -1
            
            if prev_signal != curr_signal and self.bars_held < 4:
                self.current_signal = 0
                self.bars_held = 0

        # MA trend conditions matching Pine
        is_ema_uptrend = True
        is_ema_downtrend = True
        if self.use_ema_filter:
            ema = self._calculate_ma(self.ema_period, 'ema')
            is_ema_uptrend = self.closes[-1] > ema
            is_ema_downtrend = self.closes[-1] < ema

        is_sma_uptrend = True
        is_sma_downtrend = True
        if self.use_sma_filter:
            sma = self._calculate_ma(self.sma_period, 'sma')
            is_sma_uptrend = self.closes[-1] > sma
            is_sma_downtrend = self.closes[-1] < sma

        # Signal change detection exactly like Pine
        is_different_signal = self.signal != prev_signal

        # Buy/Sell conditions matching Pine exactly
        is_buy_signal = (self.signal == 1 and is_ema_uptrend and is_sma_uptrend)
        is_sell_signal = (self.signal == -1 and is_ema_downtrend and is_sma_downtrend)
        
        # Entry signals with Pine's exact logic
        is_new_buy_signal = is_buy_signal and is_different_signal
        is_new_sell_signal = is_sell_signal and is_different_signal
        
        # Kernel trend conditions matching Pine
        if self.use_kernel_filter:
            if self.use_kernel_smoothing:
                is_new_buy_signal &= is_bullish_smooth
                is_new_sell_signal &= is_bearish_smooth
            else:
                is_new_buy_signal &= is_bullish_kernel
                is_new_sell_signal &= is_bearish_kernel

        # if is_new_buy_signal or is_new_sell_signal:
        #     print(is_new_buy_signal, is_new_sell_signal)
        return is_new_buy_signal, is_new_sell_signal