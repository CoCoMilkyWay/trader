import numpy as np
from collections import deque
from typing import Tuple
from dataclasses import dataclass

# https://www.youtube.com/watch?v=AdINVvnJfX4&ab_channel=JustinDehorty

# ┌────────────────────────┬────────────────────────────┬──────────────────────────┬───────────────────┐
# │        Job             │         ML Algos           │      Strengths           │    Limitations    │
# ├────────────────────────┼────────────────────────────┼──────────────────────────┼───────────────────┤
# │ REGIME DETECTION       │ • Variational Autoencoders │ • Learns latent space    │ • Training time   │
# │ - Market states        │ • Temporal VAEs            │ • Handles time series    │ • Parameter tuning│
# │ - State transitions    │ • Transformers w/clustering│ • Captures transitions   │ • Data hungry     │
# │ - Feature compress     │ • Neural HMMs              │ • Probabilistic output   │                   │
# ├────────────────────────┼────────────────────────────┼──────────────────────────┼───────────────────┤
# │ PREDICTION             │ • LightGBM/XGBoost         │ • Fast inference         │ • Regime dependent│
# │ - Trend following      │ • Neural Flows             │ • Non-linear patterns    │ • Can overfit     │
# │ - Mean reversion       │ • Temporal Fusion Trans.   │ • Multi-horizon pred.    │ • Need fine-tuning│
# │ - Volatility pred.     │ • Quantile Regression      │ • Uncertainty estimate   │                   │
# ├────────────────────────┼────────────────────────────┼──────────────────────────┼───────────────────┤
# │ TRUST/CONFIDENCE       │ • Deep Ensembles           │ • Uncertainty quant.     │ • Computational   │
# │ - Model uncertainty    │ • Bayesian Neural Nets     │ • Multiple views         │ • Complex setup   │
# │ - Prediction conf.     │ • Gaussian Processes       │ • Probabilistic          │ • Scaling issues  │
# │ - Risk assessment      │ • Isotonic Calibration     │ • Well-calibrated prob.  │                   │
# ├────────────────────────┼────────────────────────────┼──────────────────────────┼───────────────────┤
# │ RISK MANAGEMENT        │ • CVaR Neural Nets         │ • Risk-aware learning    │ • Complex loss    │
# │ - Position sizing      │ • Reinforcement Learning   │ • Dynamic adaptation     │ • Hard to train   │
# │ - Portfolio opt.       │ • Multi-objective opt.     │ • Portfolio-level view   │ • Needs safeguards│
# │ - Dynamic hedging      │ • Adversarial Training     │ • Robustness to noise    │                   │
# ├────────────────────────┼────────────────────────────┼──────────────────────────┼───────────────────┤
# │ EXECUTION              │ • RL with transformers     │ • Adaptive to market     │ • Market impact   │
# │ - Timing               │ • Multi-agent systems      │ • Learning from flow     │ • Need live data  │
# │ - Size splitting       │ • Imitation Learning       │ • Cost optimization      │ • Complex deploy  │
# │ - Venue selection      │ • Online Learning          │ • Real-time adaptation   │                   │
# └────────────────────────┴────────────────────────────┴──────────────────────────┴───────────────────┘
# 
# Latest Innovations:
# 
# 1. Regime Detection:
#   - Neural HMMs replacing traditional HMMs
#   - Temporal VAEs for complex state spaces
#   - Attention mechanisms for regime transitions
# 
# 2. Prediction:
#   - Temporal Fusion Transformers
#   - Neural ODEs for continuous-time modeling
#   - Quantile regression for distribution prediction
# 
# 3. Trust/Confidence:
#   - Deep Ensembles with diversity measures
#   - Bayesian Deep Learning advances
#   - Better uncertainty calibration methods
# 
# 4. Risk Management:
#   - CVaR optimization in neural nets
#   - Multi-agent RL for portfolio management
#   - Adversarial training for robustness
# 
# 5. Execution:
#   - Transformer-based RL
#   - Flow-based execution modeling
#   - Online adaptation methods

# 
# Key insights:
# 
# 1. Unsupervised:
#    - t-SNE: Dimensionality reduction
#    - SOM: Self-organizing maps
#    - DBSCAN: Clustering
#    - TDA: Topological analysis
# 
# 2. Supervised:
#    - SVM: Classification/regression
#    - Echo State Network: Time series prediction
#    - Attention NN: Sequence learning
# 
# 3. Semi-supervised:
#    - Lorentzian Classifier: Uses price action as implicit labels
#    - Can be modified to use explicit labels if available
# 
# 4. Training Requirements:
#    - Some need explicit training (NN, SVM)
#    - Some just need parameter tuning (DBSCAN, TDA)
#    - Some learn online/incrementally (SOM)

# Time Series Data (OHLCV)
#                                                                            |
#                                                                            v
#                                             +------------------------[Data Ingestion]------------------------+
#                                             |                              |                                 |
#                                             v                              v                                 v
#                                      [Price History]                [Feature Arrays]                 [ML Components]
#                                     /    |    |    \             /      |      |      \              /      |      \
#                                 Highs Lows Close Volume       RSI(14) WT(10) CCI(20) ADX(20)    Distances  Preds  LastDist
#                                     \    |    |    /             \      |      |      /              \      |      /
#                                      \   |    |   /               \     |      |     /                \     |     /
#                                       \  |    |  /                 \    |      |    /                  \    |    /
#                                        \ |    | /                   \   |      |   /                    \   |   /
#                      +------------------[State]------------------+   \  |      |  /   +------------------[Kernel]----------+
#                      |                     |                     |    \ |      | /    |                   |                |
#                      v                     v                     v     \|      |/     v                   v                v
#               [Current Signal]      [Bars Held]            [MA Arrays] \|      /  [RQ Kernel]      [Gaussian K.]    [Estimates]
#                      |                     |                     |      \    /        |                   |                |
#                      |                     |                     |       \  /         |                   |                |
#                      |                     |                     v        \/          v                   v                v
#                      |                     |              +----[Filters]--/\---[Distance Calc.]----[Kernel Regr.]---[Smoothing]
#                      |                     |              |              /  \         |                   |                |
#                      |                     |              v             /    \        v                   v                |
#                      |                     |        [Volatility]-------'      `--[Lorentzian]---[Rational Quad.]           |
#                      |                     |              |                           |            /            \          |
#                      |                     |              v                           v           /              \         |
#                      |                     |          [Regime]---------------------->[KNN]-------'                \        |
#                      |                     |              |                           |                            \       |
#                      |                     |              v                           v                             \      |
#                      |                     |            [ADX]---------------->[Pattern Matching]                     \     |
#                      |                     |              |                           |                               \    |
#                      |                     |              v                           v                                \   |
#                      |                     +---------->[MA Trend]------------->[Neighbor Selection]                     \  |
#                      |                                    |                           |                                  \ |
#                      |                                    v                           v                                   \|
#                      +-------------------------------->[Signal]<----------------[Predictions]<--------------------------[Exit]
#                                                          |
#                                                          v
#                                                   [Trade Direction]
#                                                   (-1, 0, or +1)

# Time ────────────────────────────────────────────────────────────────────────────>
#                                                                         
# Price     ╭─╮                 ╭────╮         ╭───╮                ╭───╮    
#      ╭────╯ ╰─╮            ╭──╯    ╰───╮    ╭╯   ╰──╮         ╭──╯   ╰───╮
#      │        ╰────╮    ╭──╯           ╰────╯       ╰──────────╯          │
#      ╰─╮           ╰────╯                                                  ╰─
#        ╰───╮                           [Price Action]
# 
# RSI(14)   ────────⌈70                                                     
#           ────────⌈50    ∿∿∿╭─╮  ╭───╮   ╭╮  ╭─╮  ╭╮   ╭─╮    ╭╮  ╭─╮  
#           ────────⌈30   ╭╯  ╰╯╰──╯   ╰───╯╰──╯ ╰──╯╰───╯ ╰────╯╰──╯ ╰──
# 
# WT(10,11) ⌈
#           │   ╭╮    ╭╮  ╭───╮    ╭──╮   ╭╮    ╭─╮   ╭╮    ╭─╮   ╭╮   
#           ⌈0──╯╰────╯╰──╯   ╰────╯  ╰───╯╰────╯ ╰───╯╰────╯ ╰───╯╰───
#           ⌊               [Wave Trend Oscillator]
# 
# CCI(20)   ⌈100     ╭╮    ╭───╮     ╭──╮    ╭╮     ╭──╮    ╭╮    ╭──╮  
#           ⌈0───────╯╰────╯   ╰─────╯  ╰────╯╰─────╯  ╰────╯╰────╯  ╰──
#           ⌊-100
# 
# ADX(20)   ───────────────────────────────────────────────────────────────
# Trend     ⌈40     ╱╲      ╱╲       ╱╲      ╱╲       ╱╲      ╱╲      ╱╲ 
# Strength  ⌈20────╱──╲────╱──╲─────╱──╲────╱──╲─────╱──╲────╱──╲────╱──╲
#           ⌊0    ╱    ╲  ╱    ╲   ╱    ╲  ╱    ╲   ╱    ╲  ╱    ╲  ╱    
# 
# Kernel    ⌈     ∿∿∿∿    ∿∿∿∿     ∿∿∿∿    ∿∿∿∿     ∿∿∿∿    ∿∿∿∿    ∿∿∿∿
# RQ        ⌈0───∿────∿──∿────∿───∿────∿──∿────∿───∿────∿──∿────∿──∿────
# Est.      ⌊    ∿    ∿  ∿    ∿   ∿    ∿  ∿    ∿   ∿    ∿  ∿    ∿  ∿    
# 
# Kernel    ⌈      🠕      🠕        🠕      🠕        🠕      🠕       🠕
# Smoothed  ⌈0─────●──────●────────●──────●────────●──────●───────●──────
#           ⌊      🠗      🠗        🠗      🠗        🠗      🠗       🠗    
# 
# Filters:  ────────────────────────────────────────────────────────────────
# Vol.         P  P  P  P  F  P  P  P  F  F  P  P  P  P  F  P  P  P  P  F
# Regime       P  P  F  P  P  P  P  F  P  P  P  P  F  P  P  P  P  F  P  P
# ADX          P  F  P  P  P  P  F  P  P  P  P  F  P  P  P  P  F  P  P  P
# (P=Pass, F=Fail)
# 
# KNN         ⌈8    ∿∿∿∿    ∿∿∿∿     ∿∿∿∿    ∿∿∿∿     ∿∿∿∿    ∿∿∿∿    ∿∿
# Neighbors   ⌈4───∿────∿──∿────∿───∿────∿──∿────∿───∿────∿──∿────∿──∿──
# Count       ⌊0   ∿    ∿  ∿    ∿   ∿    ∿  ∿    ∿   ∿    ∿  ∿    ∿  ∿  
# 
# Final      ╭───╮      ╭───╮       ╭───╮      ╭───╮       ╭───╮      ╭───
# Signal     │   │      │   │       │   │      │   │       │   │      │   
#           ─╯   ╰──────╯   ╰───────╯   ╰──────╯   ╰───────╯   ╰──────╯   
# 
# Legend:
# ╭╮╯╰ : Price/Indicator Movement    ∿∿∿ : Kernel Estimation
# ─── : Baseline/Threshold          🠕🠗 : Directional Change
# P/F : Filter Pass/Fail            ●  : Smoothed Point

class LorentzianClassifier:
    def __init__(
        self,
        lookback: int = 2000,
        neighbors_count: int = 8,
        
        use_volatility_filter: bool = False,
        use_regime_filter: bool = False,
        use_adx_filter: bool = False,
        regime_threshold: float = -0.1,
        adx_threshold: int = 20,
        
        use_ema_filter: bool = False,
        use_sma_filter: bool = False,
        ema_period: int = 200,
        sma_period: int = 200,
        
        use_kernel_filter: bool = True,
        use_kernel_smoothing: bool = False,
        kernel_lookback: int = 8, # the base assumption is flawed, this cannot be too large
        relative_weighting: float = 8.0, # bandwidth
        ):
        
        # # TODO:
        lookback = 100
        
        self.lookback = lookback
        self.neighbors_count = neighbors_count
        
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
        self.use_kernel_smoothing = use_kernel_smoothing
        self.kernel_lookback = kernel_lookback
        self.relative_weighting = relative_weighting

        # Price history
        self.closes = deque(maxlen=lookback)
        self.highs = deque(maxlen=lookback)
        self.lows = deque(maxlen=lookback)
        self.volumes = deque(maxlen=lookback)
        
        # Feature Arrays
        self.f1_array = deque(maxlen=lookback)  # RSI(14)
        self.f2_array = deque(maxlen=lookback)  # WT(10,11)
        self.f3_array = deque(maxlen=lookback)  # CCI(20) 
        self.f4_array = deque(maxlen=lookback)  # ADX(20)
        self.f5_array = deque(maxlen=lookback)  # RSI(9)
        
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
        self.signal = 0
        
        # Moving Averages
        self.ema_values = deque(maxlen=lookback)
        self.sma_values = deque(maxlen=lookback)

    def _calculate_rsi(self, periods: int = 14, smooth: int = 1) -> float:
        """RSI calculation"""
        if len(self.closes) < periods + 1:
            return 0.0
            
        # Match exact RSI calculation
        deltas = np.diff([c for c in self.closes])[-periods:]
        gains = np.array([max(d, 0) for d in deltas])
        losses = np.array([abs(min(d, 0)) for d in deltas])
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return float(rsi)

    def _calculate_wt(self, n1: int = 10, n2: int = 11) -> float:
        """Wave Trend calculation"""
        if len(self.closes) < max(n1, n2):
            return 0.0
            
        # Match HLC3 calculation
        hlc3 = [(h + l + c) / 3.0 for h, l, c in zip(self.highs, self.lows, self.closes)]
        
        # EMA formula
        ema1 = hlc3[-n1]
        alpha = 2.0 / (n1 + 1.0)
        for i in range(-n1+1, 0):
            ema1 = alpha * hlc3[i] + (1.0 - alpha) * ema1
            
        # Match mean deviation
        d = np.mean([abs(hlc3[-i] - ema1) for i in range(1, n1+1)])
        
        # Exact ci calculation
        ci = (hlc3[-1] - ema1) / (0.015 * d) if d != 0 else 0
        
        # Match WT1
        wt1 = ci
        alpha2 = 2.0 / (n2 + 1.0)
        for i in range(n2-1):
            wt1 = alpha2 * wt1 + (1.0 - alpha2) * wt1
            
        return float(wt1)

    def _calculate_cci(self, periods: int = 20) -> float:
        """CCI calculation"""
        if len(self.closes) < periods:
            return 0.0
            
        # Match exact CCI calculation
        tp = [(h + l + c) / 3.0 for h, l, c in zip(self.highs, self.lows, self.closes)][-periods:]
        sma_tp = np.mean(tp)
        mean_deviation = np.mean([abs(x - sma_tp) for x in tp])
        
        cci = (tp[-1] - sma_tp) / (0.015 * mean_deviation) if mean_deviation != 0 else 0
        return float(cci)

    def _calculate_adx(self, periods: int = 14) -> float:
        """ADX calculation"""
        if len(self.closes) < periods * 2:
            return 0.0
            
        # Calculate True Range
        tr = [max(h - l, abs(h - pc), abs(l - pc)) 
              for h, l, pc in zip(list(self.highs)[-periods:], 
                                list(self.lows)[-periods:], 
                                list(self.closes)[-periods-1:-1])]
        
        # Calculate +DM and -DM
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
        
        # Match exact smoothing
        tr_sum = sum(tr)
        plus_di = 100.0 * sum(plus_dm) / tr_sum if tr_sum != 0 else 0
        minus_di = 100.0 * sum(minus_dm) / tr_sum if tr_sum != 0 else 0
        
        # Calculate DX and ADX
        dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) != 0 else 0
        adx = float(np.mean([dx for _ in range(periods)]))
        
        return adx

    def _calculate_ma(self, ma_length: int, ma_type: str = 'ema') -> float:
        """Calculate EMA or SMA"""
        if len(self.closes) < ma_length:
            return 0.0

        closes_list = list(self.closes)
        prices = closes_list[-ma_length:]

        if ma_type == 'ema':
            # Match exact EMA calculation
            ema = prices[0]
            alpha = 2.0 / (ma_length + 1.0)
            for price in prices[1:]:
                ema = alpha * price + (1.0 - alpha) * ema
            return float(ema)
        else:
            # Match exact SMA calculation
            return float(np.mean(prices))

    def _calculate_kernel_estimates(self) -> Tuple[float, float]:
        """Calculate both kernel estimates"""
        h = self.kernel_lookback
        if len(self.closes) < h:
            return 0.0, 0.0
            
        # Match exact kernel calculations
        closes = np.array(list(self.closes)[-h:])
        distances = np.arange(float(len(closes)))
        
        
        # ====Kernel Density Estimator(KDE):====
        # Given: observations {x1,x2,...,xn} 
        # (NOTE: this theory also holds for multivariates x)
        # Assume:
        #   1. observations are i.i.d.
        #   2. true pdf f(x) is smooth and continuous
        #   3. Kernel function K(u) is symmetric and integrates to 1
        # Result:
        # KDE = f̂_H(x) = (1 / (n * h)) * sum[i=1 to n] { K((x - x_i) / h) } Converges to f(x) as
        #   1. h(bandwidth) -> 0 as n -> inf
        #   2. nh -> inf as n -> inf
        #   by decomposing Mean-Square-Error as Bias and Variance
        # Thus for multivariate x:
        #   KDE = f̂_H(x,y) = (1/n) ∑[i=1 to n] K_H((x-Xi, y-Yi))
        #       = (1/n) ∑[i=1 to n] (1/2π|H|^(1/2)) exp(-1/2((x-Xi, y-Yi)H^(-1)(x-Xi, y-Yi)ᵀ))
        #   this simplifies to (1/n) ∑[i=1 to n] K_h(x-xi)K_h(y-yi) when:
        #   1. Independence between x and y (product kernel)
        #   2. Same bandwidth h for both dimensions
        #   3. Separable kernel function
        
        # 
        # ====Nadaraya–Watson kernel regression:====
        # By Assuming joint distributions as KDE(Kernel Density Estimator):
        #   NOTE: this include all(6) assumptions made above :(
        #   f̂(x, y) = (1 / n) * Σ (from i=1 to n) [K_h(x - x_i) * K_h(y - y_i)]
        #   f̂(x) = (1 / n) * Σ (from i=1 to n) [K_h(x - x_i)]
        # E[Y|X=x] = Nadaraya–Watson-Estimator 
        #   = (sum[i=1 to n] { K_h(x - x_i) * y_i }) / (sum[j=1 to n] { K_h(x - x_j) })

        # 1. Given the assumptions above, the choice of kernel functions doesn't matter to
        #   the shape of the true pdf, but the following needs to be considered:
        #   finite sample performance/computational efficiency/boundary bias/smoothness of estimates
        # 2. However, choosing the same kernel distribution as true pdf can drastically improve
        #   the finite sample performance
        # 3. choice of bandwidth (h) is sometimes more important
        
        # we see that Nadaraya–Watson kernel regression made lots of assumptions
        # there are other estimators that have less assumptions

        # Scaled Kernels Weights: K(u/h)/h
        rq_weights = self._rational_quadratic_kernel(distances) # aka. student-t kernel(fat-tail)
        g_weights = self._gaussian_kernel(distances)
        
        # Nadaraya–Watson-Estimator
        yhat1 = np.sum(closes * rq_weights) / np.sum(rq_weights)
        yhat2 = np.sum(closes * g_weights) / np.sum(g_weights)
        
        return float(yhat1), float(yhat2)

    def _rational_quadratic_kernel(self, distances: np.ndarray) -> np.ndarray:
        """Rational Quadratic Kernel"""
        return (1.0 + (distances ** 2) / (2.0 * self.relative_weighting)) ** (-self.relative_weighting)

    def _gaussian_kernel(self, distances: np.ndarray) -> np.ndarray:
        """Gaussian Kernel"""
        return np.exp(-distances ** 2 / (2.0 * self.relative_weighting))

    def _check_filters(self) -> bool:
        """All filters"""
        if len(self.closes) < 100:
            return False
            
        passes_filters = True
        
        # Volatility Filter
        if self.use_volatility_filter:
            recent_std = np.std(list(self.closes)[-20:])
            historical_std = np.mean([np.std(list(self.closes)[i:i+20]) 
                                    for i in range(-100, -20, 20)])
            passes_filters &= bool(recent_std <= historical_std * 2.5)
        
        # Regime Filter
        if self.use_regime_filter:
            x = np.arange(20)
            y = list(self.closes)[-20:]
            slope = np.polyfit(x, y, 1)[0]
            passes_filters &= bool(slope > self.regime_threshold)
        
        # ADX Filter
        if self.use_adx_filter:
            adx = self._calculate_adx()
            passes_filters &= bool(adx > self.adx_threshold)
            
        # MA Filters
        if self.use_ema_filter:
            ema = self._calculate_ma(self.ema_period, 'ema')
            passes_filters &= bool(self.closes[-1] > ema)
            
        if self.use_sma_filter:
            sma = self._calculate_ma(self.sma_period, 'sma')
            passes_filters &= bool(self.closes[-1] > sma)
        
        return passes_filters

    def _get_lorentzian_distance(self, i: int) -> float:
        """Exact Lorentzian distance"""
        
        #                        [RSI]
        #                          |                           
        #                          |                           
        #                          |                           
        #                          ::....                      
        #                          ••:::•::..                  
        #                          :••::••....::.              
        #                          •••••::••:...:•.            
        #                          :::•••:•••::.:•..           
        #                          •:.:•::::::...:..           
        #  |--------.:•••..•••••••:••:...:::•:•:..:..----------[ADX]    
        #  0                       .
        #                          •
        #                          •
        #                          :
        #                          .
        #                          :
        #                          |
        #                          |
        #                          |
        #                          |
        #                         _|_ 0
        #                          
        #        Figure 1: Neighborhood in Euclidean Space
        
        #                          [RSI] 
        #                            |                    ..:::  
        #                            |                  ......
        #                            |               :••••••. 
        #                            |            :::••••••.  
        #                            |         .::.••••••.    
        #                            |       :..••••••..      
        #                            .....::••••••:..         
        #                            •••••.•••••••:.            
        #                            .•••••••••::.              
        #                            ••••.••••..                
        #   |---------------.:•••••••••••••••••.---------------[ADX]          
        #   0                        •
        #                            •
        #                            .
        #                            |
        #                            |
        #                            |
        #                            |
        #                            |
        #                            |
        #                            |
        #                           _|_ 0
        # 
        #        Figure 2: Neighborhood in Lorentzian Space
        #   
        #  Note this 'Lorentzian' metric (geodesic curve, not the neighborhood above) is calculated as:
        #  log1p(x)+log1p(y)=d not sqrt(log1p(x)^2+log1p(y)^2)=d
        #  
        #  NOTE:
        #  By implementing this metric, we are actually encouraging to find history points with "nearest neighbor"(similar situations):
        #   1. no dominant features(indicator) (e.g. ind1 >> max(ind2, ind3, ...))
        #   2. every features are somewhat present, and are of a particular order(e.g. ind3 > ind1 > ind2)
        #   3. it also eliminates some of the farthest matches to get the majority match
        #  Observations:
        #  (1) In Lorentzian Space, the shortest distance between two points is not 
        #      necessarily a straight line, but rather, a geodesic curve.
        #  (2) The warping effect of Lorentzian distance reduces the overall influence  
        #      of outliers and noise.
        #  (3) Lorentzian Distance becomes increasingly different from Euclidean Distance 
        #      as the number of nearest neighbors used for comparison increases.
        
        if len(self.f1_array) <= i:
            return float('inf')
            
        # log1p formula for each feature
        distance = (
            np.log1p(abs(self.f1_array[-1] - list(self.f1_array)[i])) +
            np.log1p(abs(self.f2_array[-1] - list(self.f2_array)[i])) +
            np.log1p(abs(self.f3_array[-1] - list(self.f3_array)[i])) +
            np.log1p(abs(self.f4_array[-1] - list(self.f4_array)[i])) +
            np.log1p(abs(self.f5_array[-1] - list(self.f5_array)[i]))
        )
        
        return float(distance)

    def _select_diverse_patterns(self, size: int) -> None:
        """Maintain chronologically and featurely diverse patterns"""
        patterns = []

        # Sample every 4 bars for chronological spacing
        for i in range(0, size, 4):
            d = self._get_lorentzian_distance(i)
            # Accept patterns that exceed our diversity threshold
            if d > self.last_distance:  # Changed from >= to > to avoid duplicates
                next_idx = min(i+4, len(self.closes)-1)
                label = 1 if self.closes[next_idx] > self.closes[i] else -1
                patterns.append((d, label))

                if len(self.distances) > self.neighbors_count:
                    # Use 75th percentile as upper bound for diverse feature space sampling
                    sorted_dists = sorted(self.distances)
                    percentile_idx = round(self.neighbors_count * 0.75)
                    self.last_distance = sorted_dists[percentile_idx]
                    self.distances.popleft()
                    self.predictions.popleft()

        # Add new patterns to our collection
        for d, label in patterns:
            self.distances.append(d)
            self.predictions.append(label)

    def update(self, high: float, low: float, close: float, volume: float) -> Tuple[bool, bool]:
        """Process new bar"""
        # Update price history
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        self.volumes.append(volume)
        
        if len(self.closes) < 20:
            return False, False

        # Calculate features
        rsi14 = self._calculate_rsi(14, 1)
        wt = self._calculate_wt(10, 11)
        cci = self._calculate_cci(20)
        adx = self._calculate_adx(20)
        rsi9 = self._calculate_rsi(9, 1)
        
        # Update feature arrays
        self.f1_array.append(rsi14)
        self.f2_array.append(wt)
        self.f3_array.append(cci)
        self.f4_array.append(adx)
        self.f5_array.append(rsi9)
        
        # Calculate kernel estimates (expected value of close)
        yhat1, yhat2 = self._calculate_kernel_estimates()
        self.yhat1.append(yhat1) # use student-t kernel for trend detection
        self.yhat2.append(yhat2) # use Gaussian kernel for smoothing
        self.kernel_estimates.append(yhat1)
        
        # =========================
        # ====  Core ML Logic  ====
        # =========================
        # 
        # Approximate Nearest Neighbors Search with Lorentzian Distance:
        # A novel variation of the Nearest Neighbors (NN) search algorithm that ensures a chronologically uniform distribution of neighbors.
        # 
        # In a traditional KNN-based approach, we would iterate through the entire dataset and calculate the distance between the current bar 
        # and every other bar in the dataset and then sort the distances in ascending order. We would then take the first k bars and use their 
        # labels to determine the label of the current bar. 
        # 
        # There are several problems with this traditional KNN approach in the context of real-time calculations involving time series data:
        # - It is computationally expensive to iterate through the entire dataset and calculate the distance between every historical bar and
        #   the current bar.
        # - Market time series data is often non-stationary, meaning that the statistical properties of the data change slightly over time.
        # - It is possible that the nearest neighbors are not the most informative ones, and the KNN algorithm may return poor results if the
        #   nearest neighbors are not representative of the majority of the data.
        # 
        # Previously, attempts were made to address some of these issues in KNN implementations by:
        # - Using a modified KNN algorithm based on consecutive furthest neighbors to find a set of approximate "nearest" neighbors.
        # Problems:
        # - The possibility of a sudden "max" value throwing off the estimation
        # - The possibility of selecting a set of approximate neighbors that are not representative of the majority of the data by oversampling 
        #   values that are not chronologically distinct enough from one another
        # - The possibility of selecting too many "far" neighbors, which may result in a poor estimation of price action
        # 
        # To address these issues, a novel Approximate Nearest Neighbors (ANN) algorithm is used in this indicator.
        # 
        # In the below ANN algorithm:
        # 1. The algorithm iterates through the dataset in chronological order, using the modulo operator to only perform calculations every 4 bars.
        #    This serves the dual purpose of reducing the computational overhead of the algorithm and ensuring a minimum chronological spacing 
        #    between the neighbors of at least 4 bars.
        # 2. A list of the k-similar neighbors is simultaneously maintained in both a predictions array and corresponding distances array.
        # 3. When the size of the predictions array exceeds the desired number of nearest neighbors specified in settings.neighborsCount, 
        #    the algorithm removes the first neighbor from the predictions array and the corresponding distance array.
        # 4. The lastDistance variable is overriden to be a distance in the lower 25% of the array. This step helps to boost overall accuracy 
        #    by ensuring subsequent newly added distance values increase at a slower rate.
        # 5. Lorentzian distance is used as a distance metric in order to minimize the effect of outliers and take into account the warping of 
        #    "price-time" due to proximity to significant economic events.


        size = min(self.lookback-1, len(self.closes)-1)
        self._select_diverse_patterns(size)

        # Generate signal
        if not self._check_filters() or len(self.predictions) < self.neighbors_count:
            return False, False
            
        prediction_sum = sum(self.predictions)
        
        # Kernel trend detection
        is_bullish_kernel = len(self.kernel_estimates) > 1 and self.kernel_estimates[-1] > self.kernel_estimates[-2]
        is_bearish_kernel = len(self.kernel_estimates) > 1 and self.kernel_estimates[-1] < self.kernel_estimates[-2]
        
        # Signal logic
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
        
        # Exit conditions
        if self.use_kernel_smoothing:
            if ((self.current_signal == 1 and is_bearish_smooth) or 
                (self.current_signal == -1 and is_bullish_smooth)):
                self.current_signal = 0
                self.bars_held = 0
        else:
            if self.bars_held >= 4:  # fixed 4-bar exit
                self.current_signal = 0
                self.bars_held = 0
        
        # Early signal flip detection
        if len(self.kernel_estimates) >= 3:
            prev_signal = 1 if self.kernel_estimates[-3] < self.kernel_estimates[-2] else -1
            curr_signal = 1 if self.kernel_estimates[-2] < self.kernel_estimates[-1] else -1
            
            if prev_signal != curr_signal and self.bars_held < 4:
                self.current_signal = 0
                self.bars_held = 0

        # MA trend conditions
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

        # Signal change detection
        is_different_signal = self.signal != prev_signal

        # Buy/Sell conditions
        is_buy_signal = (self.signal == 1 and is_ema_uptrend and is_sma_uptrend)
        is_sell_signal = (self.signal == -1 and is_ema_downtrend and is_sma_downtrend)
        
        # Entry signals
        is_new_buy_signal = is_buy_signal and is_different_signal
        is_new_sell_signal = is_sell_signal and is_different_signal
        
        # Kernel trend conditions
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