import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Optional
from config.cfg_stk import cfg_stk

# ts_momentum_label        : TimeSeries trend/breakout label
# ts_mean_reversion_label  : TimeSeries mean-reversion label
# cs_label                 : CrossSectional long-short label

NUM_LABELS = 1

class ts_label:
    """
    TIMESERIES TREND/BREAKOUT Label
    
    this label should be "Localized Sharpe/Calmar Ratio"
    high-localized sharpe = high PnL opportunities:
        1. price consolidates, test support multiple times, we go long at support level
        2. price rise quickly, very high momentum, we go long to do mid-price chasing, set SL-TP narrowly(hence being safe)
        
    how "localized" should we be?
        1. too short, your ML model would go long every time it sees a giant positive bar, which would not work
           (either too little info to predict the giant bar, or not much profit left after it)
        2. if not too short? how long(x-direction)? and how to define the returns/volatility(drawdown)(y-direction)?
            1. it would require something like the famous "Triple-Barrier-Method" in ML4F textbook
            2. because trend strategies usually apply for relatively narrow SL-TP(contrary to mean-reversion that has wide SL-TP),
               by using SL-TP, we can determine the future return direction, return magnitude(when SL-TP is triggered), thus the drawdown magnitude
        3. we use adaptive super-trend as SL-TP metric, this is a relative stable method insensitive to hyperparameter tuning
        
    also this engineered label should be continuous etc. thus ICIR and other factor analysis tools could work nicely, and model can learn relatively easily
    
    remember the last 3 switches (a, b, c) and calculate returns/drawdowns
    based on profitability of initial direction:
    - For point A between a and b:
        1. going along a's direction, possibly ends with profit at price_B
        2. going against a's direction, possibly ends with profit at price_C
    
    
    TIMESERIES MEAN-REVERSION Label

    Background: in efficient market, ATR is usually bound by fees (because of mean-reversal strategies like grid trading)
        It is by nature the counterpart for trend strategies, terminating premature breakouts, leaving fake-outs for breakout traders
        If the price can't even effectively break ATR, then it is not worth trading as trend/breakouts
    Consider upper/lower band around (High+Low)/2 by +-EMA_rolling(ATR, n)
    for point a in ATR range:
        if the next crossover b happens at upper ATR band, and denote the 2nd crossover back to ATR band as point c
            then the point is a presumable long trade
            low = min(ohlc, a->b)
            high = max(ohlc, b->c)
            label = signed local sharpe = + abs(delta(high)/delta(low))
    for point a out of ATR range:
        if the next crossover b happens at upper ATR band, and denote the 2nd crossover back to ATR band as point c
            then the point is a presumable long trade
            low = min(ohlc, a->b)
            high = max(ohlc, b->c)
            label = signed local sharpe = + abs(delta(high)/delta(low))

    """
    
    def __init__(self, shared_tensor:torch.Tensor, tensor_pointer:Dict[str, Callable[[], int]]):
        self.shared_tensor = shared_tensor
        self.timestamp_idx = tensor_pointer['x'] # keep being mutable
        self.column_idx = tensor_pointer['y']()
        self.code_idx = tensor_pointer['z']()
        
        self.FEE = cfg_stk.FEE
        self.RET_thd = 10*self.FEE # punish low absolute returns
        self.A_L = (1-self.FEE/2) # fee adjusted long ratio
        self.A_S = (1+self.FEE/2) # fee adjusted short ratio
        
        # Switch history (prices and directions)
        self.switch_prices:list[float] = []  # Last 3 switch prices
        self.switch_atrs:list[float] = []  # Last 3 switch atrs
        self.switch_directions:list[bool] = []  # Last 3 switch directions (True for long)
        self.switch_segment_prices:list[list[float]] = []  # Last 3 switch segment prices
        # if cfg_cpt.dump_ind:
        #     self.switch_segment_timestamps:list[list[float]] = []  # Last 3 switch segment timestamps
        
        self.is_long_trend = None
        
        # Output storage
        # self.labels1: list[float] = []
        # self.labels2: list[float] = []
        # if cfg_cpt.dump_ind:
        #     self.timestamps: list[float] = []

    def update(self, ts: float, close: float, atr: float, long_switch: bool, short_switch: bool) -> None:
        """Update state and return current label."""
        
        def _calculate_metrics(price_A: float, atr_A: float, prices_A_to_x: list[float], price_x: float, is_long: bool) -> float:
            """Calculate return and drawdown for a given price point."""
            FACTOR = 8 # punish severely if drawdown goes beyond atr (for label to be a continuous value)
            
            if is_long:
                ret = (price_x*self.A_L - price_A*self.A_S) / price_A # positive = profit
                if ret < 0:
                    ret = 0
                low = min(prices_A_to_x)
                drawdown_abs = price_A - low # positive
            else:
                ret = (price_x*self.A_S - price_A*self.A_L) / price_A # negative = profit
                if ret > 0:
                    ret = 0
                high = max(prices_A_to_x)
                drawdown_abs = high - price_A # positive
                
            if drawdown_abs < atr_A:
                drawdown = drawdown_abs / price_A
            else:
                # protect ** against overflow
                drawdown = FACTOR**(min(drawdown_abs/atr_A - 1, 5)) * drawdown_abs / price_A
            
            calmar = ret / (drawdown + 1) # drawdown_adjusted_return
            calmar = np.tanh(calmar/self.RET_thd) # magnitude_adjusted_return
            return calmar
        
        if long_switch or short_switch:
            # New switch detected
            new_direction = long_switch
            
            # Update switch history
            self.switch_prices.append(close)
            self.switch_atrs.append(atr)
            self.switch_directions.append(new_direction)
            self.switch_segment_prices.append([close])
            # if cfg_cpt.dump_ind:
            #     self.switch_segment_timestamps.append([ts])
            
            # Keep only last 3 switches
            if len(self.switch_prices) > 3:
                self.switch_prices.pop(0)
                self.switch_atrs.pop(0)
                self.switch_directions.pop(0)
                self.switch_segment_prices.pop(0)
                # if cfg_cpt.dump_ind:
                #     self.switch_segment_timestamps.pop(0)
            
            # Calculate labels if we have enough history (3 switches)
            if len(self.switch_prices) == 3:
                # Label points between first and second switch
                price_a, price_b, price_c = self.switch_prices
                atr_a, atr_b, atr_c = self.switch_atrs
                direction_a = self.switch_directions[0]
                segment_prices_a = self.switch_segment_prices[0]
                segment_prices_b = self.switch_segment_prices[1]
                # if cfg_cpt.dump_ind:
                #     segment_timestamps_a = self.switch_segment_timestamps[0]
                
                # Process all points between a and b
                
                num_seg_a = len(segment_prices_a)
                labels = [[0.0 for i in range(num_seg_a)] for i in range(NUM_LABELS)]
                for i in range(num_seg_a):
                    price_A = segment_prices_a[i]
                    prices_A_to_b = segment_prices_a[i:]
                    prices_A_to_c = segment_prices_a[i:] + segment_prices_b + [price_c]
                    
                    # going along a's direction(trend following), possibly ends with profit at price_B
                    # going against a's direction(mean-reversion(with stop)), possibly ends with profit at price_C
                    # try both along and against, see which works (at most one would work, and the other hit stoploss)
                    calmar1 = _calculate_metrics(price_A, atr_a, prices_A_to_b, price_b, direction_a)
                    calmar2 = _calculate_metrics(price_A, atr_a, prices_A_to_c, price_c, not direction_a)
                    
                    labels[0][i] = calmar1 + calmar2
                    # if cfg_cpt.dump_ind:
                    #     self.timestamps.append(segment_timestamps_a[i])

                # update final tensor:
                
                for i in range(NUM_LABELS):
                    # print(end_idx, num_seg_a, np.shape(labels), self.shared_tensor.shape)
                    end_idx = self.timestamp_idx()
                    start_idx = end_idx-num_seg_a
                    self.shared_tensor[start_idx:end_idx, self.column_idx+i, self.code_idx,] \
                    = torch.tensor(labels[i], dtype=torch.float16)
                
            self.is_long_trend = new_direction
            
        else:
            if len(self.switch_prices) > 0:
                # Collect price in current segment
                self.switch_segment_prices[-1].append(close)
                # if cfg_cpt.dump_ind:
                #     self.switch_segment_timestamps[-1].append(ts)

def dummy_label(df, feature_names, label_names):
    """
    naive logreturn
    """
    n_labels = len(label_names)
    
    # Find the log_return column name from feature names
    log_return_col = next(col for col in feature_names if 'log_returns' in col and '_1' in col)
    y = df[log_return_col] * 10000  # Convert to pips
    # Apply winsorization (Median Absolute Deviation)
    median = np.median(y)
    mad = np.median(np.abs(y - median))
    lower = median - 3 * mad / 0.6745  # scaled MAD
    upper = median + 3 * mad / 0.6745
    y_winsorized = np.clip(y, lower, upper)
    # Then apply Yeo-Johnson
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    y_transformed = pt.fit_transform(y_winsorized.values.reshape(-1, 1))
    
    # Calculate cumulative returns properly
    for i in range(n_labels):
        horizon = i + 1
        y_cumulated = pd.Series(y_transformed.flatten()).rolling(window=horizon, min_periods=horizon).sum().shift(-horizon)
        df[f'label_{horizon}'] = y_cumulated.astype('float32')
        # np.where(base > 0, 1, np.where(base <= 0, 0, 0))
    # Remove rows with NaN labels at the end
    df = df.iloc[:-n_labels].copy()
    return df
    