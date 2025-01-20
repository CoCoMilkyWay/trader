import numpy as np
import pandas as pd
from math import log1p
from config.cfg_cpt import cfg_cpt

# ts_momentum_label        : TimeSeries trend/breakout label
# ts_mean_reversion_label  : TimeSeries mean-reversion label
# cs_label                 : CrossSectional long-short label

class ts_momentum_label:
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
    - For points between a and b, denoted as A:
      * If profitable in a's direction: calculate using price_A to price_b(take profit)
      * If unprofitable in a's direction: use b's direction and calculate using price_A to price_c(take profit)
    """
    def __init__(self):
        self.FEE = cfg_cpt.FEE
        self.A_L = (1-self.FEE/2) # fee adjusted long ratio
        self.A_S = (1+self.FEE/2) # fee adjusted short ratio
        
        # Switch history (prices and directions)
        self.switch_prices = []  # Last 3 switch prices
        self.switch_directions = []  # Last 3 switch directions (True for long)
        
        # Segment tracking
        self.prev_segment_prices = []  # Prices between a and b
        self.curr_segment_prices = []  # Prices between b and c
        
        if cfg_cpt.dump_ind:
            self.prev_segment_timestamps = []
            self.curr_segment_timestamps = []
        
        self.is_long_trend = None
        
        # Output storage
        self.labels: list[float] = []
        if cfg_cpt.dump_ind:
            self.timestamps: list[float] = []
            self.closes: list[float] = []

    def _calculate_metrics(self, price_A: float, prices_after_A: list[float], 
                         close_price: float, is_long: bool) -> tuple[float, float]:
        """Calculate return and drawdown for a given price point."""
        if is_long:
            ret = (close_price*self.A_L - price_A*self.A_S) / price_A
            low = min(prices_after_A)
            drawdown = (price_A - low) / price_A
        else:
            ret = (close_price*self.A_S - price_A*self.A_L) / price_A
            high = max(prices_after_A)
            drawdown = (high - price_A) / price_A
        return ret, drawdown

    def update(self, ts: float, close: float, long_switch: bool, short_switch: bool) -> None:
        """Update state and return current label."""
        if long_switch or short_switch:
            # New switch detected
            new_direction = long_switch
            
            # Update switch history
            self.switch_prices.append(close)
            self.switch_directions.append(new_direction)
            
            # Shift segments
            self.prev_segment_prices = self.curr_segment_prices
            self.curr_segment_prices = [close]
            if cfg_cpt.dump_ind:
                self.prev_segment_timestamps = self.curr_segment_timestamps
                self.curr_segment_timestamps = [ts]
            
            # Keep only last 3 switches
            if len(self.switch_prices) > 3:
                self.switch_prices.pop(0)
                self.switch_directions.pop(0)
            
            # Calculate labels if we have enough history (3 switches)
            if len(self.switch_prices) == 3:
                # Label points between first and second switch
                price_a, price_b, price_c = self.switch_prices
                direction_a = self.switch_directions[0]
                
                # Process all points between a and b
                for i in range(len(self.prev_segment_prices)):
                    price_A = self.prev_segment_prices[i]
                    
                    # Check if initial direction was profitable
                    ret_initial, _ = self._calculate_metrics(
                        price_A, [price_b], price_b, direction_a
                    )
                    
                    if (direction_a and ret_initial > 0) or (not direction_a and ret_initial < 0):
                        # Profitable in initial direction - use price_b
                        ret, drawdown = self._calculate_metrics(
                            price_A, 
                            self.prev_segment_prices[i:], 
                            price_b, 
                            direction_a
                        )
                    else:
                        # Unprofitable in initial direction - use price_c and opposite direction
                        ret, drawdown = self._calculate_metrics(
                            price_A, 
                            self.prev_segment_prices[i:] + self.curr_segment_prices + [price_c], 
                            price_c, 
                            not direction_a
                        )
                    
                    # Calculate final label
                    epsilon = 1e-7
                    PNL = 3 # punish low pnl
                    RET = 10*self.FEE # punish low absolute returns
                    ratio = ret / (drawdown + epsilon)
                    label = np.tanh(ratio/PNL) * np.tanh(abs(ret)/RET)
                    
                    self.labels.append(label)
                    if cfg_cpt.dump_ind:
                        self.closes.append(self.prev_segment_prices[i])
                        self.timestamps.append(self.prev_segment_timestamps[i])
            
            self.is_long_trend = new_direction
            
        else:
            # Collect price in current segment
            self.curr_segment_prices.append(close)
            if cfg_cpt.dump_ind:
                self.curr_segment_timestamps.append(ts)
            
    def get_labels(self):
        if cfg_cpt.dump_ind:
            return self.timestamps, self.closes, self.labels
        else:
            return self.labels

class ts_mean_reversion_label:
    """
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
    
    def __init__(self, df:pd.DataFrame, feature_names:list[str], label_names:list[str]):
        pass
    
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
    