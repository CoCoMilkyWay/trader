import numpy as np
import pandas as pd

# ts_momentum_label        : TimeSeries trend/breakout label
# ts_mean_reversion_label  : TimeSeries mean-reversion label
# cs_label                 : CrossSectional long-short label

def ts_momentum_label(df:pd.DataFrame, feature_names, label_names):
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
    """

    return

def ts_mean_reversion_label(df:pd.DataFrame, feature_names, label_names):
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
    