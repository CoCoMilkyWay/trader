import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# period < 20 = stationary (ADF test)
# period > 10 = significant trend info (correlation)
PERIOD = 30
filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"data_and_label_{PERIOD}.parquet")

def load_data_and_label():
    # --- Configuration
    pd.set_option('display.max_rows', None)
    np.random.seed(42)  # For reproducibility

    df = pd.read_parquet(os.path.join(os.path.dirname(__file__), 'volume_run_bar.parquet'))
    df = df[-int(60/5*24*5*4*5):]
    df = df.set_index('time')
    series = df["close"]
    df['ref'] = np.log((series / series.iloc[0]).fillna(1))
    df['data'] = np.log((series / series.rolling(PERIOD).mean().shift(PERIOD)).fillna(1))

    from hmmlearn import hmm
    from sklearn.preprocessing import StandardScaler
    # | Data Type          | Works with HMM? | Recommended Action                      |
    # | ------------------ | --------------- | --------------------------------------- |
    # | Symmetric Gaussian | Yes             | Use standard Gaussian HMM               |
    # | Skewed data        | Maybe           | Transform or use non-Gaussian emissions |
    # | Heavy-tailed       | Maybe           | Try Student-t or GMM-HMM                |
    # | Discrete/ordinal   | Yes             | Use categorical/discrete HMM            |

    # 1. Load your data
    # periods = [3, 6, 12, 24] # longer trend than available periods, but cleaner label
    periods = [1, 2, 4, 6, 8]  # longer trend than available periods, but cleaner label
    for period in periods:
        df[f'pct_fwd_{period}'] = np.log1p(df["close"].pct_change(periods=-period).fillna(0))
        # df[f'pct_bak_{period}'] = np.log1p(df["close"].pct_change(periods=period).fillna(0))

    for i, period in enumerate(periods):
        df[f'pct_cum_fwd_{period}'] = sum(df[f'pct_fwd_{p}'] for p in periods[:(i+1)])
        # df[f'pct_cum_bak_{period}'] = sum(df[f'pct_bak_{p}'] for p in periods[:(i+1)])
        df[f'f_p_{period}'] = df[f'pct_cum_fwd_{period}'].apply(lambda x: x if x > 0 else 0)
        df[f'f_n_{period}'] = df[f'pct_cum_fwd_{period}'].apply(lambda x: x if x <= 0 else 0)
    # df['atr'] = df[f'high'] - df[f'low']
    # df['bar_ratio'] = np.where(df['atr'] != 0, (df['close'] - df['open']) / df['atr'], 0)
    X = df[[f'f_p_{p}' for p in periods]+[f'f_n_{p}' for p in periods]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_states = 3  # choose your number of regimes
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=30,
        verbose=True,    # print convergence info
        random_state=0,
    )
    model.fit(X_scaled)

    hidden_states = model.predict(X_scaled)
    df["label"] = hidden_states

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['ref'], mode='lines+markers', marker=dict(color=hidden_states,
                  colorscale='Spectral', showscale=True, colorbar=dict(title="HMM State"), size=6), name="data",))
    fig.add_trace(go.Scatter(y=df['data'], mode='lines+markers', marker=dict(color=hidden_states,
                  colorscale='Plasma', showscale=True, colorbar=dict(title="HMM State"), size=6), name="data",))
    fig.show()

    df = df[['data', 'label']]  # use data because it is stationary
    df.to_parquet(filepath)
    return df


def analyze_feature(df: pd.DataFrame):

    window_size = 4
    horizon = 1

    windows = []
    labels = {}
    window_id = 0

    df = df.sort_values('time').reset_index(drop=True)
    for i in range(len(df) - window_size - horizon):
        window = df.iloc[i:i + window_size].copy()
        future_return = df.iloc[i + window_size + horizon - 1]['return']
        window['id'] = window_id
        windows.append(window[['id', 'time', 'price', 'volume']])
        labels[window_id] = future_return
        window_id += 1

    df_windows = pd.concat(windows, ignore_index=True)
    label_series = pd.Series(labels)

    print(df_windows)
    print(label_series)

    # --- Feature extraction
    features = extract_features(
        df_windows,
        column_id="id",
        column_sort="time",
        default_fc_parameters=EfficientFCParameters(),
        disable_progressbar=True
    )

    impute(features)

    # --- Compute correlation with label
    results = {}
    for col in features.columns:
        # Use Spearman for robustness (nonlinear monotonic relationships)
        corr = pd.Series(features[col]).corr(label_series, method='spearman')
        results[col] = abs(corr)  # use absolute value to reflect strength

    # --- Sort by importance
    sorted_features = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print("\nTop statistically important features (Spearman correlation):")
    for name, score in sorted_features[:10]:
        print(f"{name}: {score:.4f}")


if __name__ == '__main__':
    if os.path.exists(filepath):
        df = pd.read_parquet(filepath)
    else:
        df = load_data_and_label()
    print(df.tail())
    analyze_feature(df)
