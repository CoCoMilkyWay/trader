import os
import numpy as np
import pandas as pd
from typing import List

dir = os.path.dirname(__file__)


def main():
    data_path = os.path.join(dir, "data/bars.parquet")

    time_bar_dtype = {
        'time': 'int64',
        'open': 'float32',
        'high': 'float32',
        'low': 'float32',
        'close': 'float32',
        'volume': 'float32',
    }

    run_bar_dtype = {
        'time': 'int64',
        'timedelta': 'int32',
        'open': 'float32',
        'high': 'float32',
        'low': 'float32',
        'close': 'float32',
        'vwap': 'float32',
        'threshold': 'float32',
        'label_continuous': 'float32',
        'label_discrete': 'int32',
        'label_uniqueness': 'float32',
        'umap_x': 'float32',
        'umap_y': 'float32',
        'umap_z': 'float32',
    }

    input_dtype = np.dtype([(k, v) for k, v in time_bar_dtype.items()])
    output_dtype = np.dtype([(k, v) for k, v in run_bar_dtype.items()])

    time_bar = pd.read_parquet(data_path).reset_index(drop=True)
    input_array = time_bar.to_records(index=False).astype(input_dtype)
    input_bytes = input_array.tobytes()

    print(time_bar)
    print("Num bars:", input_array.shape[0])
    print("Bytes per record:", input_array.dtype.itemsize)

    from cpp import Pipeline  # type: ignore
    try:
        output_bytes = Pipeline.process_bars(input_bytes, input_array.shape[0])
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Caught exception:", str(e))

    vrun_bar = pd.DataFrame(np.frombuffer(output_bytes, dtype=output_dtype))
    print(vrun_bar)
    vrun_bar['return'] = np.log1p(vrun_bar['close'].pct_change().fillna(0))

    from collections import deque
    num_strength = 7
    num_bars = 2
    idx_weighted = num_strength + num_bars*2
    idx_count = num_strength + num_bars*2 + 1
    buffer_s = deque(maxlen=num_bars+1)
    buffer_t = deque(maxlen=num_bars+1)
    # rows: num_strength**num_bars
    # columns: X0, X1, ..., X_num_bars-1, S0, S1, ... ,S_num_strength-1, S_weighted, S_count
    stats = np.zeros((num_strength**(num_bars+1)-1, num_bars+num_strength+2), dtype=float)
    indices = np.indices([num_strength]*num_bars).reshape(num_bars, -1).T  # (N, num_bars)
    stats[:, :num_bars] = indices

    for row in vrun_bar.itertuples(index=False):
        timedelta = int(row.timedelta)  # type: ignore
        open = float(row.open)  # type: ignore
        high = float(row.high)  # type: ignore
        low = float(row.low)  # type: ignore
        close = float(row.close)  # type: ignore
        # Calculate true range and section sizes
        true_range = high - low
        section_size = true_range / 4

        # Determine body position
        body_size = abs(close-open)

        # Determine if candle is bullish or bearish
        is_bullish = close > open

        # Calculate strength based on body position
        if is_bullish:
            if 3*section_size <= body_size:
                strength = 0
            elif 2*section_size <= body_size:
                strength = 1
            elif 1*section_size <= body_size:
                strength = 2
            else:
                strength = 3
        else:  # bearish
            if 3*section_size <= body_size:
                strength = 6
            elif 2*section_size <= body_size:
                strength = 5
            elif 1*section_size <= body_size:
                strength = 4
            else:
                strength = 3

        buffer_s.append(strength)
        buffer_t.append(timedelta)
        if len(buffer_s) < num_bars + 1:
            continue

        index = 1
        for i in range(num_bars):
            index += buffer_s[i]*(num_strength-1)**(num_bars-i-1)

        stats[index, num_bars*2+strength] += 1

    start_S = num_bars * 2
    end_S = num_bars * 2 + num_strength
    stats[:, idx_count] = stats[:, start_S:end_S].sum(axis=1)
    stats = stats[stats[:, idx_count] > 50]
    stats[:, idx_weighted] = (stats[:, start_S:end_S] * range(num_strength)).sum(axis=1)/stats[:, idx_count]
    round_unit = 0.1
    round_coef = int(1/round_unit)
    stats[:, idx_weighted] = np.round(stats[:, idx_weighted] * round_coef) / round_coef

    stats = pd.DataFrame(stats)
    print(stats.describe())

    X = stats.iloc[:, :start_S]
    y = stats.iloc[:, idx_weighted]
    plot_umap_3D(X, y)
    # plot_3D(stats, [0, 1, 2], idx_weighted)

    limit = int(60/5*24*5*4*1)
    time_bar = time_bar[-limit*5:]
    vrun_bar = vrun_bar[-limit:]

    # plot_line(vrun_bar, 'time', 'close', 'label_discrete')
    # plot_density(vrun_bar, 'label_continuous')
    # plot_density(vrun_bar, 'return')
    # plot_3D(vrun_bar, ['umap_x', 'umap_y', 'umap_z'], 'label_discrete')


def plot_bars(time_bar: pd.DataFrame, vrun_bar: pd.DataFrame):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_bar['time'], y=time_bar['close'], mode='lines', name='Time bar'))
    fig.add_trace(go.Scatter(x=vrun_bar['time'], y=vrun_bar['high'], line=dict(width=0), mode='lines', showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=vrun_bar['time'], y=vrun_bar['low'], fill='tonexty', fillcolor='rgba(200, 200, 200, 0.5)', line=dict(width=0), mode='lines', name='High-Low Range', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=vrun_bar['time'], y=vrun_bar['close'], mode='lines+markers', marker=dict(color=vrun_bar['label_discrete'], colorscale='RdBu', showscale=True, size=10), name="data",))
    fig.add_trace(go.Scatter(x=vrun_bar['time'], y=vrun_bar['close']-3000, mode='lines+markers', marker=dict(color=vrun_bar['label_uniqueness'], colorscale='Plasma'), name="data"))
    fig.update_layout(xaxis=dict(type='category'))
    fig.show()


def plot_line(df: pd.DataFrame, support: str, field: str, color: str):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[support], y=df[field], mode='lines+markers', marker=dict(color=df[color], colorscale='RdBu', showscale=True, size=10), name="data",))
    fig.update_layout(xaxis=dict(type='category'))
    fig.show()


def plot_3D(df: pd.DataFrame, fields: List[str | int], color: str | int):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig = go.Figure(data=[go.Scatter3d(
        x=df[fields[0]],
        y=df[fields[1]],
        z=df[fields[2]],
        mode='markers',  # or 'lines+markers' or 'lines'
        marker=dict(
            size=5,
            color=df[color],  # Optional: use z as color
            colorscale='RdBu',
            opacity=0.8
        )
    )])
    fig.show()


def plot_density(df: pd.DataFrame, field: str | int):
    from scipy.stats import gaussian_kde
    import plotly.graph_objects as go
    x = df[field]
    kde = gaussian_kde(x)
    x_vals = np.linspace(x.min(), x.max(), 1000)
    y_vals = kde(x_vals)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='KDE'))
    fig.update_layout(title='KDE of label', xaxis_title='Label', yaxis_title='Density')
    fig.show()


def plot_umap_3D(X: pd.DataFrame, y: pd.Series):
    import umap
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np

    n_neighbors = min(15, X.shape[0] - 1)

    algo_umap = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=0.1)
    X_3d = np.array(algo_umap.fit_transform(X))

    classes = np.sort(np.unique(y))[::-1]  # sort from large(red) to small(blue)
    print(f'Classes: {classes}')

    # RdBu continuous colorscale
    colorscale = px.colors.diverging.RdBu
    n_colors = len(colorscale)

    fig = go.Figure()

    for i, cls in enumerate(classes):
        idx = y == cls
        # Map class index to a color in RdBu colorscale
        color_idx = int(i / (len(classes) - 1) * (n_colors - 1)) if len(classes) > 1 else 0
        color = colorscale[color_idx]

        fig.add_trace(go.Scatter3d(
            x=X_3d[idx, 0], y=X_3d[idx, 1], z=X_3d[idx, 2],
            mode='markers',
            name=f'Class {cls}',
            marker=dict(
                size=4,
                color=color,
                opacity=1.0,
            )
        ))

    fig.show()


if __name__ == "__main__":
    main()
