import os
import numpy as np
import pandas as pd
from typing import List

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
