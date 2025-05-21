import pandas as pd
import numpy as np
import plotly.graph_objects as go

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), 'volume_run_bar.parquet'))
# df = df[-int(60/5*24*5*4*2):]
df = df.set_index('time')

# Normalize 'b' for color mapping
norm_b = (df['b'] - df['b'].min()) / (df['b'].max() - df['b'].min())

# Create a 2-row heatmap background using b
z = np.tile(norm_b.values, (2, 1))  # shape (2, N)

# Create figure
fig = go.Figure()

# Add background as heatmap
fig.add_trace(go.Heatmap(
    z=z,
    x=df.index,
    y=[df['a'].min(), df['a'].max()],
    colorscale='Viridis',
    showscale=False,
    zmin=0,
    zmax=1,
    opacity=0.6
))

# Overlay the line for column 'a'
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['a'],
    mode='lines',
    line=dict(color='black'),
    name='a'
))

fig.update_layout(
    title="Plot of 'a' with 'b' as vertical background color",
    xaxis_title="Time",
    yaxis_title="a",
    yaxis=dict(range=[df['a'].min(), df['a'].max()]),
    xaxis=dict(showgrid=False),
    template='plotly_white'
)

fig.show()

