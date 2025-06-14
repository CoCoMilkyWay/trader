import os
import numpy as np
import pandas as pd

dir = os.path.dirname(__file__)


def main():
    data_path = os.path.join(dir, "data/bars.parquet")

    # Define dtype as a dict first — single source of truth
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
        'open': 'float32',
        'high': 'float32',
        'low': 'float32',
        'close': 'float32',
        'vwap': 'float32',
        'threshold': 'float32',
        'label_continuous': 'float32',
        'label_discrete': 'int32',
        'label_uniqueness': 'float32',
    }
    input_dtype = np.dtype([(k, v) for k, v in time_bar_dtype.items()])
    output_dtype = np.dtype([(k, v) for k, v in run_bar_dtype.items()])

    time_bar = pd.read_parquet(data_path).reset_index(drop=True)
    input_array = time_bar.to_records(index=False).astype(input_dtype)
    input_bytes = input_array.tobytes()

    print(input_array)
    print("Num bars:", input_array.shape[0])

    from cpp import Pipeline  # type: ignore
    try:
        output_bytes = Pipeline.process_bars(input_bytes, input_array.shape[0])
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Caught exception:", str(e))

    # Convert bytes to structured array

    vrun_bar = pd.DataFrame(np.frombuffer(output_bytes, dtype=output_dtype))
    print(vrun_bar)

    import plotly.graph_objects as go
    limit = int(60/5*24*5*4)
    time_bar = time_bar[:limit*5]
    vrun_bar = vrun_bar[:limit]

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=time_bar['time'], y=time_bar['close'], mode='lines', name='Time bar'))
    # fig.add_trace(go.Candlestick(x=vrun_bar['time'], open=vrun_bar['open'], high=vrun_bar['high'], low=vrun_bar['low'], close=vrun_bar['close'], name='Volume run bar'))
    # fig.update_layout(
    #     title='Sampling methods',
    #     xaxis_title='Date',
    #     yaxis_title='Price',
    #     xaxis_rangeslider_visible=False,
    #     xaxis=dict(type='category')
    # )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=vrun_bar.index,
        y=vrun_bar['high'],
        line=dict(width=0),
        mode='lines',
        showlegend=False,
        hoverinfo='skip'
    ))
    fig2.add_trace(go.Scatter(
        x=vrun_bar.index,
        y=vrun_bar['low'],
        fill='tonexty',
        fillcolor='rgba(200, 200, 200, 0.5)',
        line=dict(width=0),
        mode='lines',
        name='High-Low Range',
        hoverinfo='skip'
    ))
    fig2.add_trace(go.Scatter(
        x=vrun_bar.index,
        y=vrun_bar['close'],
        mode='lines+markers',
        marker=dict(color=vrun_bar['label_discrete'], colorscale='RdBu', showscale=True, size=10),
        name="data",
    ))
    fig2.add_trace(go.Scatter(y=vrun_bar['close']-3000, mode='lines+markers', marker=dict(color=vrun_bar['label_uniqueness'], colorscale='Plasma'), name="data"))
    fig2.show()

    from scipy.stats import gaussian_kde
    x = vrun_bar['label_continuous']
    kde = gaussian_kde(x)
    x_vals = np.linspace(x.min(), x.max(), 1000)
    y_vals = kde(x_vals)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='KDE'))
    fig3.update_layout(title='KDE of label', xaxis_title='Label', yaxis_title='Density')
    fig3.show()

if __name__ == "__main__":
    main()
