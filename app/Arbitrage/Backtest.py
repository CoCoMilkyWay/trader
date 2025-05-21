import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

class Backtest:
    def __init__(self):
        self.dir = os.path.dirname(os.path.abspath(__file__))
        pass
    
    def backtest(self, history:pd.DataFrame, symbols:List[str]):
        W = 4*60*7*5  # n week of market hours
        n = 30
        history = self.filter_cn(history)
        # for idx, symbol in enumerate(tqdm(symbols)):
        #     df[f'{symbol}_pct_90_rolling'] = df[f'{symbol}_premium'].rolling(window=W, min_periods=1).quantile(0.90)
        #     df[f'{symbol}_pct_10_rolling'] = df[f'{symbol}_premium'].rolling(window=W, min_periods=1).quantile(0.10)
        
        symbol = '513300'
        df = history.loc[:, [f'close', f'{symbol}_close', f'{symbol}_premium']].copy()
        df['premium_ma'] = df[f'{symbol}_premium'].rolling(window=W, min_periods=1).mean()
        df['low_n'] = df[f'{symbol}_premium'].rolling(window=n, min_periods=1).min()
        df['high_n'] = df[f'{symbol}_premium'].rolling(window=n, min_periods=1).max()
        df['atr_n'] = (df['high_n'] - df['low_n']).rolling(window=n, min_periods=1).mean()
        df[f'pct_90_rolling'] = df[f'{symbol}_premium'].rolling(window=W, min_periods=1).quantile(0.90)
        df[f'pct_10_rolling'] = df[f'{symbol}_premium'].rolling(window=W, min_periods=1).quantile(0.10)
        
        RMB = 1000000
        USD = 1000000
        
        df['mid'] = float('nan')
        df['pos_open'] = float('nan')
        df['pos_close'] = float('nan')
        df['RMB'] = float('nan')
        df['USD'] = float('nan')
        first_index = df.index[0]
        df.loc[first_index, 'RMB'] = RMB
        df.loc[first_index, 'USD'] = USD
        
        hold = False
        entry_price_equity = 0  # Price when position is entered
        entry_price_futures = 0  # Price when position is entered
        # FEEs
        # Futures:
        comm_fut = 0.25  # entering/exiting per contract
        tax_fut = 5  # per contract
        # Equities:
        comm_eqt_rate = 0.0001
        last_levels = [0]
        
        for index, row in df.iterrows():
            time = index
            pct10 = row['pct_10_rolling']
            pct90 = row['pct_90_rolling']
            premium_ma = row['premium_ma']
            atr_n = row['atr_n']
            mid = (row['high_n'] + row['low_n'])/2
            premium = row[f'{symbol}_premium']
            equity_price = row[f'{symbol}_close']
            future_price = row['close']
            open_position = False
            close_position = False
            
            try:
                level = round((premium - mid) / (atr_n/8))
            except:
                level = 0
            if level != last_levels[-1]:
                last_levels.append(level)

            open = level < -2 and (last_levels[-1] > last_levels[-2])
            close = (level > 2) or (int(str(index)[-4:])>1455)

            # open = level < -2 and (last_levels[-1] > last_levels[-2] > last_levels[-3])
            # close = (level > 2 and (last_levels[-1] < last_levels[-2])) or (int(str(index)[-4:])>1455)
            
            # open = (premium < pct10) and (premium < 2)
            # close = (premium > pct90)
            # Open positions when abs(premium_ma) < 5 and premium is low
            if open and not hold:
                open_position = True
                hold = True
                # Calculate the maximum possible position size based on available funds
                position_amount = (RMB+USD)/2  # min(RMB, USD)  # Ensure the position sizes are equal
                if position_amount < 100000:
                    assert False, f"RMB: {RMB}, USD: {USD}"
                # Open the combined position (long equity + short futures)
                equity_price += 0.001 # taker order
                long_position = int(RMB // equity_price)  # Amount of both equity and futures (same size for both)
                short_position = int(USD // future_price)  # Amount of both equity and futures (same size for both)
                entry_price_equity = equity_price  # Record the price when the position was opened
                entry_price_futures = future_price  # Record the price when the position was opened
                open_value_equity = long_position*equity_price
                open_value_futures = short_position*future_price
                RMB -= open_value_equity * (1+comm_eqt_rate)  # Deduct the RMB spent on the long position
                USD += open_value_futures - (comm_fut+tax_fut) * short_position  # Increase the USD spent on the short position
                print(f"Opening at {time}: equity:{equity_price:2.4f}, futures:{future_price:5.2f} (premium:{premium:+2.2f}, size:{position_amount:8.2f})")
            # Close both positions when premium is high
            if close and hold:
                close_position = True
                hold = False
                # Close the combined position (long equity + short futures)
                sell_equity_price = equity_price - 0.001
                cover_futures_price = future_price
                # Profit from the long equity position
                close_value_equity = sell_equity_price * long_position
                profit_equity = close_value_equity - open_value_equity
                RMB += close_value_equity * (1-comm_eqt_rate)  # Add the profit to RMB
                # Profit from the short futures position
                close_value_futures = cover_futures_price * short_position
                profit_futures = open_value_futures - close_value_futures
                USD -= close_value_futures - comm_fut * short_position  # Add the profit to USD
                total_profit = profit_equity + profit_futures
                long_position = 0
                short_position = 0
                df.at[index, 'RMB'] = RMB
                df.at[index, 'USD'] = USD
                print(f"Closing at {time}: equity:{sell_equity_price:2.4f}, futures:{cover_futures_price:5.2f}, premium:{premium:+2.2f}, RMB:{RMB:8.2f}, USD:{USD:8.2f}, profit:{profit_equity:6.2f}/{profit_futures:6.2f}")
            
            if open_position:
                df.at[index, 'pos_open'] = premium
            if close_position:
                df.at[index, 'pos_close'] = premium
            df.at[index, 'mid'] = mid
                
        df['RMB'] = df['RMB'].ffill()
        df['USD'] = df['USD'].ffill()
        
        plot_index = pd.to_datetime(df.index.astype(str), format='%Y%m%d%H%M').strftime('%H%M-%Y%m%d')
        
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        # fig = go.Figure()
        
        # fig.add_trace(go.Scatter(y=df[f'pct_90_rolling'],         mode='lines', line=dict(width=2)                , name='up'),     row=1, col=1)
        # fig.add_trace(go.Scatter(y=df[f'pct_10_rolling'],         mode='lines', line=dict(width=2)                , name='down'),   row=1, col=1)
        # fig.add_trace(go.Scatter(y=df[f'premium_ma'],             mode='lines', line=dict(width=2)                , name='mid'),    row=1, col=1)
        
        fig.add_trace(go.Scatter(y=df[f'mid']+df['atr_n']/2,         mode='lines', line=dict(width=2)                , name='up'),     row=1, col=1)
        fig.add_trace(go.Scatter(y=df[f'mid'],         mode='lines', line=dict(width=2)                , name='down'),   row=1, col=1)
        fig.add_trace(go.Scatter(y=df[f'mid']-df['atr_n']/2,             mode='lines', line=dict(width=2)                , name='mid'),    row=1, col=1)

        fig.add_trace(go.Scatter(y=df[f'{symbol}_premium'],       mode='lines', line=dict(width=1), opacity=0.5   , name='premium'),row=1, col=1)
        fig.add_trace(go.Scatter(y=df['pos_open'],    mode='markers', marker=dict(size=10)             , name='pos_open')      ,    row=1, col=1)
        fig.add_trace(go.Scatter(y=df['pos_close'],   mode='markers', marker=dict(size=10)             , name='pos_close')     ,    row=1, col=1)

        fig.add_trace(go.Scatter(y=df['RMB'], mode='lines', name='RMB'), row=2, col=1)
        fig.add_trace(go.Scatter(y=df['USD'], mode='lines', name='USD'), row=2, col=1)
        fig.add_trace(go.Scatter(y=df['RMB']+df['USD'], mode='lines', name='total'), row=2, col=1)
        
        # print(df[df['pos_open'].notna()])
        # print(df[df['pos_close'].notna()])
        
        fig.update_layout(
            title='trading backtest',
            xaxis_title='timestamp(min)',
            yaxis_title='premium(percent)',
            template='plotly_white',
            showlegend=True,
            height=1400,  # This affects the absolute pixel height of the whole figure
        )
        # fig.update_xaxes(
        #     showgrid=True,
        #     tickmode='linear',     # Force evenly spaced ticks
        #     dtick=60*4,               # Set tick interval (e.g., every 1 unit)
        #     gridcolor='lightgray',
        #     gridwidth=0.5
        # )
        # fig.to_html(os.path.join(self.dir, "fig.html"))
        fig.show()
        
        return
    
    def filter_cn(self, df:pd.DataFrame):
        hour_min = df.index % 10000  # Get HHMM part
        # Filter for Shanghai A-share session (09:30–11:29 and 13:00–14:59)
        mask = (
            ((hour_min >= 930) & (hour_min <= 1129)) |
            ((hour_min >= 1300) & (hour_min <= 1459))
        )
        df = df[mask] # only during this time arbitrage is possible
        return df