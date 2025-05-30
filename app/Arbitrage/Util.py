import os
import sys
import calendar
import numpy as np
import pandas as pd
from functools import lru_cache

from typing import Tuple, List, Dict
from datetime import date, datetime, timedelta

# Colors
RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"


def mkdir(filepath: str):
    if not os.path.exists(filepath):
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath, exist_ok=True)
        return True
    return False


month_codes = {
    'F': 1,   # January
    'G': 2,   # February
    'H': 3,   # March
    'J': 4,   # April
    'K': 5,   # May
    'M': 6,   # June
    'N': 7,   # July
    'Q': 8,   # August
    'U': 9,   # September
    'V': 10,  # October
    'X': 11,  # November
    'Z': 12   # December
}

# Reverse map
month_nums_to_codes = {v: k for k, v in month_codes.items()}


def get_current_contracts(ref_date, months, count):
    results = []
    months = sorted(months)
    year = ref_date.year

    while len(results) < count:
        for m in months:
            contract_date = datetime(year, m, 1)
            if contract_date >= ref_date:
                code = month_nums_to_codes[m]
                results.append((code, year))
                if len(results) == count:
                    return results
        year += 1

    return results


def resolve_year(month_code: str, year_digit: int, active_contracts: List[Tuple]) -> int:
    """Resolve year digit using contract position relative to the last active contract."""
    # Get the most recent contract (latest year and month)
    latest_year, latest_month = max(
        ((y, month_codes[m]) for m, y in active_contracts),
        key=lambda x: (x[0], x[1])
    )
    # Determine current and previous decades from the latest year
    decade = (latest_year // 10) * 10
    candidate_year = decade + year_digit

    # If the candidate is *after* the latest contract, it must belong to the *previous* decade
    if (candidate_year, month_codes[month_code]) > (latest_year, latest_month):
        return decade - 10 + year_digit
    return candidate_year


def get_third_friday(year: int, month: int) -> date:
    """Return the third Friday of the given month/year."""
    c = calendar.Calendar()
    fridays = [d for d in c.itermonthdates(year, month)
               if d.month == month and d.weekday() == calendar.FRIDAY]
    return fridays[2]


def get_cme_contract_expiry_date(symbol: str) -> int:
    """Parse one or more contract symbols (e.g., MNQH5 or MNQH5-MNQM5)."""
    # CME rules:
    # The nearest 8(in max case, usually less than 8) quarterly contracts (i.e., March, June, September, December for the next 2 years)
    active_contracts = get_current_contracts(datetime.today(), [3, 6, 9, 12], 8)
    results = []
    parts = symbol.split('-')
    for part in parts:
        root = part[:-2]
        month_code = part[-2]
        year_digit = int(part[-1])
        if month_code not in month_codes:
            assert False, f"{part}: Invalid month code"
        month = month_codes[month_code]
        year = resolve_year(month_code, year_digit, active_contracts)
        expiry = get_third_friday(year, month)
        results.append(int(expiry.strftime('%Y%m%d')))
    return min(results)


def check_missing_minutes(minutes: List):
    """ in ['202504212359', ...] """
    minutes = [str(m) for m in minutes]
    start = datetime.strptime(minutes[0], "%Y%m%d%H%M")
    end = datetime.strptime(minutes[-1], "%Y%m%d%H%M")
    expected = set()
    current = start
    while current <= end:
        expected.add(current.strftime("%Y%m%d%H%M"))
        current += timedelta(minutes=1)
    given = set(minutes)
    missing = sorted(expected - given)
    print(missing)


# @lru_cache(maxsize=None)
def _build_session(start_str: str, end_str: str, start2_str: str = '', end2_str: str = '') -> Tuple[pd.DatetimeIndex, int]:
    """
    Build a minute-level session between start and end times (ISO-format strings).
    Optionally append a second segment.
    Returns (DatetimeIndex, length).
    """
    # Convert ISO strings back to datetimes
    start = datetime.fromisoformat(start_str)
    end = datetime.fromisoformat(end_str)
    session = pd.date_range(start=start, end=end, freq='1min')
    if (start2_str != '') and (end2_str != ''):
        start2 = datetime.fromisoformat(start2_str)
        end2 = datetime.fromisoformat(end2_str)
        session = session.append(pd.date_range(start=start2, end=end2, freq='1min'))
    return session, len(session)


def get_cme_day_session(trade_days: List[str], i: int) -> Tuple[pd.DatetimeIndex, int]:
    """
    Generate CME minute-session for trade_days[i], where trade_days are 'YYYYMMDD' strings.
    i must satisfy 0 < i < len(trade_days) - 1.
    Sessions:
      • Middle-day (continuous prev & next): 00:00–15:59, then 17:00–23:59
      • First after gap: 17:00–23:59
      • Last before gap: 00:00–15:59
    Caching by ISO-format strings ensures safety/hashes.
    """
    # Parse only the three relevant dates
    fmt = "%Y%m%d"
    prev_day = datetime.strptime(trade_days[i - 1], fmt)
    curr_day = datetime.strptime(trade_days[i], fmt)
    next_day = datetime.strptime(trade_days[i + 1], fmt)

    prev_gap = (curr_day - prev_day).days > 1
    next_gap = (next_day - curr_day).days > 1

    # Middle-of-chain day
    if not prev_gap and not next_gap:
        s1 = curr_day
        e1 = curr_day + timedelta(hours=15, minutes=59)
        s2 = curr_day + timedelta(hours=17)
        e2 = curr_day + timedelta(hours=23, minutes=59)
        return _build_session(s1.isoformat(), e1.isoformat(), s2.isoformat(), e2.isoformat())

    # First day after a break
    if prev_gap and not next_gap:
        start = curr_day + timedelta(hours=17)
        end = curr_day + timedelta(hours=23, minutes=59)
        return _build_session(start.isoformat(), end.isoformat())

    # Last day before a break
    if not prev_gap and next_gap:
        start = curr_day
        end = curr_day + timedelta(hours=15, minutes=59)
        return _build_session(start.isoformat(), end.isoformat())

    raise ValueError(f"Invalid surrounding gaps for {trade_days[i]}")

def get_A_stock_day_session(trade_day: str) -> Tuple[pd.DatetimeIndex, int]:
    """
    Generate SSE/SZSE minute-session for trade_day, where trade_day is in 'YYYYMMDD' strings.
    Sessions:
      • day: 09:30–11:29, then 13:00–14:59
    """
    # Parse only the three relevant dates
    fmt = "%Y%m%d"
    curr_day = datetime.strptime(trade_day, fmt)

    s1 = curr_day + timedelta(hours=9, minutes=30)
    e1 = curr_day + timedelta(hours=11, minutes=29)
    s2 = curr_day + timedelta(hours=13)
    e2 = curr_day + timedelta(hours=14, minutes=59)
    return _build_session(s1.isoformat(), e1.isoformat(), s2.isoformat(), e2.isoformat())

def filter_A_session(df:pd.DataFrame):
    print(df.index.dtype)
    hour_min = df.index % 10000  # Get HHMM part
    # Filter for Shanghai A-share session (09:30–11:29 and 13:00–14:59)
    mask = (
        ((hour_min >= 930) & (hour_min <= 1129)) |
        ((hour_min >= 1300) & (hour_min <= 1459))
    )
    return df[mask]

def plot_df_heatmap(df:pd.DataFrame, label:str, value:str):
    import plotly.graph_objects as go
    
    x = pd.to_datetime(df.index.astype(str), format='%Y%m%d%H%M').strftime('%H%M-%Y%m%d')
    
    # Normalize 'b' for color mapping
    norm_b = (df[label] - df[label].min()) / (df[label].max() - df[label].min())
    # Create a 2-row heatmap background using b
    z = np.tile(norm_b.values, (2, 1))  # shape (2, N)

    # Create figure
    fig = go.Figure()

    # Add background as heatmap
    fig.add_trace(go.Heatmap(
        z=z,
        x=x,
        y=[df[value].min(), df[value].max()],
        colorscale='Viridis',
        showscale=False,
        zmin=0,
        zmax=1,
        opacity=0.6
    ))

    # Overlay the line for column 'a'
    fig.add_trace(go.Scatter(
        x=x,
        y=df[value],
        mode='lines',
        line=dict(color='black'),
        name=value
    ))

    fig.update_layout(
        title="Plot of 'a' with 'b' as vertical background color",
        xaxis_title="Time",
        yaxis_title=value,
        yaxis=dict(range=[df[value].min(), df[value].max()]),
        # xaxis=dict(showgrid=False, type='category'),  # Categorical axis spacing
        template='plotly_white'
    )

    fig.show()

def plot_premium(etf_symbols:List[str], df:pd.DataFrame, hours:int=4):
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    index = pd.to_datetime(df.index.astype(str), format='%Y%m%d%H%M').strftime('%H%M-%Y%m%d')
    
    session = np.where(df['is_etf_session'], 10, 0)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2])
    
    # px.line(x=plot_index, y=df[f"{sym}_premium"], width=1800, height=900).write_image(os.path.join(self.dir, f"fig/premium_{sym}.png"))
    for sym in etf_symbols:
        fig.add_trace(go.Scatter(x=index, y=df[f"{sym}_premium"], mode='lines', name=f"{sym}_premium"), row=1, col=1)
    fig.add_trace(go.Scatter(x=index, y=session, mode='lines', name=f"ETF_Session", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=index, y=df[f"close"], mode='lines', name=f"Main_Contract: {df.iloc[-1]['main_contract']}"), row=2, col=1)
    
    # fig.add_trace(go.Scatter(x=[date*10000 for date, contract in fut_roll], y=[0] * len(fut_roll), mode='markers', name='Future Rolls'))
    fig.update_layout(
        title='QDII NASDAQ100 price premium to NAV',
        # xaxis=dict(showgrid=False, type='category'),  # Categorical axis spacing
        xaxis_title='timestamp(min)',
        yaxis_title='premium(percent)',
        template='plotly_white',
        showlegend=True,
        height=700,  # This affects the absolute pixel height of the whole figure
    )
    
    fig.update_xaxes(
        showgrid=True,
        tickmode='linear', # Force evenly spaced ticks
        dtick=60*hours, # Set tick interval (e.g., every 1 unit)
        gridcolor='lightgray',
        gridwidth=0.5
    )
    fig.show()

def plot_nav(etf_symbols:List[str], df:pd.DataFrame, hours:int=4):
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    index = pd.to_datetime(df.index.astype(str), format='%Y%m%d%H%M').strftime('%H%M-%Y%m%d')
        
    fig = go.Figure()
    
    # px.line(x=plot_index, y=df[f"{sym}_premium"], width=1800, height=900).write_image(os.path.join(self.dir, f"fig/premium_{sym}.png"))
    for sym in etf_symbols:
        fig.add_trace(go.Scatter(x=index, y=df[f"{sym}_nav"], mode='lines', name=f"{sym}_nav"))
        fig.add_trace(go.Scatter(x=index, y=df[f"{sym}_nav_pointer"], mode='lines', name=f"{sym}_nav_pointer"))
    
    # fig.add_trace(go.Scatter(x=[date*10000 for date, contract in fut_roll], y=[0] * len(fut_roll), mode='markers', name='Future Rolls'))
    fig.update_layout(
        title='QDII NASDAQ100 NAV to index',
        # xaxis=dict(showgrid=False, type='category'),  # Categorical axis spacing
        xaxis_title='timestamp(min)',
        yaxis_title='NAV',
        template='plotly_white',
        showlegend=True,
        height=700,  # This affects the absolute pixel height of the whole figure
    )
    
    fig.update_xaxes(
        showgrid=True,
        tickmode='linear', # Force evenly spaced ticks
        dtick=60*hours, # Set tick interval (e.g., every 1 unit)
        gridcolor='lightgray',
        gridwidth=0.5
    )
    fig.show()
