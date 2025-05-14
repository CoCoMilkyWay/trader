import os
import sys
import numpy as np
import pandas as pd
import exchange_calendars as ecals

from tqdm import tqdm
from typing import List, Dict, Any
from datetime import date, datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Arbitrage.Util import *

pd.set_option('display.max_rows', None)

dir = os.path.dirname(os.path.abspath(__file__))
FUT_CSV = path = os.path.join(dir, 'misc', 'data', 'MNQ_20190414-20250422.ohlcv-1m.CSV')
FUT_PAR = path = os.path.join(dir, 'misc', 'data', 'MNQ_20190414-20250422.ohlcv-1m.parquet')

trim_time = None  # None/202501010000

fut_exchange = 'CME'
fut_exchange_ecals_index = 'CMES'
fut_exchange_tz = 'US/Central'

etf_exchange = 'SSE'
etf_exchange_ecals_index = 'XSHG'
etf_exchange_tz = 'Asia/Shanghai'

def load_fut():
    # Load Reference Futures Data
    # time format: UTC time
    # ts_event,rtype,publisher_id,instrument_id,open,high,low,close,volume,symbol
    # 2019-05-05T22:03:00.000000000Z,33,1,8078,7748.750000000,7748.750000000,7748.750000000,7748.750000000,1,MNQM9
    print("Loading futures data...")

    # df_ref_1min = pd.read_csv(
    #     FUT_CSV,
    #     parse_dates=["ts_event"],
    #     index_col='ts_event',
    #     usecols=["ts_event", "open", "high", "low", "close", "volume", "symbol"]
    # )
    # df_ref_1min.index = pd.to_datetime(df_ref_1min.index).tz_convert(fut_exchange_tz).strftime('%Y%m%d%H%M').astype('int64')
    # df_ref_1min.to_parquet(FUT_PAR)

    df_ref_1min = pd.read_parquet(
        FUT_PAR,
        columns=["ts_event", "open", "high", "low", "close", "volume", "symbol"],
    )

    print(f"Loaded data shape: {df_ref_1min.shape}")

    # Step 0: Create complete minute-level index for the entire period as int64
    df_ref_1min = trim(df_ref_1min)
    start_time = pd.to_datetime(str(df_ref_1min.index.min()), format='%Y%m%d%H%M')
    end_time = pd.to_datetime(str(df_ref_1min.index.max()), format='%Y%m%d%H%M')

    trade_days = get_tradedays(fut_exchange_ecals_index, start_time.date(), end_time.date(), type='futures')
    trading_sessions = []
    for i in range(1, len(trade_days) - 1):
        trading_sessions.append(get_cme_day_session(trade_days, i)[0])

    # remove 1st/last day for potentially imcomplete data
    trading_sessions = trading_sessions[1:-1]
    valid_index = pd.DatetimeIndex(sorted(set().union(*trading_sessions)))
    full_index = valid_index.to_series().dt.strftime('%Y%m%d%H%M').astype('int64')
    column_names = ['open', 'high', 'low', 'close', 'volume']
    df_temp = pd.DataFrame(index=full_index, columns=column_names, dtype=np.float16)
    print('full trading sessions time index generated')

    print('Splitting reference data by symbol...')
    for symbol, group_symbol in df_ref_1min.groupby("symbol"):
        parts = str(symbol).split('-')
        if len(parts) != 1:
            print(f'Skipping: {symbol}')
            # skip spread futures
            continue

        expiry_date = get_cme_contract_expiry_date(str(symbol))  # e.g., '20250510' as str
        expiry_threshold = int(str(expiry_date) + '2359')  # '202505102359'

        df = df_temp.copy()
        df[['open', 'high', 'low', 'close', 'volume']] = group_symbol[['open', 'high', 'low', 'close', 'volume']].reindex(full_index)

        # Filter by expiry date
        df = df[df.index <= expiry_threshold]

        df['close'] = df['close'].ffill()
        df['volume'] = df['volume'].fillna(0)

        missing_mask = df['open'].isna()
        df.loc[missing_mask, ['open', 'high', 'low']] = df.loc[missing_mask, 'close'].values[:, None].repeat(3, axis=1)

        print(f"Complete reference data shape: {symbol}:{df.shape}")
        df['date'] = df.index//10000
        for date, group_date in df.groupby("date"):
            daily_volume = int(group_date['volume'].sum())
            filepath = os.path.join(dir, f"history/{symbol}/{date}_{daily_volume}.parquet")
            mkdir(filepath)
            if daily_volume != 0:
                group_date.drop(columns='date', inplace=True)
                group_date.to_parquet(filepath)
            else:
                pass
                # pd.DataFrame([]).to_parquet(filepath)
        with open(os.path.join(dir, f"history/{symbol}/VALID_{min(df['date'])}_{max(df['date'])}"), 'w'):
            pass  # Just create the file without writing anything

    print(df[-10000:])
    print("Processing complete!")

def load_etf():
    etfs = [
    ['159941', 'SZ', '   '],  # 纳指ETF
    ['513100', 'SH', '044'],  # 纳指ETF
    ['159632', 'SZ', '   '],  # 纳斯达克ETF
    ['159509', 'SZ', '   '],  # 纳指科技ETF
    ['159501', 'SZ', '   '],  # 纳斯达克指数ETF
    ['159513', 'SZ', '   '],  # 纳斯达克100指数ETF
    ['513300', 'SH', '286'],  # 纳斯达克ETF
    ['159659', 'SZ', '   '],  # 纳斯达克100ETF
    ['159696', 'SZ', '   '],  # 纳指ETF易方达
    ['159660', 'SZ', '   '],  # 纳指100ETF
    ['513110', 'SH', '551'],  # 纳斯达克100ETF
    ['513390', 'SH', '581'],  # 纳指100ETF
    ['513870', 'SH', '635'],  # 纳指ETF富国
    ]
    etfs = [etf[0] for etf in etfs]
    
    print("Loading etf data...")
    
    for etf in etfs:
        
        # Prepare file paths
        bar_filename = f"{etf}.csv"
        bar_path = os.path.join(dir, 'misc/data', bar_filename)
        par_filename = f"{etf}.parquet"
        par_path = os.path.join(dir, 'misc/data', par_filename)
        nav_filename = f"NAV_{etf}.csv"
        nav_path = os.path.join(dir, 'misc/data', nav_filename)

        # # Process bar data
        # # Load minute-level bar data (Shanghai time)
        # # time format: Shanghai time
        # # trade_time,open,high,low,close,vol,amount
        # # 2023-05-08 09:31:00,1.016,1.016,1.013,1.014,27024700.0,27420472.0
        # df_bar_1min = pd.read_csv(
        #     bar_path,
        #     parse_dates=["trade_time"],
        #     index_col='trade_time',
        #     usecols=["trade_time", "open", "high", "low", "close", "vol"]
        # )
        # df_bar_1min.index = (pd.to_datetime(df_bar_1min.index).tz_localize(etf_exchange_tz)-timedelta(minutes=1)).strftime('%Y%m%d%H%M').astype('int64')
        # df_bar_1min = df_bar_1min.rename(columns={"vol": "volume"}).rename_axis("ts_event")
        # df_bar_1min.to_parquet(par_path)

        df_bar_1min = pd.read_parquet(
            par_path,
            columns=["ts_event", "open", "high", "low", "close", "volume"],
        )
        
        # Convert datetime index to int64 format YYYYMMDDHHMM
        df_bar_1min = trim(df_bar_1min)

        start_time = pd.to_datetime(str(df_bar_1min.index.min()), format='%Y%m%d%H%M')
        end_time = pd.to_datetime(str(df_bar_1min.index.max()), format='%Y%m%d%H%M')
        trade_days = get_tradedays(etf_exchange_ecals_index, start_time.date(), end_time.date(), type='spot')
        trading_sessions = []
        for trade_day in trade_days:
            trading_sessions.append(get_A_stock_day_session(trade_day)[0])
        # remove 1st/last day for potentially imcomplete data
        valid_index = pd.DatetimeIndex(sorted(set().union(*trading_sessions)))
        full_index = valid_index.to_series().dt.strftime('%Y%m%d%H%M').astype('int64')
        
        column_names = ['open', 'high', 'low', 'close', 'volume']
        df_temp = pd.DataFrame(index=full_index, columns=column_names, dtype=np.float16)
        
        df = df_temp.copy()
        df[['open', 'high', 'low', 'close', 'volume']] = df_bar_1min[['open', 'high', 'low', 'close', 'volume']].reindex(full_index)
        # Filter by expiry date
        df['close'] = df['close'].ffill()
        df['volume'] = df['volume'].fillna(0)
        missing_mask = df['open'].isna()
        df.loc[missing_mask, ['open', 'high', 'low']] = df.loc[missing_mask, 'close'].values[:, None].repeat(3, axis=1)
        
        # Process NAV data more efficiently
        # Load daily NAV data
        # time format: date
        # trade_time,nav,subscription,redemption (1:enabled, 0:disabled -1:funding)
        # 2023-05-31,1.0,1,1
        df_nav_1day = pd.read_csv(
            nav_path,
            parse_dates=["trade_time"],
            index_col='trade_time',
            usecols=["trade_time", "nav"] # "subscription", "redemption"
        )
        
        df_nav_1day.index = (pd.to_datetime(df_nav_1day.index).tz_localize(etf_exchange_tz)).strftime('%Y%m%d').astype('int64')
        daily_nav_map = df_nav_1day["nav"].to_dict()
        
        print(f"Complete reference data shape: {etf}:{df.shape}")
        
        df['date'] = df.index//10000
        valid_days = []
        for date, group_date in df.groupby("date"):
            NAV_or_None = daily_nav_map.get(date)
            # assert NAV is not None, f"cannot get Nav for {etf}: {date}"
            if not NAV_or_None:
                print(f"{RED}Missing NAV data: {etf}: {date}{RESET}")
            else:
                NAV = NAV_or_None
            
            if date == 20250424:
                continue
            
            valid_days.append(date)
            filepath = os.path.join(dir, f"history/{etf}/{date}_{NAV}.parquet") # use last NAV if current is None
            mkdir(filepath)
            group_date.drop(columns='date', inplace=True)
            group_date.to_parquet(filepath)
        
        valid_start = min(valid_days)
        valid_end = max(valid_days)
        start_time = pd.to_datetime(str(valid_start), format='%Y%m%d')
        end_time = pd.to_datetime(str(valid_end), format='%Y%m%d')
        valid_trade_days = get_tradedays(etf_exchange_ecals_index, start_time.date(), end_time.date(), type='spot')
        valid_trade_days = [int(d) for d in valid_trade_days]
        
        set1, set2 = set(valid_trade_days), set(valid_days)
        if set1 != set2:
            only_in_1 = set1 - set2
            only_in_2 = set2 - set1
            print("NAV days that are missing:", only_in_1)
            print("NAV days that are extra:", only_in_2)
            # assert False, "Lists do not have the same members"

        with open(os.path.join(dir, f"history/{etf}/VALID_{valid_start}_{valid_end}"), 'w'):
            pass  # Just create the file without writing anything
        print(df[:2])
    
    print("Processing complete!")

def trim(df):
    if trim_time:
        return df[df.index >= trim_time]
    else:
        return df


def get_tradedays(exg: str, start_date: date, end_date: date, type: str) -> List[str]:
    """
    Note that a session may start on the day prior to the session label or
    end on the day following the session label. Such behaviour is common
    for calendars that represent futures exchange
    """
    def fill_gaps_with_previous_day(dates):
        # Convert strings to datetime objects
        date_objs = [datetime.strptime(d, '%Y%m%d') for d in dates]
        result = [date_objs[0]]
        for i in range(1, len(date_objs)):
            prev = date_objs[i - 1]
            curr = date_objs[i]
            if (curr - prev).days > 1:
                # Insert the day before the current date
                result.append(curr - timedelta(days=1))
            result.append(curr)
        # Convert back to string format
        return [d.strftime('%Y%m%d') for d in result]
    trade_days = ecals.get_calendar(exg).sessions_in_range(
        pd.Timestamp(start_date),
        pd.Timestamp(end_date)
    ).strftime('%Y%m%d').tolist()
    if type == 'spot':
        return trade_days
    elif type == 'futures':
        return fill_gaps_with_previous_day(trade_days)
    else:
        assert False, f"Invalid type: {type}"


if __name__ == "__main__":
    load_etf()
