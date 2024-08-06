from __future__ import (absolute_import, division, print_function, unicode_literals)
from math import nan
import os, sys
from unittest import result
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import baostock as bs

from wtpy.apps.datahelper import DHFactory as DHF
from run_db_maintain import cfg

class update_helper:
    def __init__(self):
        self.hlper = DHF.createHelper("baostock")
        self.hlper.auth()

    # Assuming the structure is deeply nested, you might need to normalize it:
    def normalize_data(self, data, meta):
        """ Flatten JSON structure """
        return pd.DataFrame({meta: data[meta]})

    def compare_dataframes(self, df1, df2, meta, sort, header):
        # Convert dictionary columns to strings for precise comparison
        df1['meta_str'] = df1[meta].apply(lambda x: str(sorted(x.items()) if sort else x))
        df2['meta_str'] = df2[meta].apply(lambda x: str(sorted(x.items()) if sort else x))

        # Find common keys and unique keys in each DataFrame
        common_keys = df1.index.intersection(df2.index)
        added_keys = df2.index.difference(df1.index)
        missing_keys = df1.index.difference(df2.index)

        # Print added and missing keys
        if not added_keys.empty:
            print(header, f"Added entries in DF2: {list(added_keys)}")
        if not missing_keys.empty:
            print(header, f"Missing entries from DF1: {list(missing_keys)}")

        # Compare entries for common keys to find modifications
        modified_entries = []
        for key in common_keys:
            if df1.at[key, 'meta_str'] != df2.at[key, 'meta_str']:
                modified_entries.append(key)
                print(header, f"Modified entry at {key}:")
                print(header, f"  DF1: {df1.at[key, meta]}")
                print(header, f"  DF2: {df2.at[key, meta]}")

        if not modified_entries:
            print(header, "No modifications found.")

    def update_trade_holiday(self):
        import json
        import shutil
        try:
            self.hlper.dmpHolidayssToFile(filename_holidays='holidays.json', filename_tradedays='tradedays.json')
            previous_json = load_json(file_path=cfg.HOLIDAYS_FILE)
            current_json = load_json(file_path='holidays.json')
            # Convert JSON list to set for easier comparison
            current_dates = set(current_json['CHINA'])
            previous_dates = set(previous_json['CHINA'])
            # Find differences
            new_dates = current_dates - previous_dates
            missing_dates = previous_dates - current_dates
            print("[INFO ][maintain_D_Holidays]: New Adding:", new_dates)
            print("[INFO ][maintain_D_Holidays]: New Missing:", missing_dates)
            print("[INFO ][maintain_D_Holidays]: Overwriting")
            shutil.move('holidays.json', cfg.HOLIDAYS_FILE)
            shutil.move('tradedays.json', cfg.TRADEDAYS_FILE)
        except:
            print('[ERROR][maintain_D_Holidays]: cannot connect to akshare to update holidays')

    def update_assetlist(self):
        import shutil
        try:
            self.hlper.dmpCodeListToFile(filename='stocks.json')
            previous_json = load_json(file_path=cfg.STOCKS_FILE)
            current_json = load_json(file_path='stocks.json')

            # Load and normalize data
            for exchange in ['SSE', 'SZSE']:
                previous_normalized_data = self.normalize_data(previous_json, exchange)
                current_normalized_data = self.normalize_data(current_json, exchange)
                print(previous_normalized_data)
                self.compare_dataframes(previous_normalized_data, current_normalized_data, exchange, 1, '[INFO ][maintain_D_Assetlists]: ')
            print("[INFO ][maintain_D_Assetlists]: Overwriting")
            shutil.move('stocks.json', cfg.STOCKS_FILE)
        except:
            print('[ERROR][maintain_D_Assetlists]: cannot update asset lists')

    def update_adjfactors(self):
        import shutil
        try:
            asset_list_json =  load_json(cfg.STOCKS_FILE)

            def extract_stock_codes(json_data, exchange):
                """ Extract stock codes from JSON data based on the specified exchange """
                codes = []
                if exchange in json_data:
                    for code in json_data[exchange]:
                        codes.append(f"{exchange}.{code}")
                return codes

            stock_codes = []
            for exchange in ['SSE', 'SZSE']:
                stock_codes = stock_codes + extract_stock_codes(asset_list_json, exchange)
            self.hlper.dmpAdjFactorsToFile(codes=stock_codes, filename='adjfactors.json')
            print('[INFO ][maintain_D_Adjfactors]: Update Finished, Comparing')
            previous_json = load_json(file_path=cfg.ADJFACTORS_FILE)
            current_json = load_json(file_path='adjfactors.json')

            # Load and normalize data
            for exchange in ['SSE', 'SZSE']:
                previous_normalized_data = self.normalize_data(previous_json, exchange)
                current_normalized_data = self.normalize_data(current_json, exchange)
                self.compare_dataframes(previous_normalized_data, current_normalized_data, exchange, 0, '[INFO ][maintain_D_Adjfactors]: ')
            print("[INFO ][maintain_D_Adjfactors]: Overwriting")
            shutil.move('adjfactors.json', cfg.ADJFACTORS_FILE)
        except:
            print('[ERROR][maintain_D_Adjfactors]: cannot update adj-factors')
            
class database_helper:
    '''
    will auto load holidays, stocks, adjfactors, so make sure they are already updated
    '''
    def __init__(self):
        print('[INFO ][maintain: Reading DataBase Meta info]')
        # parse meta data
        self.holidays = load_json(cfg.HOLIDAYS_FILE)
        self.tradedays = load_json(cfg.TRADEDAYS_FILE)
        self.stocks = load_json(cfg.STOCKS_FILE)
        self.adjfactors = load_json(cfg.ADJFACTORS_FILE)
        def get_hs300_stocks():
            import baostock as bs
            import pandas as pd
            lg = bs.login()
            print('login respond error_code:'+lg.error_code)
            print('login respond  error_msg:'+lg.error_msg)
            rs = bs.query_hs300_stocks()
            print('query_hs300 error_code:'+rs.error_code)
            print('query_hs300  error_msg:'+rs.error_msg)
            hs300_stocks = []
            while (rs.error_code == '0') & rs.next():
                hs300_stocks.append(rs.get_row_data())
            bs.logout()
            hs300_df = pd.DataFrame(hs300_stocks, columns=rs.fields)
            hs300_ls = list(hs300_df['code'])
            return hs300_ls
        if cfg.SHRINK_STOCK_POOL:
            # TODO
            hs300 = get_hs300_stocks()
            exchange_dict = {'SSE': 'sh', 'SZSE': 'sz'}
            for e in ['SSE', 'SZSE']:
                limited_dict = {}
                count = 0
                for key, value in self.stocks[e].items():
                    # key: 300483 value: {'code': '300483', 'exchg': 'SZSE', 'name': '首华燃气', 'product': 'STK'}
                    asset = f'{exchange_dict[e]}.{key}'
                    count += 1
                    if count >0:
                        if asset in hs300:
                            limited_dict[key] = value
                        # if value['product'] == 'IDX':
                        #     limited_dict[key] = value
                        # if count == 100:
                        #     self.stocks[e] = limited_dict
                        #     break  # Stop after adding N items
                self.stocks[e] = limited_dict
        else:
            exchange_dict = {'SSE': 'sh', 'SZSE': 'sz'}
            for e in ['SSE', 'SZSE']:
                limited_dict = {}
                for key, value in self.stocks[e].items():
                    limited_dict[key] = value
                self.stocks[e] = limited_dict
                
        # these tables are huge, use parquet DB
        asset_list = ['sh.' + code for code in list(self.stocks['SSE'].keys())] + ['sz.' + code for code in list(self.stocks['SZSE'].keys())]
        date_list = [datetime.strptime(date, "%Y%m%d").date() for date in self.tradedays['CHINA']]
        try:
            print('[INFO ][maintain: Reading Meta-Data]')
            self.metadata = pq.read_table(mkdir(cfg.METADATA_FILE)).to_pandas()
            print('[INFO ][maintain: Finished Reading Meta-Data]')
        except FileNotFoundError:
            print('[ERROR][maintain: Creating Meta-Data Table]')
            self.hlper = DHF.createHelper("baostock")
            self.hlper.auth()
            self.metadata = pd.DataFrame({
                'code': pd.Series(dtype='string'),
                'asset_name': pd.Series(dtype='string'),
                'ipoDate': pd.Series(dtype='datetime64[ns]'),
                'outDate': pd.Series(dtype='datetime64[ns]'),
                'type': pd.Series(dtype='Int8'), # 1.stk 2.idx 3.others 4.convertable bond 5.etf
                'status': pd.Series(dtype='bool'), # 0.quit 1.active
                'exchange': pd.Series(dtype='string'),
                'industry_sector_level_1': pd.Series(dtype='string'),
                'industry_sector_level_2': pd.Series(dtype='string'),
                'reserved': pd.Series(dtype='string')
            }, index=asset_list)
            with tqdm(asset_list, total=len(asset_list)) as pbar:
                for asset in pbar:
                    self.update_metadata_by_code(asset)
                    # print(self.metadata.loc[code])
                    pbar.set_description(f'Processing {asset}: ')
            pq.write_table(pa.Table.from_pandas(self.metadata), cfg.METADATA_FILE)
        # view only: meta table
        print(f'[INFO ][maintain: Writing Meta to Json for reference: {cfg.METADATA_JSON_FILE}]')
        dump_json(mkdir(cfg.METADATA_JSON_FILE), self.metadata)
        
        print('[INFO ][maintain: Syncing Integrity Table]')
        integrity_percentage = {}
        asset = ''
        for asset in tqdm(asset_list, desc=f'Syncing: {asset}'):
            folder = f"{cfg.BAR_DIR}/m1/{asset}"
            integrity_file = f'{folder}/integrity.json'
            if cfg.FORCE_INTEGRITY_SYNC and os.path.exists(integrity_file):
                os.remove(integrity_file)
            if os.path.exists(folder) and not os.path.exists(integrity_file):
                ipo_date = self.metadata[self.metadata['asset'].str.endswith(asset)]['ipoDate'].iloc[0]
                ipo_date = pd.to_datetime(ipo_date, format='%Y-%m-%d').date()
                for idx, date in enumerate(date_list):
                    if ipo_date <= date:
                        break
                asset_trade_days = date_list[idx:]
                asset_integrity = []
                days_integrity = 0
                for trade_day in asset_trade_days:
                    year = trade_day.year
                    month = trade_day.month
                    day = trade_day.day
                    if os.path.exists(f'{folder}/{year}.{month}.dsb'):
                        asset_integrity += [True]; days_integrity += 1
                    elif os.path.exists(f'{folder}/{year}.{month}.{day}.dsb'):
                        asset_integrity += [True]; days_integrity += 1
                    else:
                        asset_integrity += [False]
                integrity_percentage[asset] = f'{(days_integrity/len(asset_trade_days)):03%}'
                index = pd.Index(asset_trade_days, name='days')
                # index = pd.MultiIndex.from_tuples(itertools.product(code_list, date_list), names=['asset_code', 'date'])
                integrity = pd.DataFrame({
                    'integ': pd.Series(asset_integrity, index=index, dtype=bool)
                    })
                integrity.to_json(mkdir(integrity_file), date_format='iso', orient='table')
                # integrity = pd.read_json(integrity_file, orient='table', convert_dates=['trade_days'])
                # integrity.set_index('trade_days', inplace=True)
        with open(mkdir(cfg.INTEGRITY_JSON_FILE), 'w') as file:
            import json
            json.dump(integrity_percentage, file, indent=4)  # 'indent=4' for pretty-printing
        # with open(cfg.INTEGRITY_JSON_FILE, 'r') as file:
        #     integrity = json.load(file)
        print('[INFO ][maintain: Created Integrity Tables]')
        
    def update_metadata_by_code(self, asset_str):
        query_basic = bs.query_stock_basic(code=asset_str)
        basic = get_baostock_info(query_basic)
        query_industry = bs.query_stock_industry(code=asset_str) # ShenWan level-1
        industry = get_baostock_info(query_industry)
        exchange, _ = get_sub_exchange(asset_str)
        self.metadata.loc[asset_str, 'asset'                  ] = asset_str
        self.metadata.loc[asset_str, 'asset_name'             ] = basic['code_name']
        self.metadata.loc[asset_str, 'ipoDate'                ] = np.datetime64(basic['ipoDate'])
        self.metadata.loc[asset_str, 'outDate'                ] = np.datetime64(basic['outDate'])
        self.metadata.loc[asset_str, 'type'                   ] = np.int8(basic['type']).item()
        self.metadata.loc[asset_str, 'status'                 ] = np.bool(basic['status']).item()
        self.metadata.loc[asset_str, 'exchange'               ] = exchange
        self.metadata.loc[asset_str, 'industry_sector_level_1'] = industry['industry']
        self.metadata.loc[asset_str, 'industry_sector_level_2'] = ''
        self.metadata.loc[asset_str, 'reserved'               ] = ''
        
    def main_process_assets_from_folders(self):
        # TODO
        print(self.stocks)
        assets = self.stocks
        exchanges = ['SSE', 'SZSE']
        exchanges_prefix = ['sh', 'sz']
        asset_list = []
        processed_asset_list = os.listdir(mkdir(f"{cfg.BAR_DIR}/m1/"))
        for idx, exchange in enumerate(exchanges):
            try:
                for asset in assets[exchange].keys():
                    if asset not in processed_asset_list:
                        asset_list.append(exchanges_prefix[idx] + '.' + asset)
                    else:
                        print(f'asset {asset} already processed')
            except:
                exchanges.remove(exchange)
                print(f'removing exchange: {exchange}')
        num_assets = len(asset_list)
        print(f'num of assets to be processed: {num_assets}')
        asset_jobs = []
        for asset in asset_list:
            #　if asset != 'sz.000001':
            #　    continue
            exch_prefix = asset.split('.')[0]
            code = asset.split('.')[-1]
            for idx, exchange in enumerate(exchanges):
                try:
                    if exchanges_prefix[idx] == exch_prefix:
                        adjfactors = self.adjfactors[exchange][code]
                        ipodate = self.metadata.loc[asset, 'ipoDate']
                        # if adjfactors != []:
                        asset_dict = {asset: [adjfactors, ipodate]}
                        asset_jobs += [asset_dict]
                except KeyError:
                    pass
        processor = n_slave_1_master_queue(
            tqdm_total=num_assets,
            max_workers=cfg.max_workers,
            concurrency_mode=cfg.concurrency_mode)
        processor.add_slave_task(asset_jobs) # submit jobs
        processor.execute()
        
    def main_process_assets_from_api(self):
        pass

import signal
terminate = False
def signal_handler(signum, frame):
    global terminate
    terminate = True
    print("Interrupt received, terminating...")
signal.signal(signal.SIGINT, signal_handler)

def load_json(file_path):
    import json
    with open(file_path, 'r', encoding='gbk', errors='ignore') as file:  # Note the 'utf-8-sig' which handles BOM if present
        return json.load(file)

def dump_json(file_path, df):
    import json
    with open(file_path, 'w', encoding='gbk', errors='ignore') as file:
        df.to_json(file, orient='records', force_ascii=False, indent=4)

def mkdir(path_str):
    path = os.path.dirname(path_str)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path_str

def log(path_str, str, stdout=1, type='a'):
    # 'w+': write and read
    # 'w': overwrite
    # 'a': append
    with open(path_str,type) as file:
        # Writing data to the file
        file.write(f'{str}\n')
        #　# Moving the cursor to the beginning of the file to read from the beginning
        #　file.seek(0)
        #　data = file.read()
    if stdout:
        print(str)
    return str

def get_baostock_info(resultset):
    data_list = []
    while (resultset.error_code == '0') & resultset.next():
        data_list.append(resultset.get_row_data())
    return pd.DataFrame(data_list, columns=resultset.fields, index=[0]).iloc[0]

def get_sub_exchange(code):
    # Remove 'sh' or 'sz' prefix if present
    if code.startswith('sh') or code.startswith('sz'):
        code = code[3:]
    # Now check based on the numeric part
    if code.startswith('60'):
        return 'SSE.A', 1
    elif code.startswith('900'):
        return 'SSE.B', 1
    elif code.startswith('68'):
        return 'SSE.STAR', 2
    elif code.startswith('000') or code.startswith('001'):
        return 'SZSE.A', 1
    elif code.startswith('200'):
        return 'SZSE.B', 1
    elif code.startswith('300') or code.startswith('301'):
        return 'SZSE.SB', 2
    elif code.startswith('002') or code.startswith('003'):
        return 'SZSE.A', 1
    elif code.startswith('440') or code.startswith('430') or code.startswith('83') or code.startswith('87'):
        return 'NQ', 3
    else:
        return 'Unknown', 0

# import logging
# formatter = logging.Formatter('%(asctime)s - %(levelname)s')
# # integrity logger
# log_int = logging.getLogger('integrity')
# log_int.setLevel(logging.WARN)
# handler_int_file = logging.FileHandler(mkdir(f'{cfg.DB0}/integrity.log'))
# handler_int_file.setFormatter(formatter)
# handler_int_cons = logging.StreamHandler()
# log_int.addHandler(handler_int_file)
# log_int.propagate = False  # Prevent logs from being handled by the root logger's handlers
# # summary logger
# log_sum = logging.getLogger('summary')
# log_sum.setLevel(logging.INFO)
# handler_sum_file = logging.FileHandler(mkdir(f'{cfg.DB0}/summary.log'))
# handler_sum_file.setFormatter(formatter)
# handler_sum_cons = logging.StreamHandler()
# log_sum.addHandler(handler_sum_file)
# log_sum.propagate = False  # Prevent logs from being handled by the root logger's handlers
# # Remove the console handler from the root logger if it was added there

from wtpy.wrapper import WtDataHelper
from wtpy.WtCoreDefs import WTSBarStruct
dtHelper = WtDataHelper()
def store_bars(df, file_path): # 'date' 'time' 'open' 'high' 'low' 'close' 'vol'
    if os.path.exists(file_path):
        pass
    else:
        df['date'] = df['date'].astype('datetime64[ns]').dt.strftime('%Y%m%d').astype('int64')
        df['time'] = df['time'].apply(lambda t: t.hour * 10000 + t.minute * 100 + t.second)
        df = df.drop(['month'], axis=1)
        BUFFER = WTSBarStruct*len(df)
        buffer = BUFFER()
        def assign(procession, buffer):
            tuple(map(lambda x: setattr(buffer[x[0]], procession.name, x[1]), enumerate(procession)))
        df.apply(assign, buffer=buffer)
        store_path = mkdir(file_path)
        dtHelper.store_bars(barFile=store_path, firstBar=buffer, count=len(df), period="m1")

def check_integrity_per_asset(df, asset_dict, daily_k_bar):
    asset_adjfactors = list(asset_dict.values())[0][0]
    ipo_date = list(asset_dict.values())[0][1]
    # TODO
    asset = list(asset_dict.keys())[0]
    log_path = f"{cfg.BAR_DIR}/m1/{asset}/log.txt"
    if not os.path.exists(log_path): # logging
        log(mkdir(log_path), asset_adjfactors, stdout=0, type='w')
        log(log_path, ipo_date, stdout=0)
    if cfg.CHECK_4:
        adj_days = [day_dict['date'] for day_dict in asset_adjfactors]
        adj_factors = [day_dict['factor'] for day_dict in asset_adjfactors]
    days_passed = []
    prev_close = [df['close'][0]] * 2
    pre_adj_factor = 1
    
    for year, group_year in df.groupby('year'):
        for month, group_month in group_year.groupby('month'):
            group_month = group_month[group_month['year'] == year]
            file_path = f"{cfg.BAR_DIR}/m1/{asset}/{year}.{month}.dsb"
            if os.path.exists(file_path):
                continue
            rules_violated_tomonth = 0
            
            # sys.exit()
            for day, group_date in group_month.groupby('day'):
                date_str = f'{year}{month:02}{day:02}'
                date = pd.to_datetime(date_str, format='%Y%m%d')
                prev_close = [group_date['close'].iloc[-1], prev_close[0]]
                file_path = f"{cfg.BAR_DIR}/m1/{asset}/{year}.{month}.{day}.dsb"
                if os.path.exists(file_path):
                    continue
                
                if cfg.CHECK_5:
                    valid = 1
                    skip = 0
                    num_min_bars = group_date.shape[0]
                    # Rule 5: Verify day-open/close/mid-break price from other sources
                    if (num_min_bars<200):
                        rules_violated_tomonth |= 1<<5
                        log(log_path, f'[integrity][5]: not enough monthly data: {asset}: {year}.{month}')
                        continue
                    k_bar = daily_k_bar[(year == daily_k_bar['year']) &
                                        (month == daily_k_bar['month']) &
                                        (day == daily_k_bar['day'])]
                    if k_bar.shape[0] != 1:
                        valid = 1
                        skip = 1
                        log(log_path, f'[integrity][5]:{asset}:{year}.{month}.{day} skipping: error finding matched day k-bar', stdout=0)
                    if valid == 1 and not skip:
                        expected_open   = k_bar['open'].iloc[0]
                        expected_close  = k_bar['close'].iloc[0]
                        actual_open     = group_date['open'].iloc[0]
                        actual_close    = group_date['close'].iloc[-1]
                        if  actual_open    != expected_open or \
                            actual_close   != expected_close:
                                # tolerate a bit diff
                                if abs(actual_open - expected_open) > 1:
                                    valid =0
                                if abs(actual_close - expected_close) > 1:
                                    valid =0
                    if valid == 0:
                        rules_violated_tomonth |= 1<<5
                        log(log_path, f'[integrity][5]: not matching data source: {asset}: {year}.{month}.{day}: {expected_open}, {actual_open}, {expected_close}, {actual_close}')
                        continue
                    
                if cfg.CHECK_0:
                    # Rule 0: Non-zero/NaN/NaT OHLC
                    if (group_date[['open', 'high', 'low', 'close']].isnull().any().any() or
                        (group_date[['open', 'high', 'low', 'close']] == 0).any().any()):
                        rules_violated_tomonth |= 1<<0
                        log(log_path, f'[integrity][0]: bad data:{elem}')
                        continue

                if cfg.CHECK_1:
                    # Rule 1: Timestamp continuity/order/completeness
                    # Augest 2018: modify rules of after-hour-call-auction： 14:55 to 14:57
                    morning_times = pd.date_range(start=f"09:30", end=f"11:31", freq='min')
                    afternoon_times = pd.date_range(start=f"13:00", end=f"15:01", freq='min')
                    # uncheck_times = pd.date_range(start=f"{date} 14:56", end=f"{date} 15：00", freq='min')
                    expected_times = morning_times.union(afternoon_times).time.tolist() # type: ignore
                    actual_times = group_date['time'].tolist()
                    unexpected_time = False
                    for elem in actual_times:
                        if elem not in expected_times:
                            rules_violated_tomonth |= 1<<1
                            unexpected_time = True
                            log(log_path, f'[integrity][1]: unexpected trading time:{elem}')
                            continue
                    if not unexpected_time:
                        if len(actual_times) < len(expected_times) - 5:
                            rules_violated_tomonth |= 1<<1
                            log(log_path, f'[integrity][1]: not enough data: {len(expected_times)}, {len(actual_times)}')
                            continue
                        
                if cfg.CHECK_2:
                    # Rule 2: Intra-day price relation (may not be continuous)
                    if ((group_date['high']     < group_date['low']).any() or 
                        (group_date['open']     > group_date['high']).any() or 
                        (group_date['open']     < group_date['low']).any() or 
                        (group_date['close']    > group_date['high']).any() or 
                        (group_date['close']    < group_date['low']).any()):
                        rules_violated_tomonth |= 1<<2
                        log(log_path, f"[integrity][2]: wrong intra-day OHCL: \
                                     {group_date['open'].iloc[0]}, \
                                     {group_date['close'].iloc[0]}, \
                                     {group_date['high'].iloc[0]}, \
                                     {group_date['low'].iloc[0]}")
                        continue

                if cfg.CHECK_3:
                    # Rule 3: Inter-day price jump limit
                    adjusted_prev_close = prev_close[1]
                    adjusted = 0
                    if(len(adj_days) > 0):
                        for idx, adj_day in enumerate(adj_days):
                            if adj_day == int(date_str):
                                adjusted = 1
                                adj_factor_now = adj_factors[idx]
                                adj_factor_before = adj_factor_now if idx == 0 else adj_factors[idx-1]
                                pre_adj_factor = adj_factor_before/adj_factor_now
                                adjusted_prev_close = prev_close[1] * pre_adj_factor
                    current_open = group_date['open'].iloc[0]
                    current_close = group_date['close'].iloc[-1]
                    _, board = get_sub_exchange(asset)
                    jump_limit = board * 0.1 + 0.01  # max 10/20/30% even with call-auction
                    if abs(current_open - adjusted_prev_close) > adjusted_prev_close * jump_limit:
                        if adjusted:
                            log(log_path, f'[integrity][3]: {year}.{month}.{day} day bar price jump: {prev_close[1]}, {pre_adj_factor} ({adj_factor_before}/{adj_factor_now}), {adjusted_prev_close}, {current_open}')
                        else:
                            log(log_path, f'[integrity][3]: {year}.{month}.{day} day bar price jump: {prev_close[1]}, {adjusted_prev_close}, {current_open}')
                            
                        # TODO: skip checking for adjfsctors:
                        # rules_violated_tomonth |= 1<<3
                        # continue

                if cfg.CHECK_4:
                    # Rule 4: OHLC equal if volume is zero
                    if ((group_date['volume'] == 0) & 
                        (group_date['open'] != group_date['high']) & 
                        (group_date['high'] != group_date['low']) & 
                        (group_date['low'] != group_date['close'])).any():
                        rules_violated_tomonth |= 1<<4
                        log(log_path, f'[integrity][4]: OHLC non-equal for 0 volume')
                        continue
                # TODO
                # days_passed.append(date)
                file_path = f"{cfg.BAR_DIR}/m1/{asset}/{year}.{month}.{day}.dsb"
                store_bars(group_date, file_path) # 'date' 'time' 'open' 'high' 'low' 'close' 'vol'
                # def compare_read_dsb_bars(times:int = 100):
                #     t2 = datetime.datetime.now()
                #     num_bars = 0
                #     for i in range(times):
                #         ret = dtHelper.read_dsb_bars("./CFFEX.IF.HOT_m5.dsb")
                #         num_bars = len(ret)
                #     t3 = datetime.datetime.now()
                #     elapse = (t3-t2).total_seconds()*1000.0
                #     print(ret.to_df())
                #     print(f"read_dsb_bars {num_bars} bars for {times} times: {elapse:.2f}ms totally, {elapse/times:.2f}ms per reading")
            if rules_violated_tomonth == 0: # merge day file to month file
                # log(log_path, f'merging month data: {asset}, {year}.{month}')
                folder = f"{cfg.BAR_DIR}/m1/{asset}/"
                files_to_remove = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.dsb') and f'{year}.{month}' in file]
                for file in files_to_remove:
                    try:
                        os.remove(file)
                    except:
                        pass
                file_path = f"{cfg.BAR_DIR}/m1/{asset}/{year}.{month}.dsb"
                store_bars(group_month, file_path) # 'date' 'time' 'open' 'high' 'low' 'close' 'vol'
    return [days_passed, asset]

def process_dataframe(file_path, asset_dict, daily_k_bar):
    def process_juejin():
        df = pd.read_csv(file_path, header=None, skiprows=1, encoding='utf-8', on_bad_lines='warn')
        df.columns = ['trade_time', 'open', 'high', 'low', 'close', 'volume', 'amount']
        datetime = pd.to_datetime(df['trade_time'], format='%Y-%m-%d %H:%M:%S', errors='raise') # '%Y-%m-%d %H:%M:%S.%f'
        df['date'] = datetime
        df['year'] = datetime.dt.year
        df['month'] = datetime.dt.month
        df['day'] = datetime.dt.day
        df['time'] = datetime.dt.time
        df = df.drop(['trade_time', 'amount'], axis=1)
        return df
    def process_tdx():
        df = pd.read_csv(file_path, header=None, skiprows=2, encoding='gbk', on_bad_lines='warn')
        df = df[:-1] # remove last row declaring data_source
        df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume', 'amount']
        datetime = df['date'].astype(str) + ' ' + df['time'].astype(int).astype(str)
        datetime = pd.to_datetime(datetime, format='%Y/%m/%d %H%M', errors='raise') # '%Y-%m-%d %H:%M:%S.%f'
        df['datetime'] = datetime
        df['year'] = datetime.dt.year
        df['month'] = datetime.dt.month
        df['day'] = datetime.dt.day
        df['time'] = datetime.dt.time
        df = df.drop(['datetime', 'amount'], axis=1)
        return df
    def process_other():
        df = pd.read_csv(file_path, encoding='utf-8', header=0, on_bad_lines='warn')
        df.columns = ['index', 'datetime', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
        datetime = pd.to_datetime(df['datetime'], format='mixed', errors='raise') # '%Y-%m-%d %H:%M:%S.%f'
        df['datetime'] = datetime
        df['year'] = datetime.dt.year
        df['month'] = datetime.dt.month
        df['day'] = datetime.dt.day
        df['time'] = datetime.dt.time
        df = df.drop(['index', 'datetime', 'code', 'amount'], axis=1)
        return df
    return check_integrity_per_asset(process_juejin(), asset_dict, daily_k_bar)

def process_single_asset(asset_dict, meta):
    asset = list(asset_dict.keys())[0]
    asset_parts = asset.split('.')
    tdx_dict = {'sh': 'SH#', 'sz': 'SZ#'}
    asset_prefix = tdx_dict[asset_parts[0]]
    asset_code = asset_parts[1]
    data_valid = 1
    result = [[], asset] # [days_passed, asset]
    if cfg.CHECK_5:
        data_valid = 0
        file_path = f"{cfg.CROSS_VERIFY_DIR}/{asset_prefix + asset_code}.txt"
        if os.path.exists(file_path):
            daily_k_bar = pd.read_csv(file_path, header=None, skiprows=2, encoding='gbk')
            if daily_k_bar.shape[0] > 100:
                daily_k_bar = daily_k_bar[:-1] # remove last row declaring data_source
                daily_k_bar.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']
                datetime = pd.to_datetime(daily_k_bar['datetime'], format='%Y/%m/%d', errors='coerce')
                daily_k_bar['year'] = datetime.dt.year
                daily_k_bar['month'] = datetime.dt.month
                daily_k_bar['day'] = datetime.dt.day
                daily_k_bar = daily_k_bar.drop(['datetime', 'amount'], axis=1)
                data_valid = 1
    if data_valid:
        def load_flatten_files():
            file_name = tdx_dict[asset_parts[0]] + asset_parts[1] + '.txt'
            file = os.path.join(cfg.RAW_CSV_DIR, file_name)
            if os.path.exists(file):
                return process_dataframe(file, asset_dict, daily_k_bar)
            else:
                return [[], asset]
        def load_files_by_year():
            folders = [os.path.join(cfg.RAW_CSV_DIR, folder) for folder in os.listdir(cfg.RAW_CSV_DIR)] # if folder.endswith('年')]
            days_passed = []
            for folder in folders:
                file = [os.path.join(folder, file) for file in os.listdir(folder) if file == f'{asset_code}.{asset_prefix[:2]}.csv']
                if len(file) != 1:
                    if not file:
                        print(f'[WARN ][Loader_Once]: Missing {asset}: {folder}')
                    else:
                        print(f'[WARN ][Loader_Once]: Duplicates Found: {asset}: {folder}')
                else:
                    [days_passed_this_year,_] = process_dataframe(file[0], asset_dict, daily_k_bar)   
                    days_passed += days_passed_this_year
            return [days_passed, asset]
        return load_files_by_year()
    print(f'[WARN ][Loader_Once]: Missing {asset}: {file_path}')
    return result

def update_integrity_table(integrity_day_list, meta):
    [days_passed, asset] = integrity_day_list
    # self.integrity = pq.read_table(mkdir(cfg.INTEGRITY_FILE)).to_pandas()
    # index = pd.MultiIndex.from_tuples(itertools.product(code_list, date_list), names=['asset_code', 'date'])
    # self.integrity = pd.DataFrame({
    #     'integrity': pd.Series([False]*len(index), index=index, dtype=bool)
    #     })
    # pq.write_table(pa.Table.from_pandas(self.integrity), cfg.INTEGRITY_FILE)
    # integrity_table = f"{cfg.BAR_DIR}/m1/{asset}/integrity.parquet"
    # with open(path_str,type) as file:
    #     file.write(str + '\n')
    # #TODO

import multiprocessing
import threading
def process_slave_task(task, meta):
    try:
        return process_single_asset(task, meta)
    except KeyboardInterrupt:
        return [[], list(task.keys())[0]]
    # raise NotImplementedError("Subclass must implement process_slave_task method")
def process_master_task(task, meta):
    update_integrity_table(task, meta)
    # raise NotImplementedError("Subclass must implement process_master_task method")
class n_slave_1_master_queue:
    # ProcessPoolExecutor: CPU-bound tasks
    # ThreadPoolExecutor: I/O-bound tasks
    # slaves: processing many tasks independent of each other
    # master: processing slave products, have exclusive access to certain system resources
    def __init__(self, tqdm_total, max_workers=1, concurrency_mode='thread'):
        self.max_workers = min(multiprocessing.cpu_count(), max_workers)
        self.concurrency_mode = self.parallel_mode(concurrency_mode)
        self.slave_tasks_queue = multiprocessing.Queue()
        self.master_task_queue = multiprocessing.Queue()
        self.master_lock = threading.Lock() if self.concurrency_mode == 'thread' else multiprocessing.Lock()
        self.tqdm_cnt_with_lock = multiprocessing.Value('i', 0)  # 'i': signed integer
        self.tqdm_desc = multiprocessing.Array('c', 256)  # 'c': char
        self.pbar = tqdm(total=tqdm_total) # tqdm pbar is not 'pickleable', so create a shareable int
        self.pbar_stop_event = threading.Event()
        
    def update_pbar(self):
        while not self.pbar_stop_event.is_set():
            current_value = self.tqdm_cnt_with_lock.value
            current_desc = self.tqdm_desc.value.decode().rstrip('\x00') # type: ignore
            self.pbar.n = current_value
            self.pbar.set_description(f"Processing {current_desc}")
            self.pbar.refresh()
            if current_value >= self.pbar.total:
                break
            time.sleep(0.1)  # Update every 0.1 seconds
            
    # N-gets corresponds to N-puts
    def add_slave_task(self, tasks):
        for task in tasks:
            self.slave_tasks_queue.put(task)
    def add_master_task(self, tasks):
        for task in tasks:
            self.master_task_queue.put(task)
            
    # mutable types (like lists, dictionaries, and other objects)
    # can be modified in place (like pointers)
    @staticmethod # use global method or static method in a class
    def worker(slave_tasks_queue,
               master_tasks_queue,
               master_lock,
               tqdm_cnt,
               tqdm_desc,
               meta,
               ):
        global terminate
        if 1:
            while not terminate:
                # print('worker starts')
                if not master_tasks_queue.empty() and not terminate:
                    master_task = master_tasks_queue.get()
                    time.sleep(0.1)  # Prevent busy waiting (only master tasks left)
                    with master_lock:
                        # print('worker acquired lock')
                        # print('worker processing master task')
                        process_master_task(master_task, meta)
                        # print('worker released lock')
                if not slave_tasks_queue.empty() and not terminate:
                    slave_task = slave_tasks_queue.get()
                    # print('worker processing slave task')
                    result = process_slave_task(slave_task, meta)
                    master_tasks_queue.put(result)
                    with tqdm_cnt.get_lock(): # avoid racing condition
                        tqdm_cnt.value += 1
                        tqdm_info = list(slave_task.keys())[0]
                        tqdm_desc.value = str(tqdm_info).encode()[:255]  # Ensure it fits in the array
                if master_tasks_queue.empty() and slave_tasks_queue.empty():
                    # Stop the process when None is received
                    # print('worker finished processing')
                    break
            # Add a small sleep to allow for interrupt checking
            time.sleep(0.1)
        # except Exception as e:
        #     print(f"Worker error: {e}")
        # finally:
        #     print("Worker exiting...")
    # def process_slave_task(self, task):
    #     raise NotImplementedError("Subclass must implement process_slave_task method")
    # def process_master_task(self, task):
    #     raise NotImplementedError("Subclass must implement process_master_task method")
    
    def parallel_mode(self, concurrency_mode):
        return concurrency_mode
    
    def execute(self):
        # for multiple processes: shared info (IO/function/class) needs to be 'pickled' first
        # use global function in worker process (easier to serialize than class)
        # use @staticmethod to mark function if is in a class
        meta = 0
        workers = [] # thread or process
        global terminate
        try:
            if self.concurrency_mode == 'process':
                for _ in range(self.max_workers):
                    w = multiprocessing.Process(
                        target=self.worker,
                        args=(
                            # shareable:
                            self.slave_tasks_queue,
                            self.master_task_queue,
                            self.master_lock,
                            self.tqdm_cnt_with_lock,
                            self.tqdm_desc,

                            # non-shareable(inout):
                            meta,
                            ))
                    w.start()
                    workers.append(w)
            elif self.concurrency_mode == 'thread':
                for _ in range(self.max_workers):
                    w = threading.Thread(
                        target=self.worker,
                        args=(
                            # shareable:
                            self.slave_tasks_queue,
                            self.master_task_queue,
                            self.master_lock,
                            self.tqdm_cnt_with_lock,
                            self.tqdm_desc,

                            # non-shareable:
                            meta,
                            ))
                    w.start()
                    workers.append(w)
            else:
                raise ValueError("Invalid concurrency mode")
        
            # Start a separate thread to update the progress bar
            self.pbar_updater = threading.Thread(target=self.update_pbar)
            self.pbar_updater.start()
            if terminate:
                print("Terminating workers...")
                for w in workers:
                    w.terminate()
                    # For threads, you might need a custom termination mechanism
            for w in workers:
                w.join()
                
        except Exception as e:
            print(f"Execution error: {e}")
        finally:
            terminate = True
            self.pbar_stop_event.set()
            if hasattr(self, 'pbar_updater'):
                self.pbar_updater.join()
            print("Execution completed or terminated.")
            
if __name__ == "__main__":
    import run_db_maintain
    run_db_maintain.database_maintenance()
    pass