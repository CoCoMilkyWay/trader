from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime
import time
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import baostock as bs                    

from wtpy.apps.datahelper import DHFactory as DHF
from zmq import Errno
from run_bulk_data_import import cfg

class datahelper:
    def load_json(self, file_path):
        import json
        with open(file_path, 'r', encoding='gbk', errors='ignore') as file:  # Note the 'utf-8-sig' which handles BOM if present
            return json.load(file)
    def dump_json(self, file_path, df):
        import json
        with open(file_path, 'w', encoding='gbk', errors='ignore') as file:
            df.to_json(file, orient='records', force_ascii=False, indent=4)
        
class mischelper:
    def __init__(self):
        self.hlper = DHF.createHelper("baostock")
        self.hlper.auth()
        self.dhlper = datahelper()

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
            previous_json = self.dhlper.load_json(file_path=cfg.HOLIDAYS_FILE)
            current_json = self.dhlper.load_json(file_path='holidays.json')
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
        import json
        import shutil
        import pandas as pd
        try:
            self.hlper.dmpCodeListToFile(filename='stocks.json')
            previous_json = self.dhlper.load_json(file_path=cfg.STOCKS_FILE)
            current_json = self.dhlper.load_json(file_path='stocks.json')

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
        import json
        import shutil
        import pandas as pd
        try:
            asset_list_json =  self.dhlper.load_json(cfg.STOCKS_FILE)

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
            previous_json = self.dhlper.load_json(file_path=cfg.ADJFACTORS_FILE)
            current_json = self.dhlper.load_json(file_path='adjfactors.json')

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
        print('[INFO ][maintain: Initializing DataBase]')
        self.dhlper = datahelper()
        # parse meta data
        self.holidays = self.dhlper.load_json(cfg.HOLIDAYS_FILE)
        self.tradedays = self.dhlper.load_json(cfg.TRADEDAYS_FILE)
        self.stocks = self.dhlper.load_json(cfg.STOCKS_FILE)
        if cfg.TEST:
            for e in ['SSE', 'SZSE']:
                limited_dict = {}
                count = 0
                for key, value in self.stocks[e].items():
                    limited_dict[key] = value
                    count += 1
                    if count == 10:
                        self.stocks[e] = limited_dict
                        break  # Stop after adding 10 items
            print(self.stocks)
        self.adjfactors = self.dhlper.load_json(cfg.ADJFACTORS_FILE)

        # self.lock_path = cfg.INTEGRITY_FILE + ".lock"  # Lock file path
        # self.lock_path = cfg.METADATA_FILE + ".lock"  # Lock file path
        
        # these tables are huge, use parquet DB
        try:
            print('[INFO ][maintain: Reading Integrity Table]')
            self.integrity = pq.read_table(cfg.INTEGRITY_FILE).to_pandas()
            print('[INFO ][maintain: Finished Reading Integrity Table]')
        except FileNotFoundError:
            print('[ERROR][maintain: Creating Integrity Table]')
            import itertools
            code_list = list(self.stocks['SSE'].keys()) + list(self.stocks['SZSE'].keys())
            date_list = date_objects = [datetime.strptime(date, "%Y%m%d").date() for date in self.tradedays['CHINA']]
            index = pd.MultiIndex.from_tuples(itertools.product(code_list, date_list), names=['asset_code', 'date'])
            self.integrity = pd.DataFrame({
                'integrity': pd.Series([False]*len(index), index=index, dtype=bool)
                })
            pq.write_table(pa.Table.from_pandas(self.integrity), cfg.INTEGRITY_FILE)
            print('[INFO ][maintain: Created Integrity Table]')
            print(self.integrity)

        try:
            print('[INFO ][maintain: Reading Meta-Data]')
            self.metadata = pq.read_table(cfg.METADATA_FILE).to_pandas()
            print('[INFO ][maintain: Finished Reading Meta-Data]')
        except FileNotFoundError:
            print('[ERROR][maintain: Creating Meta-Data Table]')
            self.hlper = DHF.createHelper("baostock")
            self.hlper.auth()
            code_list = ['sh.' + code for code in list(self.stocks['SSE'].keys())] + ['sz.' + code for code in list(self.stocks['SZSE'].keys())]
            self.metadata = pd.DataFrame({
                'code': pd.Series(dtype='string'),
                'asset_name': pd.Series(dtype='string'),
                'yearly_data_integrity': pd.Series(dtype='Int32'),
                'ipoDate': pd.Series(dtype='datetime64[ns]'),
                'outDate': pd.Series(dtype='datetime64[ns]'),
                'type': pd.Series(dtype='Int8'), # 1.stk 2.idx 3.others 4.convertable bond 5.etf
                'status': pd.Series(dtype='bool'), # 0.quit 1.active
                'exchange': pd.Series(dtype='string'),
                'industry_sector_level_1': pd.Series(dtype='string'),
                'industry_sector_level_2': pd.Series(dtype='string'),
                'reserved': pd.Series(dtype='string')
            }, index=code_list)
            with tqdm(code_list, total=len(code_list)) as pbar:
                for code in pbar:
                    self.update_metadata_by_code(code)
                    # print(self.metadata.loc[code])
                    pbar.set_description(f'Querying {code}: ')
            pq.write_table(pa.Table.from_pandas(self.metadata), cfg.METADATA_FILE)
        # view only: meta table
        self.dhlper.dump_json(cfg.METADATA_JSON_FILE, self.metadata)
            
    def get_baostock_info(self, resultset):
        data_list = []
        while (resultset.error_code == '0') & resultset.next():
            data_list.append(resultset.get_row_data())
        return pd.DataFrame(data_list, columns=resultset.fields, index=[0]).iloc[0]
    
    def update_metadata_by_code(self, code_str):
        query_basic = bs.query_stock_basic(code=code_str)
        basic = self.get_baostock_info(query_basic)
        query_industry = bs.query_stock_industry(code=code_str) # ShenWan level-1
        industry = self.get_baostock_info(query_industry)
        self.metadata.loc[code_str, 'code'                   ] = code_str
        self.metadata.loc[code_str, 'asset_name'             ] = basic['code_name']
        self.metadata.loc[code_str, 'yearly_data_integrity'  ] = 0
        self.metadata.loc[code_str, 'ipoDate'                ] = np.datetime64(basic['ipoDate'])
        self.metadata.loc[code_str, 'outDate'                ] = np.datetime64(basic['outDate'])
        self.metadata.loc[code_str, 'type'                   ] = np.int8(basic['type']).item()
        self.metadata.loc[code_str, 'status'                 ] = np.bool(basic['status']).item()
        self.metadata.loc[code_str, 'exchange'               ] = self.get_sub_exchange(code_str)
        self.metadata.loc[code_str, 'industry_sector_level_1'] = industry['industry']
        self.metadata.loc[code_str, 'industry_sector_level_2'] = ''
        self.metadata.loc[code_str, 'reserved'               ] = ''
        
    def get_sub_exchange(self, code):
        if code.startswith('sh'):
            if code[3:5] == '60':
                return 'SSE.A'
            elif code[3:6] == '900':
                return 'SSE.B'
            elif code[3:5] == '68':
                return 'SSE.STAR'
        elif code.startswith('sz'):
            if code[3:6] in ['000', '001']:
                return 'SZSE.A'
            elif code[3:6] == '200':
                return 'SZSE.B'
            elif code[3:6] in ['300', '301']:
                return 'SZSE.SB'
            elif code[3:6] in ['002', '003']:
                return 'SZSE.A'
        if code[3:6] in ['440', '430'] or code[3:5] in ['83', '87']:
            return 'NQ'
        return 'Unknown'
    
    # from filelock import Timeout, FileLock
    # def lock_read_dataframe(self):
    #     """Reads a DataFrame from a file with locking to ensure exclusive access."""
    #     lock = FileLock(self.lock_path, timeout=10)
    #     with lock:
    #         df = pd.read_csv(self.filepath)  # Assuming a CSV file for simplicity
    #         return df
    #
    # def lock_write_dataframe(self, df):
    #     """Writes a DataFrame to a file with locking to ensure exclusive access."""
    #     lock = FileLock(self.lock_path, timeout=10)
    #     with lock:
    #         df.to_csv(self.filepath, index=False)

    #def parse_line_datetime_mixed_formats(self, datetime_str_line, file_path):
    #    for fmt in self.fmts:
    #        try:
    #            return pd.to_datetime(datetime_str_line, format=fmt)
    #        except ValueError:
    #            continue
    #    print('[ERROR][Loader_Once]: ', file_path)
    #    print(f'[ERROR][Loader_Once]: {datetime_str_line}: no valid date format found')
    #    return []
    #
    #def parse_df_datetime(self, datetime_str_df, file_path):
    #
    #    for fmt in self.fmts:
    #        try:
    #            df = pd.to_datetime(datetime_str_df, format=fmt)
    #            return df
    #        except:
    #            pass
    #    # try parse each line
    #    df = datetime_str_df.apply(lambda datetime_str_line:
    #        self.parse_line_datetime_mixed_formats(datetime_str_line, file_path))
    #    return []


    def main_process_assets_from_folders(self):
        assets = self.stocks
        # assets = {
        #     'SSE':
        #         {
        #             '000001': {'code': '000001', 'exchg': 'SSE', 'name': '指数001', 'product': 'IDX'},
        #             '000002': {'code': '000002', 'exchg': 'SSE', 'name': '指数002', 'product': 'IDX'}
        #         }
        #     }
        exchanges = ['SSE', 'SZSE']
        asset_list = []
        for exchange in exchanges:
            try:
                asset_list += assets[exchange] # check if exchange exists
            except:
                exchanges.remove(exchange)
        num_assets = len(asset_list)

    def check_integrity_per_day_per_asset(self, df):
        integrity = {}
        for (date, code), group in df.groupby(['date', 'code']):
            key = f"{date}_{code}"
            rules_violated = []

            # Rule 0: Non-zero OHLC
            if (group[['open', 'high', 'low', 'close']] == 0).any().any():
                rules_violated.append(0)

            # Rule 1: Timestamp continuity/order/completeness
            expected_times = pd.date_range(
                start=f"{date} 09:30", end=f"{date} 14:57", freq='1T')
            expected_times = expected_times[~((expected_times.hour == 11) & (expected_times.minute > 30)) &
                                            ~((expected_times.hour == 12)) &
                                            ~((expected_times.hour == 13) & (expected_times.minute == 0))]
            if not group['datetime'].tolist() == expected_times.tolist():
                rules_violated.append(1)

            # Rule 2: Intra-day price continuity
            if (group['high'] < group['low']).any() or (group['open'] > group['high']).any() or (group['open'] < group['low']).any() or (group['close'] > group['high']).any() or (group['close'] < group['low']).any():
                rules_violated.append(2)

            # Rule 3: Inter-day price jump limit
            if len(group) > 0:
                prev_close = df[(df['date'] == date - timedelta(days=1))
                                & (df['code'] == code)]['close'].iloc[-1]
                if abs(group['open'].iloc[0] - prev_close) / prev_close > 0.1:
                    rules_violated.append(3)

            # Rule 4: OHLC differ if volume is non-zero
            if ((group['volume'] > 0) & (group['open'] == group['high']) & (group['high'] == group['low']) & (group['low'] == group['close'])).any():
                rules_violated.append(4)

            integrity[key] = {
                'integrity_flag': len(rules_violated) == 0,
                'rules_violated': rules_violated
            }

        return integrity
    
import multiprocessing
import threading

class n_slave_1_master_queue:
    def __init__(self, tqdm_total, max_workers=cfg.no_workers, concurrency_mode=None):
        self.max_workers = max_workers
        self.concurrency_mode = self.parallel_mode()
        self.slave_tasks_queue = multiprocessing.Queue()
        self.master_task_queue = multiprocessing.Queue()
        self.master_lock = threading.Lock() if self.concurrency_mode == 'thread' else multiprocessing.Lock()
        self.pbar = tqdm(total=tqdm_total)

    def add_slave_task(self, tasks):
            self.slave_tasks_queue.put(tasks)

    def add_master_task(self, tasks):
            self.master_task_queue.put(tasks)

    def slave_worker(self, tasks):
        try:
            result = self.process_slave_task(tasks.get())
            self.master_task_queue.put(result)
            self.pbar.update(1)
            return None
        except Exception as e:
            print(f"Slave task generated an exception: {e}")
            return None

    def master_worker(self, tasks):
        while not self.master_task_queue.empty():
            self.process_master_task(tasks.get())

    def worker(self, slave_tasks_queue, master_tasks_queue, master_lock):
        if not self.master_task_queue.empty():
            with master_lock:
                self.master_worker(master_tasks_queue)
        else:
            self.slave_worker(slave_tasks_queue)

    def process_slave_task(self, task):
        raise NotImplementedError("Subclass must implement process_slave_task method")

    def process_master_task(self, task):
        raise NotImplementedError("Subclass must implement process_master_task method")

    def parallel_mode(self):
        return cfg.parallel_mode

    def execute(self):
        workers = []
        if self.concurrency_mode == 'process':
            for _ in range(self.max_workers):
                p = multiprocessing.Process(target=self.slave_worker, args=(self.slave_tasks_queue, self.master_task_queue, self.master_lock))
                p.start()
                workers.append(p)
        elif self.concurrency_mode == 'thread':
            for _ in range(self.max_workers):
                t = threading.Thread(target=self.slave_worker, args=(self.slave_tasks_queue, self.master_task_queue, self.master_lock))
                t.start()
                workers.append(t)
        else:
            raise ValueError("Invalid concurrency mode")

        for worker in workers:
            worker.join()
        if self.pbar:
            self.pbar.close()

class ExecutorDataImport(n_slave_1_master_queue):
    def __init__(self, tqdm_total):
        super().__init__(tqdm_total=tqdm_total)
        self.folders = [os.path.join(cfg.DATA_DIR, folder) for folder in os.listdir(cfg.DATA_DIR) if folder.endswith('年')]

    def process_slave_task(self, task):
        print(task)
        return None # self.process_single_asset_from_folders(task)

    def process_master_task(self, task):
        print(task)
        return None

    def process_csv_file(self, file_path):
        df = pd.read_csv(file_path)
        df.columns = ['index', 'datetime', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
        datetime = pd.to_datetime(df['datetime'], format='mixed', errors='coerce')
        df['datetime'] = datetime
        df['date'] = datetime.dt.date
        df['year'] = datetime.dt.year
        return df

    def process_single_asset_from_folders(self, asset):
        results = []
        for folder in self.folders:
            file = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv') and asset in f]
            if len(file) != 1:
                if not file:
                    print(f'[WARN ][Loader_Once]: Missing {asset} {folder}')
                else:
                    print(f'[WARN ][Loader_Once]: Duplicates Found: {asset} {folder}')
            else:
                results.append(self.process_csv_file(file[0]))
        self.pbar.set_description(f"Processing Assets {asset}")
        return {asset: results}

if __name__ == "__main__":
    num_assets = 20  # Set this according to your requirements
    task_queue = ExecutorDataImport(tqdm_total=num_assets)
    asset_list = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']
    task_queue.add_slave_task(asset_list)
    task_queue.execute()


#class executor_data_import(n_slave_1_master_queue):
#    def __init__(self, tasks, tqdm_total):
#        super().__init__(tasks=tasks, tqdm_total=tqdm_total)
#        self.folders = [os.path.join(cfg.DATA_DIR, folder) for folder in os.listdir(cfg.DATA_DIR) if folder.endswith('年')]
#
#    def process_slave_task(self, task):
#        print(task)
#        #return self.process_single_asset_from_folders(task)
#    
#    def process_master_task(self, result):
#        # Process the result here
#        # For example, you could print it or save it to a file
#        print(result)
#
#    def process_csv_file(self, file_path):
#        try:
#            df = pd.read_csv(file_path)
#            df.columns = ['index', 'datetime', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
#            datetime = pd.to_datetime(df['datetime'], format='mixed', errors='coerce')
#            df['datetime'] = datetime
#            df['date'] = datetime.dt.date
#            df['year'] = datetime.dt.year
#            return df.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries
#        except Exception as e:
#            print(f"Error processing file {file_path}: {e}")
#            return None
#
#    def process_single_asset_from_folders(self, asset):
#        results = []
#        for folder in self.folders:
#            file = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv') and asset in f]
#            if len(file) != 1:
#                if file == []:
#                    print(f'[WARN ][Loader_Once]: Missing {asset} {folder}')
#                else:
#                    print(f'[WARN ][Loader_Once]: Duplicates Found: {asset} {folder}')
#            else:
#                result = self.process_csv_file(file[0])
#                if result is not None:
#                    results.append(result)
#        # return {asset: results}

'''
import multiprocessing
import threading
import concurrent.futures
import queue
import os
import pandas as pd
from tqdm import tqdm

class n_slave_1_master_queue:
    # ProcessPoolExecutor: CPU-bound tasks
    # ThreadPoolExecutor: I/O-bound tasks

    # slaves: processing many tasks independent of each other
    # master: processing slave products, have exclusive access to certain system resources
    def __init__(self, tqdm_total, max_workers=None, concurrency_mode=None):
        self.max_workers = max_workers or os.cpu_count()
        self.concurrency_mode = concurrency_mode or self.parallel_mode()
        self.slave_tasks_queue = multiprocessing.JoinableQueue()
        self.master_task_queue = multiprocessing.Queue()
        self.pbar = tqdm(total=tqdm_total)

    def add_slave_task(self, tasks):
        for task in tasks:
            self.slave_tasks_queue.put(task)

    def add_master_task(self, tasks):
        for task in tasks:
            self.master_task_queue.put(task)

    def slave_worker(self, task):
        try:
            result = self.process_slave_task(task)
            self.master_task_queue.put(result)
            self.slave_tasks_queue.task_done()
            if self.pbar:
                self.pbar.update(1)
            return result
        except Exception as e:
            print(f"Slave task generated an exception: {e}")
            return None

    def master_worker(self):
        while not self.master_task_queue.empty() or not self.all_slave_tasks_done():
            try:
                if not self.master_task_queue.empty():
                    task = self.master_task_queue.get_nowait()
                    self.process_master_task(task)
                time.sleep(0.1)  # Prevent busy waiting
            except queue.Empty:
                pass

    def all_slave_tasks_done(self):
        return self.slave_tasks_queue.empty()

    def worker(self, task):
        self.slave_worker(task)

    def process_slave_task(self, task):
        raise NotImplementedError("Subclass must implement process_slave_task method")

    def process_master_task(self, task):
        raise NotImplementedError("Subclass must implement process_master_task method")

    def parallel_mode(self):
        return "process"

    def execute(self):
        executor_class = concurrent.futures.ThreadPoolExecutor if self.concurrency_mode == 'thread' else concurrent.futures.ProcessPoolExecutor
        tasks = []
        while not self.slave_tasks_queue.empty():
            tasks.append(self.slave_tasks_queue.get())
        with executor_class(max_workers=self.max_workers) as executor:
            self.slave_futures = [executor.submit(self.worker, task) for task in tasks]
            master_thread = threading.Thread(target=self.master_worker)
            master_thread.start()
            for slave_future in concurrent.futures.as_completed(self.slave_futures):
                try:
                    result = slave_future.result()
                    if result is not None:
                        print(result)
                except Exception as e:
                    print(f"Task generated an exception: {e}")
            master_thread.join()
        if self.pbar:
            self.pbar.close()

class ExecutorDataImport(n_slave_1_master_queue):
    def __init__(self, tqdm_total):
        super().__init__(tqdm_total=tqdm_total)
        self.folders = [os.path.join(cfg.DATA_DIR, folder) for folder in os.listdir(cfg.DATA_DIR) if folder.endswith('年')]

    def process_slave_task(self, task):
        return self.process_single_asset_from_folders(task)

    def process_master_task(self, task):
        return None

    def process_csv_file(self, file_path): # each thread processes an asset's minute bar
        df = pd.read_csv(file_path)
        df.columns = ['index', 'datetime', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
        datetime = pd.to_datetime(df['datetime'], format='mixed', errors='coerce')  # error: fill NaT
        df['datetime'] = datetime
        df['date'] = datetime.dt.date
        df['year'] = datetime.dt.year
        return df

    def process_single_asset_from_folders(self, asset):
        for folder in self.folders:
            file = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv') and asset in f]
            if len(file) != 1:
                if not file:
                    print(f'[WARN ][Loader_Once]: Missing {asset} {folder}')
                else:
                    print(f'[WARN ][Loader_Once]: Duplicates Found: {asset} {folder}')
            self.process_csv_file(file[0])
        self.pbar.set_description(f"Processing Assets {asset}")
'''