import os
import sys
from tokenize import String
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
import pyarrow as pa
import pyarrow.parquet as pq
import baostock as bs                    

from wtpy.apps.datahelper import DHFactory as DHF
from run_bulk_data_import import cfg

class datahelper:
    def load_json(file_path):
        import json
        with open(file_path, 'r', encoding='gbk', errors='ignore') as file:  # Note the 'utf-8-sig' which handles BOM if present
            return json.load(file)

class mischelper:
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
        if 1:
            self.hlper.dmpHolidayssToFile(filename_holidays='holidays.json', filename_tradedays='tradedays.json')
            previous_json = datahelper.load_json(cfg.HOLIDAYS_FILE)
            current_json = datahelper.load_json('holidays.json')
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
        #except:
        #    print('[ERROR][maintain_D_Holidays]: cannot connect to akshare to update holidays')

    def update_assetlist(self):
        import json
        import shutil
        import pandas as pd
        try:
            self.hlper.dmpCodeListToFile(filename='stocks.json')
            previous_json = datahelper.load_json(file_path=cfg.STOCKS_FILE)
            current_json = datahelper.load_json(file_path='stocks.json')

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
            asset_list_json =  datahelper.load_json(cfg.STOCKS_FILE)

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
            previous_json = datahelper.load_json(file_path=cfg.ADJFACTORS_FILE)
            current_json = datahelper.load_json(file_path='adjfactors.json')

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
        # parse meta data
        self.holidays = datahelper.load_json(cfg.HOLIDAYS_FILE)
        self.tradedays = datahelper.load_json(cfg.TRADEDAYS_FILE)
        self.stocks = datahelper.load_json(cfg.STOCKS_FILE)
        self.adjfactors = datahelper.load_json(cfg.ADJFACTORS_FILE)

        # self.lock_path = cfg.INTEGRITY_FILE + ".lock"  # Lock file path
        # self.lock_path = cfg.METADATA_FILE + ".lock"  # Lock file path
        
        # these tables are huge, use parquet DB
        try:
            print('[INFO ][maintain: Reading Meta-Data]')
            self.metadata = pq.read_table(cfg.METADATA_FILE).to_pandas()
            print('[INFO ][maintain: Finished Reading Meta-Data]')
        except FileNotFoundError:
            print('[ERROR][maintain: Creating Meta-Data Table]')
            self.hlper = DHF.createHelper("baostock")
            self.hlper.auth()
            code_list = [list(self.stocks['SSE'].keys()) + list(self.stocks['SZSE'].keys())]
            self.metadata = pd.DataFrame({
                'asset_name': pd.Series(dtype='string'),
                'yearly_data_integrity': pd.Series(dtype='Int32'),
                'ipoDate': pd.Series(dtype='datetime64[ns]'),
                'outDate': pd.Series(dtype='datetime64[ns]'),
                'type': pd.Series(dtype='Int8'),
                'status': pd.Series(dtype='bool'),
                'exchange': pd.Series(dtype='string'),
                'industry_sector_level_1': pd.Series(dtype='string'),
                'industry_sector_level_2': pd.Series(dtype='string'),
                'reserved': pd.Series(dtype='string')
            }, index=code_list)
            for code in code_list:
                self.update_metadata_by_code(code)
            pq.write_table(pa.Table.from_pandas(self.metadata), cfg.METADATA_FILE)
        try:
            print('[INFO ][maintain: Reading Integrity Table]')
            self.integrity = pq.read_table(cfg.INTEGRITY_FILE).to_pandas()
            print('[INFO ][maintain: Finished Reading Integrity Table]')
        except FileNotFoundError:
            print('[ERROR][maintain: Creating Integrity Table]')
            import itertools
            from datetime import datetime
            code_list = list(self.stocks['SSE'].keys()) + list(self.stocks['SZSE'].keys())
            date_list = date_objects = [datetime.strptime(date, "%Y%m%d").date() for date in self.tradedays['CHINA']]
            index = pd.MultiIndex.from_tuples(itertools.product(code_list, date_list), names=['asset_code', 'date'])
            self.integrity = pd.DataFrame({
                'integrity': pd.Series([False]*len(index), index=index, dtype=bool)
                })
            pq.write_table(pa.Table.from_pandas(self.integrity), cfg.INTEGRITY_FILE)
            print('[INFO ][maintain: Created Integrity Table]')
            print(self.integrity)
    def update_metadata_by_code(self, code_str):
        rs = bs.query_stock_basic(code=code_str)
        print(rs)
        return rs
    def get_metadata(self, df):
        metadata = {}
        for code, group in df.groupby('code'):
            metadata[code] = {
                # Placeholder, replace with actual name
                'asset_name': f"Asset_{code}",
                # Placeholder
                'yearly_data_integrity': {year: True for year in range(2000, 2032)},
                'ipoDate': group['datetime'].min().strftime('%Y-%m-%d'),
                'end_date': group['datetime'].max().strftime('%Y-%m-%d'),
                'first_traded': group['datetime'].min().strftime('%Y-%m-%d'),
                'auto_close_date': (group['datetime'].max() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'exchange': 'SH' if code.startswith('SH') else 'SZ',
                'sub_exchange': self.get_sub_exchange(code),
                'industry_sector': {
                    'level_1': 'Placeholder',  # Replace with actual data
                    'level_2': 'Placeholder'
                },
                'reserved': ''
            }
        return metadata
    #        # Convert column_0 explicitly to uint32 if it's not automatically inferred
    #        df['column_0'] = df['column_0'].astype('uint32')

    #        init_metadata_per_asset = {
    #            'asset_code': [0],
    #            'asset_name': [0],
    #            'yearly_data_integrity': [[0] * 32],
    #            'start_date': pd.to_datetime(['2000-01-01']),
    #            'end_date': pd.to_datetime(['2000-01-01']),
    #            'first_traded': pd.to_datetime(['2000-01-01']),
    #            'auto_close_date': pd.to_datetime(['2000-01-01']),
    #            'exchange': [],
    #            'industry_sector_level_1': [101, 102, 103, 104, 105, 106, 107],  # Example industry sectors
    #            'industry_sector_level_2': [201, 202, 203, 204, 205, 206, 207],
    #            'reserved': ['info', 'info', 'info', 'info', 'info', 'info', 'info']
    #        }
    #        df = pd.DataFrame(meta_data)
    #        df['sub_exchange'] = df['asset_code'].apply(determine_sub_exchange)
    #        table = pa.Table.from_pandas(full_data)
    #        pq.write_table(table, DB_FILE)

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

    # # Step 2: Perform integrity checks
    # df = pq.read_table(cfg.DB_FILE).to_pandas()
    # integrity_table = dhpr.check_integrity(df)
    # with open(cfg.INTEGRITY_FILE, 'w') as f:
    #     json.dump(integrity_table, f)
    #     #
    # # Step 3: Generate metadata
    # metadata_table = dhpr.get_metadata(df)
    # with open(cfg.METADATA_FILE, 'w') as f:
    #     json.dump(metadata_table, f)

    def process_csv_file(self, file_path): # each thread processes an asset's yearly minute bar
        df = pd.read_csv(file_path)
        df.columns =    ['index', 'datetime', 'code', 'open',
                        'high', 'low', 'close', 'volume', 'amount']
        # fmts = ['%Y/%m/%d %H:%M', '%Y-%m-%d %H:%M']
        #datetime = self.parse_df_datetime(df['datetime'], file_path)
        datetime = pd.to_datetime(df['datetime'], format='mixed', errors='coerce') # error: fill NaT
        df['datetime'] = datetime
        df['date'] = datetime.dt.date
        df['year'] = datetime.dt.year
        self.check_integrity_per_day_per_asset()
        return df

    def process_single_asset_from_single_folder(self, folder, folder_idx, folder_item):
        # Assuming each folder contains several CSV files
        file = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv') and folder_item in f]
        if len(file) != 1:
            if file == []:
                sys.exit(f'[ERROR][Loader_Once]: Exiting: Missing {folder_item} {folder}')
            else:
                sys.exit(f'[ERROR][Loader_Once]: Exiting: Duplicates Found: {folder_item} {folder}')
        return self.process_csv_file(file[0])

    def process_assets_from_folders(self):
        import concurrent
        from concurrent.futures import ProcessPoolExecutor
        # List all directories to process
        folders = [os.path.join(cfg.DATA_DIR, folder) for folder in os.listdir(cfg.DATA_DIR) if folder.endswith('年')]
        # assets = self.stocks
        assets = {
            'SSE':
                {
                    '000001': {'code': '000001', 'exchg': 'SSE', 'name': '指数001', 'product': 'IDX'},
                    '000002': {'code': '000002', 'exchg': 'SSE', 'name': '指数002', 'product': 'IDX'}
                }
            }
        exchanges = ['SSE', 'SZSE']
        num_assets = 0
        for exchange in exchanges:
            try:
                assets[exchange] # check if exchange exists
                num_assets += len(assets[exchange])
            except:
                exchanges.remove(exchange)

        pbar = tqdm(total=num_assets)
        for exchange in exchanges:
            for folder_item in assets[exchange]:
                pbar.set_description(f"Processing bulk asset {folder_item}")
                with ProcessPoolExecutor() as executor:
                    # Manual submission of tasks to the executor to control tqdm position
                    futures = {executor.submit(self.process_single_asset_from_single_folder, folder, folder_idx, folder_item): folder_idx for folder_idx, folder in enumerate(folders)}

                    # Collect results as they complete
                    asset_all_data = []
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        asset_all_data.append(result)

        # # Concatenate all DataFrame results
        # full_data = pd.concat(all_data)
        # table = pa.Table.from_pandas(full_data)
#
        # # Write to a Parquet file
        # pq.write_table(table, cfg.DB_FILE)

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


    def get_sub_exchange(self, code):
        if code.startswith('SH'):
            if code[3:5] == '60':
                return 'SSE.A'
            elif code[3:6] == '900':
                return 'SSE.B'
            elif code[3:5] == '68':
                return 'SSE.STAR'
        elif code.startswith('SZ'):
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
