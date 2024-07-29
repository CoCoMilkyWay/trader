import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
import pyarrow as pa
import pyarrow.parquet as pq
from run_bulk_data_import import cfg

from wtpy.apps.datahelper import DHFactory as DHF

class mischelper:
    def __init__(self):
        self.hlper = DHF.createHelper("baostock")
        self.hlper.auth()

    # Assuming the structure is deeply nested, you might need to normalize it:
    def normalize_data(self, data, meta):
        """ Flatten JSON structure """
        return pd.DataFrame({meta: data[meta]})
    
    def load_json(self, file_path):
        import json
        with open(file_path, 'r', encoding='gbk', errors='ignore') as file:  # Note the 'utf-8-sig' which handles BOM if present
            return json.load(file)

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
            self.hlper.dmpHolidayssToFile(filename='holidays.json')
            previous_json = self.load_json(cfg.HOLIDAYS_FILE)
            current_json = self.load_json('holidays.json')
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
        except:
            print('[ERROR][maintain_D_Holidays]: cannot connect to akshare to update holidays')

    def update_assetlist(self):
        import json
        import shutil
        import pandas as pd
        try:
            self.hlper.dmpCodeListToFile(filename='stocks.json')
            previous_json = self.load_json(file_path=cfg.STOCKS_FILE)
            current_json = self.load_json(file_path='stocks.json')
            
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
            asset_list_json =  self.load_json(cfg.STOCKS_FILE)

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
            previous_json = self.load_json(file_path=cfg.ADJFACTORS_FILE)
            current_json = self.load_json(file_path='adjfactors.json')
            
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
    def parse_datetime(self, date_str):
        return pd.to_datetime(date_str, format='%Y/%m/%d %H:%M')
    
    def process_csv_file(self, file_path):
        df = pd.read_csv(file_path)
        df.columns = ['index', 'datetime', 'code', 'open',
                      'high', 'low', 'close', 'volume', 'amount']
        datetime = self.parse_datetime(df['datetime'])
        df['datetime'] = datetime
        df['date'] = datetime.dt.date
        return df
    
    def process_folder(self, folder, position):
        # Assuming each folder contains several CSV files
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
        all_data = []

        # Initialize tqdm progress bar for files within the folder with a unique position
        for file in tqdm(files, desc=f"Processing files in {os.path.basename(folder)}", position=position):
            df = pd.read_csv(file)  # Simplified process example
            all_data.append(df)

        # Combine all dataframes processed from this folder into one dataframe
        return pd.concat(all_data)

    import concurrent
    from concurrent.futures import ProcessPoolExecutor

    def import_csv_data(self):
        # List all directories to process
        folders = [os.path.join(cfg.DATA_DIR, folder) for folder in os.listdir(cfg.DATA_DIR) if folder.endswith('年')]

        with ProcessPoolExecutor() as executor:
            # Manual submission of tasks to the executor to control tqdm position
            futures = {executor.submit(self.process_folder, folder, idx): idx for idx, folder in enumerate(folders)}

            # Collect results as they complete
            all_data = []
            for future in tqdm(self.concurrent.futures.as_completed(futures), total=len(futures), desc="Processing folders"):
                result = future.result()
                all_data.append(result)

        # Concatenate all DataFrame results
        full_data = pd.concat(all_data)
        table = pa.Table.from_pandas(full_data)

        # Write to a Parquet file
        pq.write_table(table, cfg.DB_FILE)
    
    def check_integrity(self, df):
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
    
    
    def get_metadata(self, df):
        metadata = {}
        for code, group in df.groupby('code'):
            metadata[code] = {
                # Placeholder, replace with actual name
                'asset_name': f"Asset_{code}",
                # Placeholder
                'yearly_data_integrity': {year: True for year in range(2000, 2032)},
                'start_date': group['datetime'].min().strftime('%Y-%m-%d'),
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
    