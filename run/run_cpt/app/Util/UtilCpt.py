import os, sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Tuple

RED     = '\033[91m'
GREEN   = '\033[92m'
YELLOW  = '\033[93m'
BLUE    = '\033[94m'
PURPLE  = '\033[95m'
CYAN    = '\033[96m'
DEFAULT = '\033[0m'

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
        # # Moving the cursor to the beginning of the file to read from the beginning
        # file.seek(0)
        # data = file.read()
    if stdout:
        print(str)
    return str
    
from wtpy import WtBtEngine, EngineType, WtDtServo
from wtpy.monitor import WtBtSnooper
from wtpy.wrapper import WtDataHelper
from wtpy.apps import WtBtAnalyst
from wtpy.WtCoreDefs import WTSBarStruct
from wtpy.SessionMgr import SessionMgr

def testBtSnooper():
    pass

# ================================================
# CSV to DATABASE file (DSB indexed my month/day)
# ================================================
from config.cfg_cpt import cfg_cpt

dtHelper = WtDataHelper()

def generate_database_files(force_sync:bool=False):
    '''
    force-sync: would check DATABASE file for each csv \n
    non-force-sync(default): only check symbol name exist in DATABASE
    '''
    # print('DB(1m): Generating monthly/daily 1m db files...')
    print(f"SRC_CSV:          {GREEN}{cfg_cpt.CRYPTO_CSV_DIR} /<symbol>/1m/{DEFAULT}")
    print(f"DB_DSB:           {GREEN}{cfg_cpt.CRYPTO_DB_DIR} /<symbol>/1m/{DEFAULT}")
    
    assets = cfg_cpt.symbols
    unprocessed_asset_list = []
    processed_asset_list = os.listdir(mkdir(f"{cfg_cpt.CRYPTO_DB_DIR}/"))
    for asset in assets:
        if asset not in processed_asset_list or force_sync:
            unprocessed_asset_list.append(asset)
        else:
            pass
            # print(f'asset {asset} already processed')
    num_assets = len(unprocessed_asset_list)
    # print(f'DB(1m): num of assets to be processed: {num_assets}')
    
    # Replace your for loop with:
    if len(unprocessed_asset_list)!=0:
        process_all_assets(unprocessed_asset_list)
    
from multiprocessing import Pool
import multiprocessing as mp

def process_all_assets(unprocessed_asset_list):
    # Use number of CPU cores minus 1 to avoid overloading
    num_processes = max(1, mp.cpu_count() - 1)
    
    with Pool(num_processes) as pool:
        # Use tqdm to show progress
        results = list(tqdm(
            pool.imap(process_single_asset, unprocessed_asset_list),
            total=len(unprocessed_asset_list)
        ))
    return results

def process_single_asset(asset):
    src_folder = f"{cfg_cpt.CRYPTO_CSV_DIR}/{asset}/1m/"
    src_csvs = os.listdir(src_folder)
    db_folder = mkdir(f"{cfg_cpt.CRYPTO_DB_DIR}/{asset}/1m/")
    db_dsbs = os.listdir(db_folder)
    
    for src_csv in src_csvs:
        if not str(src_csv).endswith('csv'):
            continue
        year = src_csv.split("-")[2]
        month = src_csv.split("-")[3].split(".")[0]
        db_dsb = f'{year}.{month}.dsb'
        if db_dsb not in db_dsbs:
            src_csv_path = os.path.join(src_folder, src_csv)
            db_dsb_path = os.path.join(db_folder, db_dsb)
            process_dataframe(src_csv_path, db_dsb_path)
    return asset

def process_dataframe(csv_file_path, dsb_file_path):
    def process_Binance():
        # Binance csv starts from month xx: 8:01am
        # print(csv_file_path)
        df = pd.read_csv(csv_file_path, header=None, skiprows=1, encoding='utf-8', on_bad_lines='warn')
        # Open_time Open High Low Close Volume Close_time Quote_asset_volume Number_of_trades Taker_buy_base_asset_volume Taker_buy_quote_asset_volume Ignore
        df = df.iloc[:, :6]
        df.columns = ['time', 'open', 'high', 'low', 'close', 'vol']
        # datetime = pd.to_datetime(df['trade_time'], format='%Y-%m-%d %H:%M:%S', errors='raise') # '%Y-%m-%d %H:%M:%S.%f'
        # df['date'] = datetime
        # df['year'] = datetime.dt.year
        # df['month'] = datetime.dt.month
        # df['day'] = datetime.dt.day
        # df['time'] = datetime.dt.time
        return df
    store_bars(process_Binance(), dsb_file_path)
    # print(dtHelper.read_dsb_bars(dsb_file_path).to_df())
    
def store_bars(df, file_path): # 'date' 'time' 'open' 'high' 'low' 'close' 'vol'
    if os.path.exists(file_path):
        pass
    else:
        # '%Y-%m-%d %H:%M:%S.%f'
        datetime = pd.to_datetime(df['time'], unit='ms') # Unix Epoch time to datetime
        df['date'] = datetime.dt.strftime('%Y%m%d').astype('int64')
        df['time'] = datetime.dt.strftime('%H%M').astype('int')
        
        # wt-specific times
        df['time'] = (df['date'] - 19900000)*10000 + df['time']
        
        df = df[['date', 'time', 'open', 'high', 'low', 'close', 'vol']].reset_index(drop=True)
        BUFFER = WTSBarStruct*len(df)
        buffer = BUFFER()
        def assign(procession, buffer):
            tuple(map(lambda x: setattr(buffer[x[0]], procession.name, x[1]), enumerate(procession)))
        df.apply(assign, buffer=buffer)
        store_path = mkdir(file_path)
        dtHelper.store_bars(barFile=store_path, firstBar=buffer, count=len(df), period="m1")

# ================================================
# DATABASE file to MERGED DATABASE file
# ================================================
def generate_merged_database_files(resample_n:int=1, begin_date=datetime(1990,1,1), end_date=datetime(2050,1,1), total=True):
    ''' 
    generate temporary merged DB files for back-testing
    '''
    # print('Merge_DB(resample): Generating Merged db files...')
    print(f"MERGED_DB_DSB:    {GREEN}{cfg_cpt.WT_STORAGE_DIR}/his/min1/{cfg_cpt.market}/ <symbol>.dsb{DEFAULT}")
    print(f"RESAMPLED_DB_DSB: {GREEN}{cfg_cpt.WT_STORAGE_DIR}/his/{'min'+str(resample_n)}/{cfg_cpt.market}/ <symbol>.dsb{DEFAULT}")
    
    assets = cfg_cpt.symbols
    for asset in tqdm(assets, desc=f'Merging and Resampling(x{resample_n})...'):
        database_db_folder  = f"{cfg_cpt.CRYPTO_DB_DIR}/{asset}/1m/"
        merged_db_path      = f'{cfg_cpt.WT_STORAGE_DIR}/his/min1/{cfg_cpt.market}/{asset}.dsb'
        resampled_db_path   = f'{cfg_cpt.WT_STORAGE_DIR}/his/{'min'+str(resample_n)}/{cfg_cpt.market}/{asset}.dsb'
        try:
            combine_dsb_1m(asset, database_db_folder, merged_db_path)
            # print(dtHelper.read_dsb_bars(merged_db_path).to_df())

        except Exception as e:
            print(f'Err processing: {asset}')
            print(e)
            continue
        if resample_n != 1:
            resample(merged_db_path, resample_n, resampled_db_path)

def combine_dsb_1m(asset, database_db_folder, merged_db_path, begin_date=datetime(1990,1,1), end_date=datetime(2050,1,1), total=True):
    df = []
    if not os.path.exists(merged_db_path):
        if total: # read ALL dsb file and return DF
            sorted_file_list = sort_files_by_date(database_db_folder)
        else: # only read SOME data and return DF(do not combine all dsb)
            sorted_file_list = sort_files_by_date(database_db_folder, begin_date, end_date)
        for file in tqdm(sorted_file_list, desc=f'{asset}'):
            file_path = os.path.join(database_db_folder, file)
            df.append(dtHelper.read_dsb_bars(file_path).to_df())
        df = pd.concat(df, ignore_index=False)
        print(df)
        wt_df_2_dsb(df, mkdir(merged_db_path))

def sort_files_by_date(folder_path, start_date=datetime(1900,1,1), end_date=datetime(2050,1,1)):
    files = [f for f in os.listdir(folder_path) if f.endswith('.dsb')]
    files_with_dates = []
    for file in files:
        file_date = get_file_date(file)
        if start_date < file_date < end_date:
            files_with_dates.append((file, file_date))
    # Sort the files by their date
    sorted_files = sorted(files_with_dates, key=lambda x: x[1])
    return [file for file, _ in sorted_files]
    
def get_file_date(filename):
    from datetime import datetime
    base_name = filename[:-4]  # Remove last 4 characters (".dsb")
    parts = base_name.split('.')
    if len(parts) == 3:
        year, month, day = parts
        return datetime(int(year), int(month), int(day))
    elif len(parts) == 2:
        year, month = parts
        return datetime(int(year), int(month), 1)  # Treat as the first day of the month
    else:
        return datetime(0000, 00 ,00)  # Invalid format

def resample(src_path, times, store_path):
    if not os.path.exists(store_path):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        sessMgr = SessionMgr()
        sessMgr.load(f"{script_dir}/../config/cpt_sessions.json")
        sInfo = sessMgr.getSession("ALLDAY")
        df = dtHelper.resample_bars(
            barFile=src_path,
            period='m1',
            times=times,
            fromTime=200001010931,
            endTime=205001010931,
            sessInfo=sInfo,
            alignSection=False).to_df()
        # time: day: yyyymmdd，min: yyyymmddHHMMSS
        wt_df_2_dsb(df, mkdir(store_path))
        # print(dtHelper.read_dsb_bars(src_path).to_df())
        # print(df)

def wt_df_2_dsb(df, store_path):
    # index: open high low close settle turnover volume open_interest diff bartime
    # 'date' 'time' 'open' 'high' 'low' 'close' 'vol'
    df = df.rename(columns={'volume': 'vol'})
    df['date'] = df['bartime'].astype(str).str[:8].astype(int)
    df['time'] = df['bartime']-199000000000
    df = df[['date', 'time', 'open', 'high', 'low', 'close', 'vol']].reset_index(drop=True)
    BUFFER = WTSBarStruct*len(df)
    buffer = BUFFER()
    def assign(procession, buffer):
        tuple(map(lambda x: setattr(buffer[x[0]], procession.name, x[1]), enumerate(procession)))
    df.apply(assign, buffer=buffer)
    dtHelper.store_bars(barFile=store_path, firstBar=buffer, count=len(df), period="m1")

# ================================================
# generate asset list
# ================================================
def generate_asset_list():
    symbols = cfg_cpt.symbols
    
    if not os.path.exists(cfg_cpt.ASSET_FILE):
        output = {
            "Binance": {}
        }
        # Populate the data for each symbol
        for symbol in symbols:
            output["Binance"][symbol] = {
                "code": symbol,
                "exchg": "Binance",
                "name": symbol,
                "product": "UM"
            }
        # Write to JSON file
        with open(cfg_cpt.ASSET_FILE, 'w') as f:
            json.dump(output, f, indent=4)

    with open(cfg_cpt.ASSET_FILE, 'r', encoding='gbk', errors='ignore') as file:
        asstes = json.load(file)
        
    wt_assets = []
    exchange = 'Binance'
    for symbol in symbols:
        wt_assets.append(f'{exchange}.{asstes[exchange][symbol]['product']}.{symbol}')
    return wt_assets

# ================================================
# Others
# ================================================

def enable_logging():
    import logging
    '''
    This needs to be set up before init(import) of packages(with logging)
    '''
    log_file=mkdir('logs/wtcpp.log')
    with open(log_file, 'w', encoding='utf-8') as f:
        # f.write('@charset "gbk";\n')  # CSS-style encoding declaration
        f.write('/* @encoding=gbk */\n')  # Comment-style declaration
    logging.basicConfig(
        filename=log_file,      # Specify the output file
        filemode='a',           # 'w' to overwrite, 'a' to append
        level=logging.NOTSET,   # Capture all levels
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'    # Only output the message without timestamps etc
    )
