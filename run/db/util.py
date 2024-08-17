import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

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
    
from wtpy import WtBtEngine, EngineType, WtDtServo
from wtpy.monitor import WtBtSnooper
from wtpy.wrapper import WtDataHelper
from wtpy.apps import WtBtAnalyst
from wtpy.WtCoreDefs import WTSBarStruct
from wtpy.SessionMgr import SessionMgr

def testBtSnooper():
    dtServo = WtDtServo()
    dtServo.setBasefiles(
        folder          = "cfg/",
        commfile        = "assets_cfg/stk_comms.json",
        contractfile    = "assets_list/stocks.json",
        holidayfile     = "misc/holidays.json",
        sessionfile     = "sessions/sessions.json",
        hotfile         = "assets_list/hots.json"
                         )
    dtServo.setStorage(
        path='storage',
        adjfactor='cfg/misc/adjfactors.json'
        )
    snooper = WtBtSnooper(dtServo)
    snooper.run_as_server(port=8081, host="0.0.0.0")

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

def wt_csv_2_dsb(dtHelper, df, store_path):
    # index: open high low close settle turnover volume open_interest diff bartime
    # 'date' 'time' 'open' 'high' 'low' 'close' 'vol'
    df = df.rename(columns={'volume': 'vol'})
    df['date'] = df['bartime'].astype(str).str[:7].astype(int)
    df['time'] = df['bartime']-199000000000
    df = df[['date', 'time', 'open', 'high', 'low', 'close', 'vol']].reset_index(drop=True)
    BUFFER = WTSBarStruct*len(df)
    buffer = BUFFER()
    def assign(procession, buffer):
        tuple(map(lambda x: setattr(buffer[x[0]], procession.name, x[1]), enumerate(procession)))
    df.apply(assign, buffer=buffer)
    dtHelper.store_bars(barFile=store_path, firstBar=buffer, count=len(df), period="m1")

def combine_dsb_1m(dtHelper, read_path, store_path, begin_date=datetime(1990,1,1), end_date=datetime(2050,1,1), store=True):
    if store: # read ALL dsb file and return DF
        df = []
        if not os.path.exists(store_path):
            # asset_dsb = [os.path.join(read_path, file) for file in os.listdir(read_path) if file.endswith('.dsb')]
            sorted_file_list = sort_files_by_date(read_path)
            for file in tqdm(sorted_file_list):
                file_path = os.path.join(read_path, file)
                df.append(dtHelper.read_dsb_bars(file_path).to_df())
            df = pd.concat(df, ignore_index=False)
            wt_csv_2_dsb(dtHelper, df, store_path)
        return df
    else: # only read SOME data and return DF(do not combine all dsb)
        sorted_file_list = sort_files_by_date(read_path, begin_date, end_date)
        df = []
        for file in tqdm(sorted_file_list):
            file_path = os.path.join(read_path, file)
            df.append(dtHelper.read_dsb_bars(file_path).to_df())
        return pd.concat(df, ignore_index=False)
        
        

def resample(dtHelper, src_path, times, store_path):
    if not os.path.exists(store_path):
        sessMgr = SessionMgr()
        sessMgr.load("cfg/sessions/sessions.json")
        sInfo = sessMgr.getSession("SD0930")
        df = dtHelper.resample_bars(src_path,'m1',times,200001010931,209901010931,sInfo, True).to_df()
        wt_csv_2_dsb(dtHelper, df, store_path)
        print(df)