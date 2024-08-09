from __future__ import (absolute_import, division, print_function, unicode_literals)
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategies.DualThrust import StraDualThrust
from db.run_db_maintain import cfg

from wtpy import WtBtEngine, EngineType, WtDtServo
from wtpy.monitor import WtBtSnooper
from wtpy.wrapper import WtDataHelper
from wtpy.apps import WtBtAnalyst
from wtpy.WtCoreDefs import WTSBarStruct
from wtpy.SessionMgr import SessionMgr
dtHelper = WtDataHelper()

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
        return None  # Invalid format

def sort_files_by_date(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.dsb')]
    files_with_dates = []
    for file in files:
        file_date = get_file_date(file)
        if file_date:
            files_with_dates.append((file, file_date))
    # Sort the files by their date
    sorted_files = sorted(files_with_dates, key=lambda x: x[1])
    return [file for file, _ in sorted_files]

def wt_csv_2_dsb(df, store_path):
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

def combine_dsb_1m(read_path, store_path):
    if not os.path.exists(store_path):
        # asset_dsb = [os.path.join(read_path, file) for file in os.listdir(read_path) if file.endswith('.dsb')]
        sorted_file_list = sort_files_by_date(read_path)
        df = []
        for file in tqdm(sorted_file_list):
            file_path = os.path.join(read_path, file)
            df.append(dtHelper.read_dsb_bars(file_path).to_df())
        df = pd.concat(df, ignore_index=False)
        wt_csv_2_dsb(df, store_path)

def resample(src_path, times, store_path):
    if not os.path.exists(store_path):
        sessMgr = SessionMgr()
        sessMgr.load("cfg/sessions/sessions.json")
        sInfo = sessMgr.getSession("SD0930")
        df = dtHelper.resample_bars(src_path,'m1',times,200001010931,209901010931,sInfo, True).to_df()
        wt_csv_2_dsb(df, store_path)
        print(df)

run     = 1
analyze = 1
period  = 'm60'
start   = 200501010930
end     = 202407010930
if __name__ == "__main__":
    print('Preparing dsb data ...')
    #　asset = 'SSE.STK.600000'
    asset = 'sh.000001'
    asset_dict = {'sh':'SSE', 'sz':'SZSE'}
    parts = asset.split(sep='.')
    exchange = asset_dict[parts[0]]
    code = parts[1]
    with open(cfg.STOCKS_FILE, 'r', encoding='gbk', errors='ignore') as file:
        stocks = json.load(file)
    try: 
        type = stocks[exchange][code]['product']
    except:
        type = 'STK'
    wt_asset = f'{asset_dict[parts[0]]}.{type}.{parts[1]}'
    read_path = f"{cfg.BAR_DIR}/m1/{asset}"
    store_path_1m = f"{cfg.WT_STORAGE_DIR}/his/min1/{exchange}/{code}.dsb"
    store_path_5m = f"{cfg.WT_STORAGE_DIR}/his/min5/{exchange}/{code}.dsb"
    store_path_60m = f"{cfg.WT_STORAGE_DIR}/his/min60/{exchange}/{code}.dsb"
    store_path_240m = f"{cfg.WT_STORAGE_DIR}/his/min240/{exchange}/{code}.dsb"
    
    print('Resampling ...')
    combine_dsb_1m(read_path, store_path_1m)
    resample(store_path_1m, 5, store_path_5m)
    resample(store_path_1m, 60, store_path_60m)
    resample(store_path_1m, 240, store_path_240m)

    # backtesting =================================================================================
    engine = WtBtEngine(EngineType.ET_CTA)
    engine.init(folder='./run', cfgfile="./cfg/configbt.yaml")
    engine.configBacktest(start, end)
    engine.configBTStorage(mode="wtp", path='./storage')
    engine.commitBTConfig()
    
    str_name = f'bt_{asset}'
    bt_folder = f'./outputs_bt'
    straInfo = StraDualThrust(name=str_name, code=wt_asset, barCnt=50, period=period, days=30, k1=0.1, k2=0.1)
    engine.set_cta_strategy(straInfo)
    
    print('Running Backtest ...')
    if run:
        engine.run_backtest()
    
    print('Analyzing ...')
    analyst = WtBtAnalyst()
    analyst.add_strategy(str_name, folder=bt_folder, init_capital=500000, rf=0.0, annual_trading_days=240)
    if analyze:
        analyst.run_flat()
    
    print('http://127.0.0.1:8081/backtest/backtest.html')
    testBtSnooper()
    
    kw = input('press any key to exit\n')
    engine.release_backtest()

'''
from wtpy.monitor import WtMonSvr
# 如果要配置在线回测，则必须要配置WtDtServo
from wtpy import WtDtServo
dtServo = WtDtServo()
dtServo.setBasefiles(commfile="./cfg/assets_list/commodities.json", 
                contractfile="./cfg/assets_list/stocks.json", 
                holidayfile="./cfg/misc/holidays.json", 
                sessionfile="./cfg/sessions/sessions.json", 
                hotfile="./cfg/assets_list/hots.json")
dtServo.setStorage("./storage/")
dtServo.commitConfig()
# 创建监控服务，deploy_dir是策略组合部署的根目录，默认不需要传，会自动定位到wtpy内置的html资源目录
svr = WtMonSvr(deploy_dir="./deploy")
# 将回测管理模块提交给WtMonSvr
from wtpy.monitor import WtBtMon
btMon = WtBtMon(deploy_folder="./bt_deploy", logger=svr.logger) # 创建回测管理器
svr.set_bt_mon(btMon) # 设置回测管理器
svr.set_dt_servo(dtServo) # 设置dtservo
# 启动服务
svr.run(port=8099, bSync=False)
print("PC版控制台入口地址: http://127.0.0.1:8099/console")
print("移动版控制台入口地址： http://127.0.0.1:8099/mobile")
print("superman/Helloworld!")
input("press enter key to exit\n")
'''