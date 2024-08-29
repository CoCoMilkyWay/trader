from __future__ import (absolute_import, division, print_function, unicode_literals)
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategies.DualThrust import StraDualThrust
from strategies.ML_pred import ML_pred
from strategies.Chan_Bsp import Chan_bsp
from db.run_db_maintain import cfg
from db.util import *

from wtpy import WtBtEngine, EngineType, WtDtServo
from wtpy.monitor import WtBtSnooper
from wtpy.wrapper import WtDataHelper
from wtpy.apps import WtBtAnalyst
from wtpy.WtCoreDefs import WTSBarStruct
from wtpy.SessionMgr import SessionMgr
dtHelper = WtDataHelper()

run     = 1
analyze = 1
period  = 'm60'
start   = 202001010930
end     = 202101010930
if __name__ == "__main__":
    print('Preparing dsb data ...')
    #　asset = 'SSE.STK.600000'
    asset = 'sh.000300'
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
    combine_dsb_1m(dtHelper, read_path, store_path_1m, total=True)
    resample(dtHelper, store_path_1m, 5, store_path_5m)
    resample(dtHelper, store_path_1m, 60, store_path_60m)
    resample(dtHelper, store_path_1m, 240, store_path_240m)
    
    # backtesting =================================================================================
    engine = WtBtEngine(EngineType.ET_CTA)
    engine.init(folder='./run', cfgfile="./cfg/configbt.yaml")
    engine.configBacktest(start, end)
    engine.configBTStorage(mode="wtp", path='./storage')
    engine.commitBTConfig()
    
    str_name = f'bt_{asset}'
    bt_folder = f'./outputs_bt'
    from Chan.Common.CEnum import KL_TYPE
    lv_list = [KL_TYPE.K_60M]
    
    # straInfo = StraDualThrust(name=str_name, code=wt_asset, barCnt=50, period=period, days=30, k1=0.1, k2=0.1)
    # straInfo = ML_pred(name=str_name, code=wt_asset, barCnt=1, period=period)
    straInfo = Chan_bsp(name=str_name, code=wt_asset, barCnt=1, period=period, lv_list=lv_list)
    
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