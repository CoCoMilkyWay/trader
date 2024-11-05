from __future__ import (absolute_import, division, print_function, unicode_literals)
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategies.DualThrust import StraDualThrust
from strategies.ML_pred import ML_pred
# from strategies.Main_Sel import Main_Sel
# from strategies.Main_Cta import Main_Cta
from strategies.Main_Cta_Paral.Main_Cta import Main_Cta
from db.run_db_maintain import cfg
from db.util import combine_dsb_1m, resample, testBtSnooper, get_bao_stocks

from wtpy import WtBtEngine, EngineType, WtDtServo
from wtpy.WtCoreDefs import WTSBarStruct
from wtpy.SessionMgr import SessionMgr
from wtpy.wrapper import WtDataHelper
from wtpy.monitor import WtBtSnooper
from wtpy.apps import WtBtAnalyst

dtHelper = WtDataHelper()

run             = True
analyze         = False
snoop           = False
profile         = False
period, n       = 'm', 1 # bar period
start           = 202401010931
end             = 202404010000
capital         = 10000000

def run_bt():
    
    # Restart the Python interpreter, effectively killing all threads
    # os.execv(sys.executable, [sys.executable] + sys.argv)
    
    NUM= None # N/None
    print('Pulling stock pool ...')
    if NUM:
        assets_list, assets_valid = get_bao_stocks(pool='zz500')
        assets =  [asset for asset in assets_list if asset.startswith('sh')][:NUM]
        assets += [asset for asset in assets_list if asset.startswith('sz')][:NUM]
    else: assets = ['sh.600004',] # 'sz.000009']
    
    print('Preparing dsb data (Combining and Resampling) ...')
    asset_dict = {'sh':'SSE', 'sz':'SZSE'} # wt_asset = 'SSE.STK.600000'
    period_dict = {'m': 'min'}
    period_str = period + str(n)
    wt_assets = []
    wt_assets_types = []
    with open(cfg.STOCKS_FILE, 'r', encoding='gbk', errors='ignore') as file:
        stocks = json.load(file)
    for asset in assets:
        parts = asset.split(sep='.')
        exchange = asset_dict[parts[0]]
        code = parts[1]
        try:
            type = stocks[exchange][code]['product']
        except:
            type = 'STK'
        read_path = f'{cfg.BAR_DIR}/m1/{asset}'
        store_path_1m = f'{cfg.WT_STORAGE_DIR}/his/min1/{exchange}/{code}.dsb'
        store_path_target = f'{cfg.WT_STORAGE_DIR}/his/{period_dict[period]+str(n)}/{exchange}/{code}.dsb'
        try:
            combine_dsb_1m(dtHelper, read_path, store_path_1m, total=True)
        except:
            print(f'Err processing: {asset}')
            continue
        if n!=1:
            resample(dtHelper, store_path_1m, n, store_path_target)
        wt_assets.append(f'{asset_dict[parts[0]]}.{type}.{parts[1]}')
        wt_assets_types.append(type)
    wt_assets_skt = [True if assets_type == 'STK' else False for assets_type in wt_assets_types]
    print('Data ready: ', list(zip(wt_assets, wt_assets_types)))
    
    # backtesting =================================================================================
    print('Initializing Backtest ...')
    engine = WtBtEngine(EngineType.ET_CTA)
    engine.init(folder='./run', cfgfile='./cfg/configbt.yaml')
    engine.configBacktest(start, end)
    engine.configBTStorage(mode='wtp', path='./storage')
    engine.commitBTConfig()
    
    str_name = f'bt_{asset}'
    bt_folder = f'./outputs_bt'
    
    # straInfo = StraDualThrust(name=str_name, code=wt_asset, barCnt=50, period=period_str, days=30, k1=0.1, k2=0.1)
    # straInfo = ML_pred(name=str_name, code=wt_asset, barCnt=1, period=period_str)
    straInfo = Main_Cta(name=str_name, codes=wt_assets, barCnt=1, period=period_str, capital=capital, areForStk=wt_assets_skt)
    
    engine.set_cta_strategy(straInfo, slippage=0)
    
    if run:
        print('Running Backtest ...')
        print('Use CTRL+4 to interrupt')
        engine.run_backtest()
        print(f'Backtest Done')
        
    if analyze:
        print('Analyzing ...')
        analyst = WtBtAnalyst()
        analyst.add_strategy(str_name, folder=bt_folder, init_capital=capital, rf=0.0, annual_trading_days=240)
        analyst.run_new()
        
    if snoop:
        print('http://127.0.0.1:8081/backtest/backtest.html')
        testBtSnooper()
        # kw = input('press any key to exit\n')
        engine.release_backtest()
        
if __name__ == '__main__':
    if profile:
        import cProfile
        cProfile.run('run_bt()', 'run_bt.prof')
        # snakeviz run_bt.prof
    else:
        run_bt()
    
'''
from wtpy.monitor import WtMonSvr
# 如果要配置在线回测，则必须要配置WtDtServo
from wtpy import WtDtServo
dtServo = WtDtServo()
dtServo.setBasefiles(commfile='./cfg/assets_list/commodities.json', 
                contractfile='./cfg/assets_list/stocks.json', 
                holidayfile='./cfg/misc/holidays.json', 
                sessionfile='./cfg/sessions/sessions.json', 
                hotfile='./cfg/assets_list/hots.json')
dtServo.setStorage('./storage/')
dtServo.commitConfig()
# 创建监控服务，deploy_dir是策略组合部署的根目录，默认不需要传，会自动定位到wtpy内置的html资源目录
svr = WtMonSvr(deploy_dir='./deploy')
# 将回测管理模块提交给WtMonSvr
from wtpy.monitor import WtBtMon
btMon = WtBtMon(deploy_folder='./bt_deploy', logger=svr.logger) # 创建回测管理器
svr.set_bt_mon(btMon) # 设置回测管理器
svr.set_dt_servo(dtServo) # 设置dtservo
# 启动服务
svr.run(port=8099, bSync=False)
print('PC版控制台入口地址: http://127.0.0.1:8099/console')
print('移动版控制台入口地址： http://127.0.0.1:8099/mobile')
print('superman/Helloworld!')
input('press enter key to exit\n')
'''

'''
Exchange Location: Mainland China ======================================================
Share Class: A-shares
Description: China securities incorporated in mainland China, listed on the Shanghai or Shenzhen Stock Exchange and traded in yuan (CNY).
Trading FX: CNY
Largest Sector: Financials
Companies: Kweichow Moutai, Ping An Insurance, China Merchants Bank

Share Class: B-shares
Description: China securities incorporated in mainland China, listed on the Shanghai Stock Exchange (USD) and Shenzhen Stock Exchange (HKD).
Trading FX: USD/HKD
Largest Sector: Industrials
Companies: Shanghai Lujiazui Fin & Trade Dev, Inner Mongolia Yitai Coal, Chongqing

Exchange Location: International ======================================================
Share Class: H-shares
Description: China securities incorporated in mainland China, listed on the Hong Kong Stock Exchange (HKD).
Trading FX: HKD
Largest Sector: Financials
Companies: China Construction Bank, ICBC, Ping An Insurance

Share Class: Red chips
Description: China securities of state-owned companies incorporated outside mainland China, listed on the Hong Kong Stock Exchange (HKD).
Trading FX: HKD
Largest Sector: Telecom. Services
Companies: China Mobile, CNOOC, BOC Hong Kong

Share Class: P chips
Description: China securities of non-government-owned companies incorporated outside mainland China, listed on the Hong Kong Stock Exchange (HKD).
Trading FX: HKD
Largest Sector: Consumer Discretionary
Companies: Tencent, Geely Automobile, China Evergrande

Share Class: N-shares
Description: China securities (including ADRs) incorporated outside Greater China (mainland China, Hong Kong, Macao, and Taiwan), listed on the NYSE Euronext-New York, NASDAQ, and NYSE AMEX.
Trading FX: USD
Largest Sector: Information Technology
Companies: Alibaba, Baidu, JD.com

Share Class: S-shares
Description: China securities traded on Singapore Exchanges, in Singapore dollars (SGD).
Trading FX: SGD
Largest Sector: Industrials
Companies: Yangzijiang Shipbuilding Holdings, Yanlord Land Group, SIIC Environmental

'''