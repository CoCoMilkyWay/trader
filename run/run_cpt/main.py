from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import os
import sys

# the path include need to be earlier than relative library
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), "./app"))
from Util.UtilCpt import enable_logging
enable_logging()
from wtpy.apps import WtBtAnalyst
from wtpy.monitor import WtBtSnooper, WtLogger
from wtpy.wrapper import WtDataHelper
from wtpy.SessionMgr import SessionMgr
from wtpy.WtCoreDefs import WTSBarStruct
from wtpy import WtBtEngine, EngineType, WtDtServo

from Util.UtilCpt import generate_asset_list, generate_database_files, generate_merged_database_files, testBtSnooper
from config.cfg_cpt import cfg_cpt
from strategy.CPT_Chart.Main_Cta import Main_Cta

run = True
analyze = False
snoop = False
profile = False

def run_bt():
    ''' refer to run/db/db_cfg.py for other configs '''

    assets = cfg_cpt.symbols
    CSV_DIR = cfg_cpt.CRYPTO_CSV_DIR
    period_str = cfg_cpt.period + str(cfg_cpt.n)

    print('Preparing dsb data (Combining and Resampling) ...')
    wt_assets = generate_asset_list()
    generate_database_files()
    generate_merged_database_files(resample_n=cfg_cpt.n)
    
    print('Data ready: ', cfg_cpt.symbols)

    # backtesting =================================================================================
    print('Initializing Backtest ...')
    engine = WtBtEngine(EngineType.ET_CTA)
    engine.init(folder='.', cfgfile='./config/configbt.yaml')
    engine.configBacktest(cfg_cpt.start, cfg_cpt.end)
    engine.commitBTConfig()

    str_name = f'bt_crypto'
    bt_folder = f'./outputs_bt'

    # straInfo = StraDualThrust(name=str_name, code=wt_assets[0], barCnt=50, period=period_str, days=30, k1=0.1, k2=0.1)
    # straInfo = ML_pred(name=str_name, code=wt_asset, barCnt=1, period=period_str)
    straInfo = Main_Cta(name=str_name, codes=wt_assets,
                        period=period_str, capital=cfg_cpt.capital)

    engine.set_cta_strategy(straInfo, slippage=0)
    
    if run:
        print('Running Backtest ...')
        print('Use CTRL+4 to interrupt')
        engine.run_backtest()
        print(f'Backtest Done')

    if analyze:
        print('Analyzing ...')
        analyst = WtBtAnalyst()
        analyst.add_strategy(str_name, folder=bt_folder,
                             init_capital=cfg_cpt.capital, rf=0.0, annual_trading_days=240)
        analyst.run_new()

    if snoop:
        print('http://127.0.0.1:8081/backtest/backtest.html')
        testBtSnooper()
        # kw = input('press any key to exit\n')
        engine.release_backtest()
''' '''

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
