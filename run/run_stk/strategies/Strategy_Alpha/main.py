import os
import sys
from typing import List, Dict
from datetime import datetime

# the path include need to be earlier than relative library
TOP = "../../../../"
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), TOP))
sys.path.append(os.path.join(os.path.dirname(__file__), TOP + "app"))
sys.path.append(os.path.join(os.path.dirname(__file__), TOP + "app/Exchange_API/"))

from Util.UtilStk import enable_logging, prepare_all_files, mkdir
from wtpy.apps import WtBtAnalyst
from wtpy.monitor import WtBtSnooper
from wtpy import WtBtEngine, EngineType, WtDtServo

from config.cfg_stk import cfg_stk
from strategies.Strategy_Alpha.Main_Alpha import Main_Alpha


run = True
analyze = cfg_stk.analyze
snoop = cfg_stk.snoop if analyze else False
panel = cfg_stk.panel if snoop else False

def run_bt():
    ''' refer to run/db/db_cfg.py for other configs '''
    enable_logging()
    
    wt_asset = prepare_all_files()

    wt_assets: List[str] = []
    ipo_dates: List[str] = []
    for exg in cfg_stk.exchg:
        assets_exchg = os.listdir(mkdir(f"{cfg_stk.WT_STORAGE_DIR}/his/min1/{exg}/"))
        for key in wt_asset[exg]:
            if f"{key}.dsb" in assets_exchg:
                wt_assets.append(f'{exg}.{wt_asset[exg][key]['product']}.{key}')
                ipo_dates.append(wt_asset[exg][key]['extras']['ipoDate'])
    # load balancing
    ipo_dates_parsed = [datetime.fromisoformat(date[:-6]) for date in ipo_dates]
    wt_assets = [x for _, x in sorted(zip(ipo_dates_parsed, wt_assets))][:cfg_stk.num]

    # backtesting =================================================================================
    engine = WtBtEngine(EngineType.ET_SEL, logCfg='./config/logcfg.yaml')
    engine.init(folder='.', cfgfile='./config/configbt.yaml')
    engine.configBacktest(cfg_stk.start, cfg_stk.end)
    engine.commitBTConfig()
    
    str_name = f'bt_stock'
    bt_folder = f'./outputs_bt'
    
    if run:
        straInfo = Main_Alpha(
            name=str_name, 
            codes=wt_assets,
            period=cfg_stk.wt_period_l)
        engine.set_sel_strategy(
            straInfo, 
            date=0, time=cfg_stk.n, period=cfg_stk.period_u, 
            isRatioSlp=False)
        engine.run_backtest()
    
    if analyze:
        print('Analyzing ...')
        analyst = WtBtAnalyst()
        analyst.add_strategy(str_name, folder=bt_folder,
                             init_capital=cfg_stk.capital, rf=0.0, annual_trading_days=240)
        analyst.run_new()
    
    if snoop:
        print('http://127.0.0.1:8081/backtest/backtest.html')
        
        dtServo = WtDtServo()
        
        dtServo.setBasefiles(
            folder          = "./config/",
            commfile=     'cpt_comms.json', 
            contractfile= 'cpt_assets.json', 
            holidayfile=  'cpt_holidays.json', 
            sessionfile=  'cpt_sessions.json',
            hotfile         = "",
                             )
        dtServo.setStorage(
            path='../storage',
            adjfactor=''
            )
        snooper = WtBtSnooper(dtServo)
        snooper.run_as_server(port=8081, host="0.0.0.0")
        # kw = input('press any key to exit\n')
        engine.release_backtest()

if __name__ == '__main__':
    run_bt()

    if panel:
        from wtpy.monitor import WtMonSvr
        # 如果要配置在线回测，则必须要配置WtDtServo
        from wtpy import WtDtServo
        dtServo = WtDtServo()
        dtServo.setBasefiles(
            commfile='./config/cpt_comms.json', 
            contractfile= './config/cpt_assets.json', 
            holidayfile=  './config/cpt_holidays.json', 
            sessionfile=  './config/cpt_sessions.json',
            )
        dtServo.setStorage('../storage/')
        dtServo.commitConfig()
        # 创建监控服务，deploy_dir是策略组合部署的根目录，默认不需要传，会自动定位到wtpy内置的html资源目录
        svr = WtMonSvr(deploy_dir='./deploy')
        # 将回测管理模块提交给WtMonSvr
        from wtpy.monitor import WtBtMon
        btMon = WtBtMon(deploy_folder='./outputs_bt', logger=svr.logger) # 创建回测管理器
        svr.set_bt_mon(btMon) # 设置回测管理器
        svr.set_dt_servo(dtServo) # 设置dtservo
        # 启动服务
        svr.run(port=8099, bSync=False)
        print('PC版控制台入口地址: http://127.0.0.1:8099/console')
        print('移动版控制台入口地址： http://127.0.0.1:8099/mobile')
        print('superman/Helloworld!')
        input('press enter key to exit\n')
