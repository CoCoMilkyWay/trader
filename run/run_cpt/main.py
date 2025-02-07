import os
import sys

# the path include need to be earlier than relative library
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), "./app"))
sys.path.append(os.path.join(os.path.dirname(__file__), "./app/Api"))

from Util.UtilCpt import enable_logging
from wtpy.apps import WtBtAnalyst
from wtpy.monitor import WtBtSnooper
from wtpy import WtBtEngine, EngineType, WtDtServo

from Util.UtilCpt import generate_asset_list, generate_database_files, generate_merged_database_files, testBtSnooper
from config.cfg_cpt import cfg_cpt
from strategy.CPT_Alpha.Main_Alpha import Main_Alpha

train = cfg_cpt.train

run = train
analyze = cfg_cpt.analyze
snoop = cfg_cpt.snoop if analyze else False
panel = cfg_cpt.panel if snoop else False

def run_bt():
    ''' refer to run/db/db_cfg.py for other configs '''
    enable_logging()
    
    period = cfg_cpt.period + str(cfg_cpt.n)
    
    wt_assets, symbols = generate_asset_list(num=cfg_cpt.num)
    generate_database_files(symbols, force_sync=False)
    generate_merged_database_files(symbols, resample_n=cfg_cpt.n)
    
    print(f'Data ready({len(symbols)} symbols): ', symbols)
    
    # backtesting =================================================================================
    print('Initializing Backtest ...')
    engine = WtBtEngine(EngineType.ET_SEL)
    engine.init(folder='.', cfgfile='./config/configbt.yaml')
    engine.configBacktest(cfg_cpt.start, cfg_cpt.end)
    engine.commitBTConfig()
    
    str_name = f'bt_crypto'
    bt_folder = f'./outputs_bt'
    
    if run:
        if train:
            straInfo = Main_Alpha(name=str_name, codes=wt_assets,period=period)
        engine.set_sel_strategy(straInfo, time=1, period='min', trdtpl='NO_HOLIDAYS', session='ALLDAY', slippage=0, isRatioSlp=False)
        print('Running Backtest ...')
        print('Use CTRL+4 to interrupt')
        engine.run_backtest()
        print(f'Backtest Done')
    
    if analyze:
        print('Analyzing ...')
        analyst = WtBtAnalyst()
        analyst.add_strategy(str_name, folder=bt_folder,
                             init_capital=cfg_cpt.capital, rf=0.0, annual_trading_days=365)
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
