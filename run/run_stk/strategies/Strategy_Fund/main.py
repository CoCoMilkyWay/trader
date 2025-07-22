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

from config.cfg_stk import cfg_stk
from Util.UtilStk import enable_logging, prepare_all_files, mkdir

from strategies.Strategy_Fund.Parallel_Process_Core import Parallel_Process_Core

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

    # cpu load balancing (from early to new)
    parsed_ipo_dates = [datetime.fromisoformat(date[:-6]) for date in ipo_dates]
    sorted_wt_assets = [x for _, x in sorted(zip(parsed_ipo_dates, wt_assets))][:cfg_stk.num]

    code_info: Dict[str, Dict] = {}
    # 1. prepare meta data
    for idx, code in enumerate(sorted_wt_assets):
        code_info[code] = {'idx':idx}

    # 2. init precess manager
    P = Parallel_Process_Core(code_info)
        
if __name__ == '__main__':
    run_bt()


