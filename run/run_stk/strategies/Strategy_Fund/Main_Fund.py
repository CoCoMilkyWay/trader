from time import time
from tqdm import tqdm
from typing import List, Dict

from wtpy import SelContext
from wtpy import BaseSelStrategy

from .Parallel_Process_Core import Parallel_Process_Core
from .Parallel_Process_Worker import Parallel_Process_Worker

from config.cfg_stk import cfg_stk
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def stdio(str):
    print(str)
    return str

class Main_Fund(BaseSelStrategy):
    def __init__(self, name: str, codes: List[str], period: str):
        BaseSelStrategy.__init__(self, name)
        
        self.__period__ = period
        self.__codes__ = codes
        
        # stats
        self.inited = False
        self.barnum = 0
        
        # code_info
        self.code_info: Dict[str, Dict] = {}
        
    def on_init(self, context: SelContext):
        # 1. prepare meta data
        for idx, code in enumerate(self.__codes__):
            self.code_info[code] = {
                'idx':idx,
            }
            
        # 2. init worker process
        self.P = Parallel_Process_Core(self.code_info, Parallel_Process_Worker, cfg_stk.parallel)
        
        # 3. prepare backtest data
        print('Preparing Bars in DDR...')
        self.pbar = tqdm(total=len(self.__codes__))
        
        for idx, code in enumerate(self.__codes__):
            r = context.stra_prepare_bars(code, self.__period__, 1)
            self.pbar.update(1)
            self.pbar.set_description(f'Init: {code}', True)
        self.pbar.close()
        
        # 4. check workers
        self.P.check_workers()
        
        context.stra_log_text(stdio("Strategy Initiated, timer started..."))
        self.start_time = time()
        
    def on_tick(self, context:SelContext, code:str, newTick:dict):
        print(code, 'newTick')
        return
    
    def on_bar(self, context:SelContext, code:str, period:str, newBar:dict):
        print(f"on_bar({code}): {newBar}")
        print(context.get_time())
        # self.P.parallel_feed(code, newBar)
        return
    
    def on_calculate(self, context:SelContext):
        """
        TimeSeries   (cpu1):                    |   on_bar_A(T0)    on_bar_A(T1)    ... 
        TimeSeries   (cpu2):                    |   on_bar_B(T0)    on_bar_B(T1)    ... 
        TimeSeries   (... ):                ->  |   ...          -> ...          -> ... 
        CrossSection (cpu0):    on_calc(skip)   V   on_calc(init)   on_calc(T0)     ... (CS is delaying TS by 1 for better compute efficiency)
        """
        if not self.inited:
            self.barnum += 1
            if self.barnum == 1:
                self.inited = True
            return
        # compare_read_dsb_bars()

from wtpy.WtDataDefs import WtNpKline
from wtpy.wrapper import WtDataHelper

dtHelper = WtDataHelper()

def compare_read_dsb_bars():
    
    ret:WtNpKline = dtHelper.read_dsb_bars(f"/home/chuyin/work/trader/database/stock/bars/000032.SZ/1m/2021.11.dsb")
    # ret:WtNpKline = dtHelper.read_dsb_bars(f"{cfg_stk.WT_STORAGE_DIR}/his/min1/SSE/600000.dsb")
    num_bars = len(ret)
    print(f"read_dsb_bars {num_bars} bars")
    print(ret.ndarray[-500:])
    return False
        # self.P.parallel_collect()
    
    # def on_calculate(self, context: SelContext):
    #     # all sub-ed bars closed (main/non-main) at this period
    #     self.barnum += 1
    #     
    #     date = context.get_date()
    #     time = context.stra_get_time()
    #     print(date, time)
    #     if date!=self.date:
    #         print(date)
    #     self.date = date
    #     self.time = time
    #     
    #     for idx, code in enumerate(self.__codes__):
    #         try:
    #             np_bars = context.stra_get_bars(code, self.__period__, 1,)
    #             print(code, np_bars)
    #         except Exception as e:
    #             print(f'{code}: {e}')
    #             continue
    #         if not np_bars:
    #             print(f'no bars: {code}')
    #             continue
    #         # print(np_bars.ndarray)
    #         # multi-level k bar generation
    #         # TA = self.tech_analysis[code]
    #         # TA.analyze(np_bars)
    #         
    #         # indicator guard (prepare and align)
    #         if not self.inited:
    #             if self.barnum < 1*24*60: # need 1 day(s) of 1M data
    #                 self.inited = True
    #             continue
    #         
    #         # strategy
    #         # self.ST_Train(context, code)
    
    def on_backtest_end(self, context: SelContext):
        self.elapsed_time = time() - self.start_time
        self.P.parallel_close()
        print(f'main BT loop time elapsed: {self.elapsed_time:2f}s')
        return
    