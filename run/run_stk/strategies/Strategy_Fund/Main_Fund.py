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
        assert False
        
        # 4. check workers
        self.P.check_workers()
        
        context.stra_log_text(stdio("Strategy Initiated, timer started..."))
        self.start_time = time()
        
    def on_tick(self, context:SelContext, code:str, newTick:dict):
        print(code, 'newTick')
        return
    
    def on_bar(self, context:SelContext, code:str, period:str, newBar:dict):
        print(f"on_bar({code}): {newBar} {period}")
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

    def on_backtest_end(self, context: SelContext):
        # self.elapsed_time = time() - self.start_time
        # self.P.parallel_close()
        # print(f'main BT loop time elapsed: {self.elapsed_time:2f}s')
        return
