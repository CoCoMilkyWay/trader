from time import time
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional

from wtpy import SelContext
from wtpy import BaseSelStrategy

from .Parallel_Process import Parallel_Process
from .Main_Alpha_Core import Main_Alpha_Core

from config.cfg_cpt import cfg_cpt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def stdio(str):
    print(str)
    return str

class Main_Alpha(BaseSelStrategy):
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
        print('Preparing Bars in DDR...')
        self.pbar = tqdm(total=len(self.__codes__))
        for idx, code in enumerate(self.__codes__):
            r = context.stra_prepare_bars(code, self.__period__, 1)
            self.code_info[code] = {
                'idx':idx,
            }
            self.pbar.update(1)
            self.pbar.set_description(f'init: {code}', True)
        self.pbar.close()
        
        self.P = Parallel_Process(self.code_info, Main_Alpha_Core)
        
        context.stra_log_text(stdio("Strategy Initiated, timer started..."))
        self.start_time = time()
            
    def on_tick(self, context:SelContext, code:str, newTick:dict):
        print(code, 'newTick')
        return
    
    def on_bar(self, context:SelContext, code:str, period:str, newBar:dict):
        self.P.parallel_feed(code, newBar)
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
        
        self.P.parallel_collect()
        
        
    
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
        self.P.parallel_close()
        self.elapsed_time = time() - self.start_time
        print(f'main BT loop time elapsed: {self.elapsed_time:2f}s')
        return
        