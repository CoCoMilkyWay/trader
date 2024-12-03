import math
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../app"))

import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from pprint import pprint
from typing import List, Dict

from wtpy import BaseCtaStrategy
from wtpy import CtaContext

from Chan.Chan import CChan
from Chan.ChanConfig import CChanConfig
from Chan.Common.CEnum import KL_TYPE
from Chan.Common.kbar_parser import KLineHandler
from Chan.KLine.KLine_List import CKLine_List

from Util.MemoryAnalyzer import MemoryAnalyzer

red     = "\033[31m"
green   = "\033[32m"
yellow  = "\033[33m"
default = "\033[0m"

CHECK_SYNC = True

def stdio(str):
    print(str)
    return str

def pause():
    import time
    time.sleep(1000)
    return

class Main_Cta(BaseCtaStrategy):
    def __init__(self, name:str, codes:List[str], period:str, capital:float):
        BaseCtaStrategy.__init__(self, name)
        self.__period__                     = period
        self.__codes__                      = codes
        self.__capital__                    = capital

        self.config: CChanConfig            = CChanConfig()
        self.KLineHandler: Dict[str, KLineHandler] = {}
        self.kl_datas: Dict[str, Dict[KL_TYPE, CKLine_List]] = {}
        
        # stats
        self.barnum                         = 0
        self.last_price: Dict[str, float]   = {}
        self.cur_money                      = capital
        self.start_time                     = time()
        self.date                           = None
        
        # models
        self.chan_snapshot: Dict[str, CChan] = {}
        
        # factors
                
    def on_init(self, context:CtaContext):
        print('Initializing Strategy...')
        self.lv_list = [lv[0] for lv in self.config.lv_list]
        
        for idx, code in enumerate(self.__codes__):
            context.stra_prepare_bars(code, self.__period__, 1, isMain = idx==0)
            # only 1 series is registered as 'Main', which works as clock, more registrations would result in fault
            # on_calculate is triggered once main bar is closed
            # if hook is installed, on_calculate_done would be triggered(for RL)
            
            self.KLineHandler[code] = KLineHandler(self.config.lv_list)
            self.init_shared_kl_datas(code)
            self.last_price[code] = 0
        
        context.stra_log_text(stdio("Strategy Initiated"))
        self.pbar = tqdm(total=len(self.__codes__), desc='Preparing Bars in DDR...')

    def init_shared_kl_datas(self, code:str):
        """Initialize K-line data structures for each time level."""
        self.kl_datas[code] = {level: CKLine_List(level, conf=self.config)for level in self.lv_list}

    def init_new_code(self, code:str):

        # initiate new code specific models/structs
        self.chan_snapshot[code] = CChan(
            code=code,
            kl_datas=self.kl_datas[code],
            # begin_time=begin_time,
            # end_time=end_time,
            # data_src=data_src,
            lv_list=self.lv_list,
            # config=config,
            # autype=AUTYPE.QFQ,
        )
        
    def on_calculate(self, context:CtaContext):
        self.barnum += 1 # all sub-ed bars closed (main/non-main) at this period
        
        date = context.get_date()
        time = context.stra_get_time()
        if date!=self.date:
            print(date)
            # print(date, time)
            self.date = date
            
        for idx, code in enumerate(self.__codes__):
            if self.barnum == 1: 
                self.pbar.update(1)
                self.init_new_code(code)
            else: self.pbar.close()
            
            np_bars = context.stra_get_bars(code, self.__period__, 1, isMain=idx==0)
            # close = np_bars.closes[-1]
            # print(f"{code:010} {date:08}:{time:04}")
            
            # multi-level k bar generation
            klu_dict = self.KLineHandler[code].process_bar(np_bars)
            
            # process Chan elements
            self.chan_snapshot[code].trigger_load(klu_dict)
            
            # process PA elements
            for lv in self.lv_list:
                self.kl_datas[code][lv].PA_Core.parse_dynamic_bi_list()
            
    def trade_order(self, context:CtaContext, code:str, buy:bool, sell:bool, price:float, date:int, time:int):
        cpu_id  = 0
        curPos = context.stra_get_position(code)
        curPrice = context.stra_get_price(code)
        self.cur_money = self.__capital__ + context.stra_get_fund_data(flag=0)
        amount = math.floor(self.cur_money/curPrice/len(self.__codes__))
        
        pnl: float = 0
        color: str = default
        FX = '^' if sell else 'v' if buy else '?'
        if sell and curPos >= 0:
            if curPos != 0:
                pnl, color = self.pnl_cal(self.last_price[code], price, False)
                context.stra_set_position(code, 0, 'exitlong')
            context.stra_set_position(code, -amount, 'entershort')
            context.stra_log_text(stdio(f"cpu:{cpu_id:2}:{date}-{time:4}:({code:>15}) {FX}，enter short:{self.cur_money:>12.2f}, pnl:{color}{pnl*100:>+5.1f}%{default}"))
            # self.xxx = 1
            # context.user_save_data('xxx', self.xxx)
            self.last_price[code] = price
            self.check_capital()
            return
        elif buy and curPos <= 0:
            if curPos != 0:
                pnl, color = self.pnl_cal(self.last_price[code], price, True)
                context.stra_set_position(code, 0, 'exitshort')
            context.stra_set_position(code, amount, 'enterlong')
            context.stra_log_text(stdio(f"cpu:{cpu_id:2}:{date}-{time:4}:({code:>15}) {FX}, enter long :{self.cur_money:>12.2f}, pnl:{color}{pnl*100:>+5.1f}%{default}"))
            self.last_price[code] = price
            self.check_capital()
            return
        return
    
    def check_capital(self):
        try:
            assert self.cur_money>0
        except AssertionError:
            print(f"lost, stopping ...")
            os._exit(1)
            
    def pnl_cal(self, last_price: float, price: float, long_short: bool):
        if long_short: dir = 1
        else: dir = -1
        pnl = dir*(last_price - price)/last_price
        if pnl > 0.01: color = green
        elif pnl < -0.01: color = red
        else: color = default
        return pnl, color
    
    def profile(self, date):
        try:
            num = 1 # self.barnum-self.barnum_bak
            print(f'({date}): {num/(time()-self.time):.2f} days/sec')
        except:
            pass
        self.time = time()
        self.barnum_bak = self.barnum
        
    def on_backtest_end(self, context:CtaContext):
        self.elapsed_time = time() - self.start_time
        print(f'main BT loop time elapsed: {self.elapsed_time:2f}s')
        
        # chan=self.chan_snapshot[self.__codes__[0]]
        # MemoryAnalyzer().analyze_object(chan)
        # # MemoryAnalyzer().analyze_object(list(chan.kl_datas.items())[-1][1])
        # for obj in list(chan.kl_datas.items()):
        #     size = MemoryAnalyzer().get_deep_size(obj)
        #     print(f'{size/1000/1000:3.2f}MB: {obj}')
        from Chan.Plot.PlotDriver import ChanPlotter
        from Util.plot.plot_fee_grid import plot_fee_grid
        from Util.plot.plot_show import plot_show
        
        for code in self.__codes__:
            fig = ChanPlotter().plot(self.kl_datas[code])
            fig = plot_fee_grid(fig)
            plot_show(fig)
            
        return fig