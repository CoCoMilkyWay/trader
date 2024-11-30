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
from wtpy.WtDataDefs import WtNpKline

from Chan.Chan import CChan
from Chan.Common.CEnum import KL_TYPE, DATA_FIELD
from Chan.Common.CTime import CTime
from Chan.KLine.KLine_Unit import CKLine_Unit

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

        # stats
        self.barnum                         = 0
        self.last_price: Dict[str, float]   = {}
        self.cur_money                      = capital
        self.start_time                     = time()
        
        # models
        self.chan_snapshot: Dict[str, CChan] = {}
        
        # factors
        self.np_bars_batch: Dict[str, List[WtNpKline]] = {}  # store temp bar to form larger bar
        
    def on_init(self, context:CtaContext):
        for idx, code in enumerate(self.__codes__):
            context.stra_prepare_bars(code, self.__period__, 1, isMain = idx==0)
            # only 1 series is registered as 'Main', which works as clock, more registrations would result in fault
            # on_calculate is triggered once main bar is closed
            # if hook is installed, on_calculate_done would be triggered(for RL)
            self.last_price[code] = 0
            self.np_bars_batch[code] = []
        
        self.lv_list = [KL_TYPE.K_1M]
        
        context.stra_log_text(stdio("Strategy Initiated"))
        self.pbar = tqdm(total=len(self.__codes__), desc='Preparing Bars in DDR...')
    
    def on_calculate(self, context:CtaContext):
        self.barnum += 1 # all sub-ed bars closed (main/non-main) at this period
        
        date = context.get_date()
        time = context.stra_get_time()
        
        for idx, code in enumerate(self.__codes__):
            if self.barnum == 1: 
                self.pbar.update(1)
                self.init_new_code(code)
            else: self.pbar.close()
            
            np_bars = context.stra_get_bars(code, self.__period__, 1, isMain=idx==0)
            close = np_bars.closes[-1]
            # print(f"{code:010} {date:08}:{time:04}")
            
            chan_snapshot = self.chan_snapshot[code]
            chan_snapshot.trigger_load({self.lv_list[0]: [self.parse_klu(np_bars)]}) # feed day bar

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


    def init_new_code(self, code):
            
        # initiate new code specific models/structs
        self.chan_snapshot[code] = CChan(
            code=code,
            # begin_time=begin_time,
            # end_time=end_time,
            # data_src=data_src,
            lv_list=self.lv_list,
            # config=config,
            # autype=AUTYPE.QFQ,
        )

    @staticmethod
    def parse_klu(np_bars:WtNpKline) -> CKLine_Unit:
        def parse_time_column(time_str:str):
            # 2020_1102_0931
            if len(time_str) == 12:
                year = int(time_str[:4])
                month = int(time_str[4:6])
                day = int(time_str[6:8])
                hour = int(time_str[8:10])
                minute = int(time_str[10:12])
            else:
                raise Exception(f"unknown time column from csv:{time_str}")
            return CTime(year, month, day, hour, minute, auto=False)
        
        return CKLine_Unit(
        {
            DATA_FIELD.FIELD_TIME:      parse_time_column(str(np_bars.bartimes[-1])),
            DATA_FIELD.FIELD_OPEN:      np_bars.opens[-1],
            DATA_FIELD.FIELD_HIGH:      np_bars.highs[-1],
            DATA_FIELD.FIELD_LOW:       np_bars.lows[-1],
            DATA_FIELD.FIELD_CLOSE:     np_bars.closes[-1],
            DATA_FIELD.FIELD_VOLUME:    int(np_bars.volumes[-1]),
        },
        autofix=True)

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
        
        self.chan_snapshot[self.__codes__[0]].plot()
