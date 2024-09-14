import math
import os, sys
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from pprint import pprint
from typing import List, Dict
from wtpy import BaseCtaStrategy, BaseSelStrategy
from wtpy import CtaContext, SelContext

from Chan.Chan import CChan
from Chan.ChanConfig import CChanConfig
from Chan.Common.CEnum import DATA_SRC, KL_TYPE, AUTYPE, DATA_FIELD, BSP_TYPE, FX_TYPE
from Chan.DataAPI.wtAPI import parse_time_column
from Chan.KLine.KLine_Unit import CKLine_Unit

from db.util import print_class_attributes_and_methods, mkdir
from strategies.Main_Cta_Paral.Parallel_pool import n_processor_queue
from strategies.Main_Cta_Paral.Define import MetadataIn, MetadataOut, bt_config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

red     = "\033[31m"
green   = "\033[32m"
yellow  = "\033[33m"
default = "\033[0m"

def stdio(str):
    print(str)
    return str

def pause():
    import time
    time.sleep(1000)
    return

class Main_Cta(BaseCtaStrategy):
    # CTA engine:
    #   1. timestamp closure
    #   2. each registered k-bar closure -> on_bar()
    #   3. all(include main) k-bar closure -> on_schedule()
    #   4. position/execution (parallel)
    #
    # flow:
    #   1. daily chan with ML pred
    #   2. intra_day T/ stop_loss/gain
    def __init__(self, name:str, codes:List[str], barCnt:int, period:str, capital:float, areForStk:List[bool] = []):
        BaseCtaStrategy.__init__(self, name)
        # when declare __ as static variable, use with caution, compilier may do funny things
        self.__period__         = period
        self.__bar_cnt__        = barCnt
        self.__codes__          = codes
        self.__capital__        = capital
        self.__are_stks__       = areForStk
        self.barnum             = 0
        self.theCodes           : List[str] = []
        self.last_price         : Dict[str, float] = {}
        self.cur_money          = capital
        self.resample_buffer: Dict[str, List[CKLine_Unit]] = {}  # store temp bar to form larger bar
        
        self.column_name = [
            DATA_FIELD.FIELD_TIME,
            DATA_FIELD.FIELD_OPEN,
            DATA_FIELD.FIELD_HIGH,
            DATA_FIELD.FIELD_LOW,
            DATA_FIELD.FIELD_CLOSE,
            DATA_FIELD.FIELD_VOLUME,
            # DATA_FIELD.FIELD_TURNOVER,
            # DATA_FIELD.FIELD_TURNRATE,
            ]  # 每一列字段
        
    def combine_klu(self, resample_buffer: List[CKLine_Unit]) -> CKLine_Unit:
        return CKLine_Unit(
            {
                DATA_FIELD.FIELD_TIME: resample_buffer[-1].time,
                DATA_FIELD.FIELD_OPEN: resample_buffer[0].open,
                DATA_FIELD.FIELD_HIGH: max(klu.high for klu in resample_buffer),
                DATA_FIELD.FIELD_LOW: min(klu.low for klu in resample_buffer),
                DATA_FIELD.FIELD_CLOSE: resample_buffer[-1].close,
                DATA_FIELD.FIELD_VOLUME: sum(klu.volume for klu in resample_buffer),
            },
            autofix=True,
        )
        
    def on_init(self, context:CtaContext):
        # ProcessPoolExecutor: CPU-bound tasks
        # ThreadPoolExecutor: I/O-bound tasks
        # create a N salve 1 master queue as a process pool for assets (persist over the whole bt)
        self.xxx = context.user_load_data('xxx',1)
        for idx, code in enumerate(self.__codes__):
            if self.__are_stks__[idx]:
                self.theCodes.append(code + "-")   # +表示后复权，-表示前复权
            else:
                self.theCodes.append(code)
            # register main k_bar for on_bar(), set bar cache
            context.stra_prepare_bars(code, self.__period__, self.__bar_cnt__, isMain = True)
            self.last_price[code] = 0
            self.resample_buffer[code] = []
        self.processor = n_processor_queue(
            max_workers=128,
            concurrency_mode='process')
        context.stra_log_text(stdio("Data-Struct Initiated"))
        self.pbar = tqdm(total=len(self.__codes__), desc='Preparing Bars in DDR...')
        
    # if you have time sensitive processing (e.g. stop_loss / HFT pattern)
    def on_tick(self, context: CtaContext, stdCode: str, newTick: dict):
        print('on_tick: ', newTick)
        pass
    def on_bar(self, context:CtaContext, stdCode:str, period:str, newBar:dict): # deprecated?
        print('on_bar: ', newBar)
        pass
    
    def on_calculate(self, context:CtaContext):
        self.barnum += 1 # all sub-ed bars closed (main/non-main) at this period
        curTime = context.stra_get_time()
        
        rebalance   = False
        if curTime == 1500: # new date
            rebalance = True
            Meta_queue: List[MetadataIn] = []
        else:
            pass
        
        date = context.get_date()
        health_asset = 0
        for idx, code in enumerate(self.__codes__):
            theCode = self.theCodes[idx]
            if self.barnum == 1:
                self.pbar.update(1)
            else:
                self.pbar.close()
            # sInfo = context.stra_get_sessioninfo(self.theCode)
            # pInfo = context.stra_get_comminfo(self.theCode)
            # if not sInfo.isInTradingTime(curTime): # type: ignore
            #     continue
            capital = self.__capital__
            try: # some asset may not have bar at xxx time
                np_bars = context.stra_get_bars(code, self.__period__, self.__bar_cnt__, isMain=False)
                open    = np_bars.opens[-1]
                high    = np_bars.highs[-1]
                low     = np_bars.lows[-1]
                close   = np_bars.closes[-1]
                volume  = np_bars.volumes[-1]
                bartime = np_bars.bartimes[-1]
                health_asset += 1
            except:
                continue
            
            # ["time_key", "open", "high", "low", "close", "volume", "turnover"] # not include "turnover_rate"
            klu = CKLine_Unit(dict(zip(self.column_name, [
                parse_time_column(str(bartime)),
                open,
                high,
                low,
                close,
                volume,
            ])))
            self.resample_buffer[code].append(klu)
            
            if rebalance:
                combined_klu = self.combine_klu(self.resample_buffer[code])
                Ctime = combined_klu.time
                MetaIn = MetadataIn(
                    idx=idx,
                    code=code,
                    date=date,
                    curTime=curTime,
                    bar=combined_klu,
                    rebalance=rebalance,
                )
                Meta_queue.append(MetaIn)
                self.resample_buffer[code] = []
                
        # print(f'Healthy assets: {health_asset}/{len(self.__codes__)}')
        
        if rebalance:
        # load data to other CPUs and do calc:
        #   1bar        5bar    10bar   50bar   100bar  500bar
        #   120000/s    60000/s 40000/s 8000/s  3500/s  1200/s
            self.processor.clear_result()
            self.processor.add_in_task(Meta_queue) # submit jobs
            # ================================================
            orders = self.processor.step_execute() # blocking until all cpu finish
            # self.profile(date)
            # waiting for exec
            # ================================================
            
            # exec trade orders:
            if len(orders) == 0:
                return
            # print('Main CTA: ', orders)
            for order in orders:
                cpu_id = order.cpu_id
                code = order.code
                buy = order.buy
                sell = order.sell

                curPos = context.stra_get_position(code)
                curPrice = context.stra_get_price(code)
                self.cur_money = capital + context.stra_get_fund_data(flag=0)
                amount = 1000 # math.floor(self.cur_money/curPrice)

                pnl: float = 1
                color: str = default
                if sell and curPos >= 0:
                    if curPos != 0:
                        pnl, color = self.pnl_cal(self.last_price[code], close, False)
                        context.stra_set_position(code, 0, 'exitlong')
                    context.stra_set_position(code, -amount, 'entershort')
                    context.stra_log_text(stdio(f"cpu:{cpu_id:2}:{date}-{curTime}:({code}) top    FX，enter short:{self.cur_money:2f}, pnl:{color}{pnl:2f}{default}"))
                    bt_config.plot_para["marker"]["markers"][Ctime] = ('short', 'down', 'orange')
                    # self.xxx = 1
                    # context.user_save_data('xxx', self.xxx)
                    self.last_price[code] = close
                    self.check_capital()
                    continue
                elif buy and curPos <= 0:
                    if curPos != 0:
                        pnl, color = self.pnl_cal(self.last_price[code], close, True)
                        context.stra_set_position(code, 0, 'exitshort')
                    context.stra_set_position(code, amount, 'enterlong')
                    context.stra_log_text(stdio(f"cpu:{cpu_id:2}:{date}-{curTime}:({code}) bottom FX, enter long :{self.cur_money:2f}, pnl:{color}{pnl:2f}{default}"))
                    bt_config.plot_para["marker"]["markers"][Ctime] = ('long', 'up', 'blue')
                    self.last_price[code] = close
                    self.check_capital()
                    continue
                continue
                    
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
        print('Backtest Done ...')
        self.processor.terminate()
        