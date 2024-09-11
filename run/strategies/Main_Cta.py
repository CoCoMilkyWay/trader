import math
import os, sys
import numpy as np
import pandas as pd
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
        self.lv_list            = [KL_TYPE.K_DAY]
        self.resample_buffer    : Dict[str, List[CKLine_Unit]] = {}  # store temp bar to form larger bar
        self.barnum             = 0
        self.date               = 0
        self.theCodes           : List[str] = []
        self.last_price         : Dict[str, float] = {}
        self.cur_money          = capital
        
        self.num_bsp_T1         = 0
        self.num_bsp_T2         = 0
        self.num_bsp_T3         = 0
                
        # tune Chan config for day bar
        self.config = CChanConfig({
            "trigger_step"      : True,
            "skip_step"         : 0,
            
            # Bi
            "bi_algo"           : "normal",
            "bi_strict"         : True,
            "bi_fx_check"       : "strict", # 突破够强，回撤不大(滤掉绞肉机行情)
        })
        
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
            }
        )
        
    def on_init(self, context:CtaContext):
        print(self.__codes__)
        self.xxx = context.user_load_data('xxx',1)
        self.chan_snapshot: Dict[str, CChan] = {}
        for idx, code in enumerate(self.__codes__):
            if self.__are_stks__[idx]:
                self.theCodes.append(code + "-")   # +表示后复权，-表示前复权
            else:
                self.theCodes.append(code)
            # register main k_bar for on_bar(), set bar cache
            context.stra_prepare_bars(code, self.__period__, self.__bar_cnt__, isMain = True)
            self.last_price[code] = 0
            self.resample_buffer[code] = []
            self.chan_snapshot[code] = CChan(
                code=code,
                # begin_time=begin_time,
                # end_time=end_time,
                # data_src=data_src,
                lv_list=self.lv_list,
                config=self.config,
                # autype=AUTYPE.QFQ,
                )
        context.stra_log_text("Chan Initiated")
        
    # if you have time sensitive processing (e.g. stop_loss / HFT pattern)
    def on_tick(self, context: CtaContext, stdCode: str, newTick: dict):
        print('on_tick: ', newTick)
        pass
    
    def on_bar(self, context:CtaContext, stdCode:str, period:str, newBar:dict): # deprecated?
        print('on_bar: ', newBar)
        pass
    
    def on_calculate(self, context:CtaContext):
        self.barnum += 1 # all sub-ed bars closed (main/non-main) at this period
        curTime     = context.stra_get_time()
        date        = context.get_date()
        new_date    = False
        rebalance   = False
        
        if self.date != date: # new date
            if self.date != 0: # not the first day (data ready)
                rebalance = True
            new_date    = True
            self.date = date
        else:
            return
        for idx, code in enumerate(self.__codes__):
            theCode = self.theCodes[idx]
            
            # sInfo = context.stra_get_sessioninfo(self.theCode)
            # pInfo = context.stra_get_comminfo(self.theCode)
            # if not sInfo.isInTradingTime(curTime): # type: ignore
            #     continue
            np_bars = context.stra_get_bars(code, self.__period__, self.__bar_cnt__, isMain=False)
            capital = self.__capital__
            
            open    = np_bars.opens[-1]
            high    = np_bars.highs[-1]
            low     = np_bars.lows[-1]
            close   = np_bars.closes[-1]
            volume  = np_bars.volumes[-1]
            bartime = np_bars.bartimes[-1]
            
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
            if new_date:
                if rebalance:
                    combined_klu = self.combine_klu(self.resample_buffer[code])
                    Ctime = combined_klu.time
                self.resample_buffer[code] = []
            
            if rebalance:
                # feed & calculate
                chan_snapshot = self.chan_snapshot[code]
                chan_snapshot.trigger_load({self.lv_list[0]: [combined_klu]}) # feed day bar
                bsp_list = chan_snapshot.get_bsp()

                if not bsp_list:
                    continue
                last_bsp = bsp_list[-1]
                t = last_bsp.type
                T = [0,0,0]
                if BSP_TYPE.T1 in t or BSP_TYPE.T1P in t:
                    self.num_bsp_T1 += idx==0; T[0] = 1
                if BSP_TYPE.T2 in t or BSP_TYPE.T2S in t:
                    self.num_bsp_T2 += idx==0; T[1] = 1
                if BSP_TYPE.T3A in t or BSP_TYPE.T3B in t:
                    self.num_bsp_T3 += idx==0; T[2] = 1

                cur_lv_kline = chan_snapshot[0] # __getitem__: return Kline list of level n
                metrics = cur_lv_kline.metric_model_lst
                if last_bsp.klu.klc.idx != cur_lv_kline[-2].idx:
                    continue

                T_sum = 0
                for idx, t in enumerate(T):
                    T_sum += t * (idx+1)

                top = False; bottom = False
                if cur_lv_kline[-2].fx == FX_TYPE.BOTTOM and last_bsp.is_buy:
                    bottom = True
                    # self.config.plot_para["marker"]["markers"][Ctime] = (f'B{T_sum}', 'down', 'red')
                elif cur_lv_kline[-2].fx == FX_TYPE.TOP and not last_bsp.is_buy:
                    top = True
                    # self.config.plot_para["marker"]["markers"][Ctime] = (f'S{T_sum}', 'up', 'green')
                # note that for fine data period (e.g. 1m_bar), fx(thus bsp) of the same type would occur consecutively

                if T[0] != 1:
                    continue

                curPos = context.stra_get_position(code)
                curPrice = context.stra_get_price(code)
                self.cur_money = capital + context.stra_get_fund_data(flag=0)
                amount = 1000 # math.floor(self.cur_money/curPrice)
                
                pnl: float = 1
                color: str = default
                if top and curPos >= 0:
                    if curPos != 0:
                        pnl, color = self.pnl_cal(self.last_price[code], close, False)
                        context.stra_set_position(code, 0, 'exitlong')
                    context.stra_set_position(code, -amount, 'entershort')
                    context.stra_log_text(stdio(f"{date}-{curTime}:({code}) top    FX，enter short:{self.cur_money:2f}, pnl:{color}{pnl:2f}{default}"))
                    self.config.plot_para["marker"]["markers"][Ctime] = ('short', 'down', 'orange')
                    # self.xxx = 1
                    # context.user_save_data('xxx', self.xxx)
                    self.last_price[code] = close
                    self.check_capital()
                    continue
                elif bottom and curPos <= 0:
                    if curPos != 0:
                        pnl, color = self.pnl_cal(self.last_price[code], close, True)
                        context.stra_set_position(code, 0, 'exitshort')
                    context.stra_set_position(code, amount, 'enterlong')
                    context.stra_log_text(stdio(f"{date}-{curTime}:({code}) bottom FX, enter long :{self.cur_money:2f}, pnl:{color}{pnl:2f}{default}"))
                    self.config.plot_para["marker"]["markers"][Ctime] = ('long', 'up', 'blue')
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
    
    def on_backtest_end(self, context:CtaContext):
        print('Backtest Done, plotting ...')
        print('T1:', self.num_bsp_T1, ' T2:', self.num_bsp_T2, ' T3:', self.num_bsp_T3)
        self.config.plot_config["plot_bsp"] = False
        self.config.plot_config["plot_marker"] = True
        # self.config.plot_config["plot_mean"] = True
        self.config.plot_config["plot_eigen"] = True
        self.config.plot_config["plot_seg"] = True
        self.config.plot_para["seg"]["plot_trendline"] = True
        # print(self.config.plot_para["marker"]["markers"])
        self.chan_snapshot[self.__codes__[0]].plot(save=True, animation=False, update_conf=True, conf=self.config)
