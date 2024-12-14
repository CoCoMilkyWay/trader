import os
import sys
import math
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from pprint import pprint
from datetime import datetime
from typing import Tuple, List, Dict, Optional

from wtpy import CtaContext
from wtpy import BaseCtaStrategy

from Chan.Chan import CChan
from Chan.ChanConfig import CChanConfig
from Chan.Common.CEnum import KL_TYPE, FX_TYPE, BI_DIR
from Chan.Common.kbar_parser import KLineHandler
from Chan.KLine.KLine_List import CKLine_List

from PA.PA_Pattern_Chart import conv_type
from Util.MemoryAnalyzer import MemoryAnalyzer
from Math.Chandelier_Stop import ChandelierIndicator
from Math.Parabolic_SAR_Stop import ParabolicSARIndicator
from Math.Mini_Entry_Pattern import Mini_Entry_Pattern
from config.cfg_cpt import cfg_cpt

sys.path.append(os.path.join(os.path.dirname(__file__), "../../app"))

# CTA = EA = BOT
# CTA: commodity trading advisor
# EA: MT-4 bot

red = "\033[31m"
green = "\033[32m"
yellow = "\033[33m"
default = "\033[0m"

CHECK_SYNC = True
MEM_ANALYZE = False


def stdio(str):
    print(str)
    return str


def pause():
    import time
    time.sleep(1000)
    return


class Main_Train(BaseCtaStrategy):
    def __init__(self, name: str, codes: List[str], period: str, capital: float):
        BaseCtaStrategy.__init__(self, name)
        self.__period__ = period
        self.__codes__ = codes
        self.__capital__ = capital

        self.config: CChanConfig = CChanConfig()
        self.KLineHandler: Dict[str, KLineHandler] = {}
        self.kl_datas: Dict[str, Dict[KL_TYPE, CKLine_List]] = {}

        # stats
        self.barnum = 0
        self.last_price: Dict[str, float] = {}
        self.last_ts: Dict[str, float] = {}
        self.cur_money = capital
        self.start_time = time()
        self.date = None
        self.pnl: float = 0
        self.pnl_color: str = default

        # models
        self.chan_snapshot: Dict[str, CChan] = {}

        # factors

        # bsp:
        # (1: long, -1: short, 0: no hold, position open period)
        self.holds: Dict[str, List[int]] = {}
        self.markers: Dict[str, List[Tuple]] = {}

        # ST(Strategies): align
        self.align = False
        # ST(Strategies): liquidity
        self.ST_signals: Dict[str, List[List]] = {}  # multiple signals
        self.ST_trade: Dict[str, List] = {}  # 1 on-going trades at a time

        # indicators
        self.chandelier_stop: Dict[str, ChandelierIndicator] = {}
        self.parabola_sar_stop: Dict[str, ParabolicSARIndicator] = {}
        self.mini_entry_pattern: Dict[str, Mini_Entry_Pattern] = {}

        # debug
        if cfg_cpt.dump_ind:
            self.ind_ts = []
            self.ind_value = []
            self.ind_text = []

    def on_init(self, context: CtaContext):
        print('Initializing Strategy...')
        self.lv_list = [lv[0] for lv in self.config.lv_list]

        for idx, code in enumerate(self.__codes__):
            context.stra_prepare_bars(
                code, self.__period__, 1, isMain=idx == 0)
            # only 1 series is registered as 'Main', which works as clock, more registrations would result in fault
            # on_calculate is triggered once main bar is closed
            # if hook is installed, on_calculate_done would be triggered(for RL)

            self.KLineHandler[code] = KLineHandler(self.config.lv_list)
            self.init_shared_kl_datas(code)
            self.last_price[code] = 0.0
            self.last_ts[code] = 0.0
            self.holds[code] = [0, 0]
            self.markers[code] = []

            # Indicators
            # conservative stop(use earlier)
            self.chandelier_stop[code] = ChandelierIndicator(
                length=22, atr_period=2, mult=3)
            
        context.stra_log_text(stdio("Strategy Initiated"))
        self.pbar = tqdm(total=len(self.__codes__),
                         desc='Preparing Bars in DDR...')

    def init_shared_kl_datas(self, code: str):
        """Initialize K-line data structures for each time level."""
        self.kl_datas[code] = {level: CKLine_List(
            level, conf=self.config)for level in self.lv_list}

    def init_new_code(self, code: str):
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

    def on_calculate(self, context: CtaContext):
        # all sub-ed bars closed (main/non-main) at this period
        self.barnum += 1

        self.date = context.get_date()
        self.time = context.stra_get_time()
        # if date!=self.date:
        #     print(date)
        #     # print(date, time)
        #     self.date = date

        for idx, code in enumerate(self.__codes__):
            if self.barnum == 1:
                self.pbar.update(1)
                self.init_new_code(code)
            else:
                self.pbar.close()

            np_bars = context.stra_get_bars(
                code, self.__period__, 1, isMain=idx == 0)
            self.open = np_bars.opens[-1]
            self.high = np_bars.highs[-1]
            self.low = np_bars.lows[-1]
            self.close = np_bars.closes[-1]
            # multi-level k bar generation
            klu_dict = self.KLineHandler[code].process_bar(np_bars)
            self.ts = klu_dict[KL_TYPE.K_1M][-1].time.ts

            # process Chan elements (generates Bi)
            self.chan_snapshot[code].trigger_load(klu_dict)

            # update indicators
            self.longcs, self.shortcs = self.chandelier_stop[code].update(
                self.high, self.low, self.close, self.ts)

    def trade_order(self, context: CtaContext, code: str, buy: bool, sell: bool, clear: bool, text: str):
        cpu_id = 0
        curPos = context.stra_get_position(code)
        curPrice = context.stra_get_price(code)
        self.cur_money = self.__capital__ + context.stra_get_fund_data(flag=0)
        # print(code, curPos, curPrice, self.cur_money)
        amount = math.floor(self.cur_money/curPrice /
                            (len(self.__codes__)))  # equal position
        FX = '^' if sell else 'v' if buy else '-'
        if clear:
            if curPos != 0:
                self.pnl, self.pnl_color = self.pnl_cal(
                    self.last_price[code], self.close, curPos > 0)
                hold_hours = round(abs(self.ts - self.last_ts[code]) / 3600, 2)
                context.stra_set_position(code, 0, text)
                self.markers[code].append(
                    (self.ts, self.close, text, 'gray'))
                self.holds[code][0] = 0
                context.stra_log_text(stdio(f"cpu:{cpu_id:2}:{self.date}-{self.time:04}:({code:>15}) {FX}, {text:>10}:{
                                      self.cur_money:>12.2f}, pnl:{self.pnl_color}{self.pnl*100:>+5.2f}{default}%, hold:{hold_hours:>3.2f}hrs"))
            self.check_capital()

        if sell and curPos >= 0:
            context.stra_set_position(code, -amount, text)
            self.markers[code].append((self.ts, self.close, text, 'red'))
            self.holds[code][0] = -1
            context.stra_log_text(stdio(f"cpu:{cpu_id:2}:{self.date}-{self.time:04}:({code:>15}) {
                                  FX}, {text:>10} :{self.cur_money:>12.2f}{default}"))
            # self.xxx = 1
            # context.user_save_data('xxx', self.xxx)
            self.last_price[code] = self.close
            self.last_ts[code] = self.ts

        elif buy and curPos <= 0:
            context.stra_set_position(code, amount, text)
            self.markers[code].append((self.ts, self.close, text, 'green'))
            self.holds[code][0] = 1
            context.stra_log_text(stdio(f"cpu:{cpu_id:2}:{self.date}-{self.time:04}:({code:>15}) {
                                  FX}, {text:>10} :{self.cur_money:>12.2f}{default}"))
            self.last_price[code] = self.close
            self.last_ts[code] = self.ts

    def check_capital(self):
        try:
            assert self.cur_money > 0
        except AssertionError:
            print(f"lost, stopping ...")
            os._exit(1)

    def pnl_cal(self, last_price: float, price: float, long_short: bool):
        if long_short:
            dir = 1
        else:
            dir = -1
        pnl = dir*(price - last_price)/last_price
        if pnl > 0.01:
            color = green
        elif pnl < -0.01:
            color = red
        else:
            color = default
        return pnl, color

    def profile(self, date):
        try:
            num = 1  # self.barnum-self.barnum_bak
            print(f'({date}): {num/(time()-self.time):.2f} days/sec')
        except:
            pass
        self.time = time()
        self.barnum_bak = self.barnum

    def on_backtest_end(self, context: CtaContext):
        self.elapsed_time = time() - self.start_time
        print(f'main BT loop time elapsed: {self.elapsed_time:2f}s')

        if MEM_ANALYZE:
            chan = self.chan_snapshot[self.__codes__[0]]
            MemoryAnalyzer().analyze_object(chan)
            # MemoryAnalyzer().analyze_object(list(chan.kl_datas.items())[-1][1])
            for obj in list(chan.kl_datas.items()):
                size = MemoryAnalyzer().get_deep_size(obj)
                print(f'{size/1000/1000:3.2f}MB: {obj}')

        from Chan.Plot.PlotDriver import ChanPlotter
        from Util.plot.plot_fee_grid import plot_fee_grid
        from Util.plot.plot_show import plot_show

        for code in self.__codes__:
            indicators = [
                self.chandelier_stop[code],
                self.parabola_sar_stop[code],
                self.ind_ts,
            ]
            if cfg_cpt.dump_ind:
                indicators.extend([
                    self.ind_value,
                    self.ind_text,
                ])
            fig = ChanPlotter().plot(
                self.kl_datas[code], self.markers[code], indicators)
            fig = plot_fee_grid(fig, dtick=self.last_price[code]*0.0015)
            plot_show(fig)

