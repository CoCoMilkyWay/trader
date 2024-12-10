from Math.Chandelier_Stop import ChandelierIndicator
from PA.PA_Pattern_Chart import conv_type
from Util.MemoryAnalyzer import MemoryAnalyzer
from Chan.KLine.KLine_Unit import CKLine_Unit
from Chan.KLine.KLine_List import CKLine_List
from Chan.Common.kbar_parser import KLineHandler
from Chan.Common.CEnum import KL_TYPE, FX_TYPE, BI_DIR
from Chan.ChanConfig import CChanConfig
from Chan.Chan import CChan
from wtpy import CtaContext
from wtpy import BaseCtaStrategy
from typing import Tuple, List, Dict, Optional
from datetime import datetime
from pprint import pprint
from tqdm import tqdm
from time import time
import pandas as pd
import numpy as np
import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../app"))

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


class Main_Cta(BaseCtaStrategy):
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
        self.lstate: Dict[str, int] = {}
        self.llong_short: Dict[str, int] = {}
        self.lmprice: Dict[str, float] = {}
        self.ltol: Dict[str, float] = {}
        self.ltprice: Dict[str, float] = {}

        # indicators
        self.chandelier_stop = ChandelierIndicator(
            length=22, atr_period=22, mult=1)

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
            self.last_price[code] = 0
            self.holds[code] = [0, 0]
            self.markers[code] = []

            # ST(Strategies): liquidity
            self.lstate[code] = 0
            self.llong_short[code] = 0
            self.lmprice[code] = 0.0
            self.ltol[code] = 0.0
            self.ltprice[code] = 0.0

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

            # process PA elements
            for lv in self.lv_list:
                self.kl_datas[code][lv].PA_Core.parse_dynamic_bi_list()

            # update indicators
            self.stop_dir = self.chandelier_stop.update(self.high, self.low, self.close)
            
            # update bsp
            self.ST_liquidity_zones(context, code)

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
                    self.last_price[code], self.close, curPos>0)
                context.stra_set_position(code, 0, text)
                self.markers[code].append(
                    (self.ts, self.close, text, 'gray'))
                self.holds[code][0] = 0
                context.stra_log_text(stdio(f"cpu:{cpu_id:2}:{self.date}-{self.time:04}:({code:>15}) {
                    FX}, {text:>10} :{self.cur_money:>12.2f}, pnl:{self.pnl_color}{self.pnl*100:>+5.2f}%{default}"))
            self.check_capital()

        if sell and curPos >= 0:
            context.stra_set_position(code, -amount, text)
            self.markers[code].append((self.ts, self.close, text, 'red'))
            self.holds[code][0] = -1
            context.stra_log_text(stdio(f"cpu:{cpu_id:2}:{self.date}-{self.time:04}:({code:>15}) {
                                  FX}, {text:>10} :{self.cur_money:>12.2f}%{default}"))
            # self.xxx = 1
            # context.user_save_data('xxx', self.xxx)
            self.last_price[code] = self.close

        elif buy and curPos <= 0:
            context.stra_set_position(code, amount, text)
            self.markers[code].append((self.ts, self.close, text, 'green'))
            self.holds[code][0] = 1
            context.stra_log_text(stdio(f"cpu:{cpu_id:2}:{self.date}-{self.time:04}:({code:>15}) {
                                  FX}, {text:>10} :{self.cur_money:>12.2f}%{default}"))
            self.last_price[code] = self.close

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
            fig = ChanPlotter().plot(self.kl_datas[code], self.markers[code])
            fig = plot_fee_grid(fig, price=self.last_price[code], rate=0.0015)
            plot_show(fig)

        return fig

    # ======================================================================
    # ===============================Strategies=============================
    # ======================================================================
    def ST_bi_align(self, code):
        dirs = []
        bsp = False
        dir = None
        for lv_idx, lv in enumerate(self.lv_list):
            bi_list = self.kl_datas[code][lv].bi_list
            if len(bi_list) == 0:
                return bsp, dir
            dir = bi_list[-1].dir
            dirs.append(dir)
        # you may tempt to do long = [long, long, long, ...] from high to low level
        # actually long = [short, long, long, ...]
        align = (len(set(dirs[1:])) == 1) and (dirs[0] != dirs[1])
        if align and not self.align:
            bsp = True
        else:
            bsp = False
        self.align = align
        return bsp, dir

    def ST_MW_bsp123(self, code):
        """bsp type 1/2/3 on M/W shaped bi"""
        for lv in self.lv_list[-2:]:  # focus on lower levels
            exist, static_shapes = self.kl_datas[code][lv].PA_Core.get_static_shapes(
                potential=True)
            if exist:
                for shape in static_shapes['conv_type']:
                    # print(self.close, shape.top_y[-1], shape.bot_y[-1])
                    if self.close > shape.top_y[-1] > self.close*0.9:
                        # bi up
                        return True, True, False
                    elif self.close < shape.bot_y[-1] < self.close*1.1:
                        # bi down
                        return True, False, True

    def ST_liquidity_zones(self, context: CtaContext, code: str):  # -> buy, sell, clear
        """bsp type liquidity zones"""
        # self.lstate:
        # 0: searching (price flowing in direction of least resistance)
        # 1: sup/res triggered
        # 2: pull back away enough distance from res/sup
        # 3: place order, use 1:1 mini stop-loss and take-profit (wait till direction is clear)
        # 4: if not stopped, use further algorithmic take-profit, also rise stop-loss to make sure exiting with no loss
        FEE = 0.002
        PNL = 5
        def clear():
            self.lstate[code] = 0  # FSM
            self.llong_short[code] = 0  # long or short
            self.lmprice[code] = 0.0  # mark price when FX is generated
            self.ltol[code] = 0.0  # mark tolerance when FX is generated
            self.ltprice[code] = 0.0  # trade price when order is placed

        if self.lstate[code] == 0:
            unformed_bi_dy = 0
            PA = self.kl_datas[code][KL_TYPE.K_15M].PA_Core
            if len(PA.bi_list) > 1:
                unformed_bi_dy = abs(self.close - PA.bi_list[-2].get_end_val())
            tol = unformed_bi_dy * 0.1
            sup, res, depth = PA.PA_Liquidity.check_sup_res(self.close, tol)
            kl_list = self.kl_datas[code][self.lv_list[-1]]
            if sup and kl_list.fx == FX_TYPE.BOTTOM:
                self.llong_short[code] = 1
                self.lmprice[code] = kl_list.lst[-2].low
            elif res and kl_list.fx == FX_TYPE.TOP:
                self.llong_short[code] = -1
                self.lmprice[code] = kl_list.lst[-2].high
            if self.llong_short[code] != 0:
                self.lstate[code] = 1
                self.ltol[code] = max(tol, depth)
        if self.lstate[code] == 1:  # if a single bar is strong enough, order immediately
            if self.close > self.lmprice[code] + self.ltol[code]:
                if self.llong_short[code] == 1:
                    self.trade_order(context, code, True, False,
                                     False, 'long')  # buy
                    self.lstate[code] = 2
                    self.ltprice[code] = self.close
                else:
                    clear()
            elif self.close < self.lmprice[code] - self.ltol[code]:
                if self.llong_short[code] == -1:
                    self.trade_order(context, code, False, True,
                                     False, 'short')  # sell
                    self.lstate[code] = 2
                    self.ltprice[code] = self.close
                else:
                    clear()
        elif self.lstate[code] == 2:
            if self.llong_short[code] == 1:
                if self.close < self.ltprice[code] - self.ltol[code]:
                    self.trade_order(context, code, False,
                                     False, True, 'stop')  # clear
                    clear()
                elif self.close > self.ltprice[code] + PNL*self.ltol[code]:
                    self.lstate[code] = 3
            elif self.llong_short[code] == -1:
                if self.close > self.ltprice[code] + self.ltol[code]:
                    self.trade_order(context, code, False,
                                     False, True, 'stop')  # clear
                    clear()
                elif self.close < self.ltprice[code] - PNL*self.ltol[code]:
                    self.lstate[code] = 3
        elif self.lstate[code] == 3:
            if self.llong_short[code] == 1:
                if self.close < self.ltprice[code] + PNL*self.ltol[code]:
                    self.trade_order(context, code, False,
                                     False, True, 'pnl-tp')  # clear
                    clear()
                else:  # algo take-profit
                    if self.stop_dir == -1:
                        self.trade_order(context, code, False, False, True, 'algo-tp')
                        clear()
            elif self.llong_short[code] == -1:
                if self.close > self.ltprice[code] - PNL*self.ltol[code]:
                  # stop loss
                    self.trade_order(context, code, False,
                                     False, True, 'pnl-tp')  # clear
                    clear()
                else:  # algo take-profit
                    if self.stop_dir == 1:
                        self.trade_order(context, code, False, False, True, 'algo-tp')
                        clear()
