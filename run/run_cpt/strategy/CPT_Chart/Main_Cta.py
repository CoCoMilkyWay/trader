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
from Math.VolumeWeightedBands import VolumeWeightedBands
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

"""
in Market lacking Smooth Trends: we do 2 types of trade
1. Reversal in Ranging Market conditions
2. Breakout Chasing when fast price movement happens

Here we do 1:

Zones:
1. Identify key S&R zones @15M, Trend-lines etc.

Entry:
1. confirm zone reaction
2. high potential long/short mini shapes forming @1M 
    (head & shoulder, M/W, 3 point trend-line etc.)
3. wait for 1M bar to stabilize (usually formed just right to the new extreme value)
    (using VWAP, BollingerBand, and atr to confirm volatility squeeze)
    (or simply confirm pullback to establish the extreme position we want to entry(reversal))

"""

"""
Stops:
Chandelier Stop
Parabolic SAR
Adaptive SuperTrend(K-means)
Anchored VWAP
"""

"""
For the green triangles (upward breakouts), I'd guess the conditions might be:

Price is likely near/below VWAP (oversold condition)
Price is probably near/below lower Bollinger Band
ATR is likely expanding (increasing volatility)

Specific trigger conditions might be:

Price crosses above VWAP while being near/below lower BB
AND ATR is expanding above its short-term average
AND Bollinger Bands width is starting to expand after being narrow (squeeze ending)

For purple triangles (downward breakouts), the reverse logic:

Price crosses below VWAP while being near/above upper BB
AND ATR expanding
AND BB width expanding after squeeze

The key insight is that these triangles appear to signal breakout trades after periods of consolidation:

First wait for BB squeeze (narrow bands)
Then look for price crossing VWAP in the direction of potential breakout
Confirm with expanding ATR showing increasing momentum
"""
class Main_CTA(BaseCtaStrategy):
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
        self.volume_weighted_bands: Dict[str, VolumeWeightedBands] = {}
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
            # aggressive stop(use latter)
            self.parabola_sar_stop[code] = ParabolicSARIndicator(
                acceleration=0.002, max_acceleration=0.02)
            self.volume_weighted_bands[code] = VolumeWeightedBands(window_size=60*4)
            self.mini_entry_pattern[code] = Mini_Entry_Pattern(
                self.kl_datas[code][KL_TYPE.K_1M].bi_list)

            # ST(Strategies): liquidity
            self.ST_signals[code] = []
            self.ST_trade[code] = []

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
            self.volume = np_bars.volumes[-1]
            # multi-level k bar generation
            klu_dict = self.KLineHandler[code].process_bar(np_bars)
            self.ts = klu_dict[KL_TYPE.K_1M][-1].time.ts

            # process Chan elements (generates Bi)
            self.chan_snapshot[code].trigger_load(klu_dict)

            # process PA elements
            for lv in self.lv_list:
                self.kl_datas[code][lv].PA_Core.parse_dynamic_bi_list()

            # update indicators
            self.longcs, self.shortcs = self.chandelier_stop[code].update(
                self.high, self.low, self.close, self.ts)
            self.longshortps = self.parabola_sar_stop[code].update(
                self.high, self.low, self.ts)
            self.volume_weighted_bands[code].update(
                self.high, self.low, self.close, self.volume, self.ts)

            # update bsp
            self.ST_SR_reversal(context, code)

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
                self.volume_weighted_bands[code],
                self.ind_ts,
            ]
            if cfg_cpt.dump_ind:
                indicators.extend([
                    self.ind_value,
                    self.ind_text,
                ])
            fig = ChanPlotter().plot(
                self.kl_datas[code], self.markers[code], indicators)
            fig = plot_fee_grid(fig, dtick=self.last_price[code]*cfg_cpt.FEE)
            plot_show(fig)

    def ST_SR_reversal(self, context: CtaContext, code: str):  # -> buy, sell, clear
        """
        Strategy: Support & Resistance Reversal
            15M: S&R analysis
            1M: entry pattern
        """
        # NOTE: we are doing reversal here
        # when do you expect price to reverse at S&R regions?
        # yes, when trend is not strong, market is ranging, which is what happens most of times
        # thus we expect to see some mini-structures forming in lower level candles (e.g. 1M)

        # ST_signals: [[signal_state, long_short, targets, m1_bi_index, entry_price, pullback, type], ...]
        # ST_trade: [exec_state, long_short, trade_price, targets, type]
        PA = self.kl_datas[code][KL_TYPE.K_15M].PA_Core
        m1_bi_list = self.kl_datas[code][KL_TYPE.K_1M].PA_Core.bi_list
        bi_list = PA.bi_list
        kl_list = self.kl_datas[code][self.lv_list[-1]]
        if len(bi_list) < 3:  # make sure we have both sup and res
            return
        if self.ST_trade[code] == []:  # no ongoing trade
            # get new potential trade
            #   1. premium region of a relatively strong zone @t1
            #   2. entry mini-pattern formed (trend is not strong) @t1
            match, long_short, lower_targets, upper_targets, hint = \
            PA.PA_Liquidity.check_sup_res(self.close)
            
            found = False
            if match:
                if long_short:
                    targets = upper_targets
                else:
                    targets = lower_targets
                self.ind_ts.append(m1_bi_list[-1].get_end_klu().time.ts)
                self.ind_value.append(self.close)
                self.ind_text.append((long_short,hint))

                if len(targets) > 0:  # this is usually not a problem, just to be safe
                    found, idx_list, type, entry_price, pullback = self.mini_entry_pattern[code].check_patterns(
                        long_short)
                    if found and m1_bi_list[-1].is_sure:
                        self.ST_signals[code].append(
                            [0, long_short, targets, idx_list[-1],
                                entry_price, pullback, '',]
                        )
                        if cfg_cpt.dump_ind:
                            idx = len(self.ind_text)
                            self.ST_signals[code][-1].append(idx)
                            self.ind_ts.append([bi.get_end_klu().time.ts for bi in m1_bi_list[-6:]])
                            self.ind_value.append([bi.get_end_val() for bi in m1_bi_list[-6:]])
                            self.ind_text.append(type)

            # confirm potential trades
            #   3. pull-away from FX in the right direction to avoid fake bi(FX) @t3
            #   4. valid candle pattern(FX strength) @t3
            #   5. valid volume character @t3
            if len(self.ST_signals[code]) > 0:

                for signal in self.ST_signals[code][:]:
                    signal_state = signal[0]
                    long_short = signal[1]
                    targets = signal[2]
                    m1_bi_index = signal[3]
                    entry_price = signal[4]
                    pullback = signal[5]
                    type = signal[6]
                    if found and (m1_bi_index not in idx_list):
                        # signal is even out of pattern checking range, remove
                        self.ST_signals[code].remove(signal)
                        continue
                    if signal_state == 0:
                        if cfg_cpt.dump_ind:
                            idx = signal[-1]

                        # pull-away safety check
                        #if abs(m1_bi_index-m1_bi_list[-1].idx) > 1:
                        if m1_bi_index!=m1_bi_list[-1].idx:
                            # not pulling back in time
                            self.ST_signals[code].remove(signal)
                            continue
                        pull_away_check = False
                        if long_short and self.close > (entry_price + pullback):
                            pull_away_check = True
                        elif not long_short and self.close < (entry_price - pullback):
                            pull_away_check = True
                        if not pull_away_check:
                            continue
                        if cfg_cpt.dump_ind:
                            self.ind_text[idx] += '1'

                        self.ST_trade[code] = [
                            0, long_short, self.close, targets, type]
                        self.ST_signals[code] = []  # clear all signals
                        break

        if self.ST_trade[code]:  # has ongoing trade
            # state =
            #   0: place order
            #   1: set 1:1 pnl with w=pullback (may lose money)
            #   2: set target to premium region of target opposite liquidity zone,
            #       also raise take_profit to at least break even
            #   3: entered target region, use algo take-profit
            state = self.ST_trade[code][0]
            long_short = self.ST_trade[code][1]
            trade_price = self.ST_trade[code][2]
            targets = self.ST_trade[code][3]
            type = self.ST_trade[code][4]
            if state == 0:
                if long_short:
                    self.trade_order(context, code, True,
                                     False, False, f'long({type})')  # buy
                else:
                    self.trade_order(context, code, False,
                                     True, False, f'short({type})')  # sell
                self.ST_trade[code][0] = 1
            elif state == 1:
                stop = False
                if long_short and self.shortcs:
                    stop = True
                elif not long_short and self.longcs:
                    stop = True
                if stop:
                    self.trade_order(context, code, False,
                                     False, True, 'algo-tp')  # clear
                    self.ST_trade[code] = []  # exit trade

    # ======================================================================
    # ===============================Strategies=============================
    # ======================================================================
    # def ST_bi_align(self, code):
    #     dirs = []
    #     bsp = False
    #     dir = None
    #     for lv_idx, lv in enumerate(self.lv_list):
    #         bi_list = self.kl_datas[code][lv].bi_list
    #         if len(bi_list) == 0:
    #             return bsp, dir
    #         dir = bi_list[-1].dir
    #         dirs.append(dir)
    #     # you may tempt to do long = [long, long, long, ...] from high to low level
    #     # actually long = [short, long, long, ...]
    #     align = (len(set(dirs[1:])) == 1) and (dirs[0] != dirs[1])
    #     if align and not self.align:
    #         bsp = True
    #     else:
    #         bsp = False
    #     self.align = align
    #     return bsp, dir
    #
    # def ST_MW_bsp123(self, code):
    #     """bsp type 1/2/3 on M/W shaped bi"""
    #     for lv in self.lv_list[-2:]:  # focus on lower levels
    #         exist, static_shapes = self.kl_datas[code][lv].PA_Core.get_static_shapes(
    #             potential=True)
    #         if exist:
    #             for shape in static_shapes['conv_type']:
    #                 # print(self.close, shape.top_y[-1], shape.bot_y[-1])
    #                 if self.close > shape.top_y[-1] > self.close*0.9:
    #                     # bi up
    #                     return True, True, False
    #                 elif self.close < shape.bot_y[-1] < self.close*1.1:
    #                     # bi down
    #                     return True, False, True
    #
    # def ST_liquidity_zones(self, context: CtaContext, code: str):  # -> buy, sell, clear
    #     """bsp type liquidity zones"""
    #     FEE = 0.002
    #     NO_FEE = 5
    #
    #     # ST_signals: [[long_short, fx_price, zone_depth, target], ...]
    #     # ST_trade: [exec_state, long_short, trade_price, pull_back]
    #
    #     PA = self.kl_datas[code][KL_TYPE.K_15M].PA_Core
    #     bi_list = PA.bi_list
    #     kl_list = self.kl_datas[code][self.lv_list[-1]]
    #     if len(bi_list) < 2:
    #         return
    #
    #     if self.ST_trade[code] == []:  # no ongoing trade
    #         # get new potential trades
    #         #   0. wait for 1M to show ChoCh
    #         #   1. FX @t1
    #         #   2. sup/res @t1
    #         #   3. swing range to 1st target > NO_FEE*FEE
    #         unformed_bi_dy = abs(self.close - PA.bi_list[-2].get_end_val())
    #         entering_strength = 0.1 * unformed_bi_dy
    #         if kl_list.fx == FX_TYPE.BOTTOM:
    #             sup, _, depth, targets = PA.PA_Liquidity.check_sup_res(
    #                 self.close, entering_strength)
    #             if sup:
    #                 if abs(targets[0] - self.close)/self.close > NO_FEE*FEE:
    #                     self.ST_signals[code].append(
    #                         [True, kl_list.lst[-2].low, depth, targets])
    #         elif kl_list.fx == FX_TYPE.TOP:
    #             _, res, depth, targets = PA.PA_Liquidity.check_sup_res(
    #                 self.close, entering_strength)
    #             if res:
    #                 if abs(self.close - targets[0])/self.close > NO_FEE*FEE:
    #                     self.ST_signals[code].append(
    #                         [False, kl_list.lst[-2].high, depth, targets])
    #
    #         # confirm potential trades
    #         #   3. pull back enough @t2
    #         #   4. valid candle pattern(FX strength) @t2
    #         #   5. valid volume character @t2
    #         for signal in self.ST_signals[code]:
    #             long_short = signal[0]
    #             fx_price = signal[1]
    #             zone_depth = signal[2]
    #             targets = signal[3]
    #             pull_back = max(entering_strength, zone_depth)
    #             if long_short and self.close > fx_price + pull_back:
    #                 # if ...
    #                 self.ST_trade[code] = [0, long_short,
    #                                        self.close, pull_back, targets]
    #                 self.ST_signals[code] = []  # clear all signals
    #                 break
    #             if not long_short and self.close < fx_price - pull_back:
    #                 # if ...
    #                 self.ST_trade[code] = [0, long_short,
    #                                        self.close, pull_back, targets]
    #                 self.ST_signals[code] = []  # clear all signals
    #                 break
    #
    #     if self.ST_trade[code]:  # has ongoing trade
    #         # state =
    #         #   0: place order
    #         #   1: set 1:1 pnl with w=pullback (may lose money)
    #         #   2: set target to premium region of target opposite liquidity zone,
    #         #       also raise take_profit to at least break even
    #         #   3: entered target region, use algo take-profit
    #         state = self.ST_trade[code][0]
    #         long_short = self.ST_trade[code][1]
    #         trade_price = self.ST_trade[code][2]
    #         pull_back = self.ST_trade[code][3]
    #         targets = self.ST_trade[code][4]
    #         if state == 0:
    #             if long_short:
    #                 self.trade_order(context, code, True,
    #                                  False, False, 'long')  # buy
    #             else:
    #                 self.trade_order(context, code, False,
    #                                  True, False, 'short')  # sell
    #             self.ST_trade[code][0] = 1
    #         elif state == 1:
    #             break_low = self.close < trade_price - pull_back
    #             break_high = self.close > trade_price + pull_back
    #             if long_short:
    #                 if break_low:
    #                     self.trade_order(context, code, False,
    #                                      False, True, 'stop')  # clear
    #                     self.ST_trade[code] = []  # exit trade
    #                 elif break_high:
    #                     self.ST_trade[code][0] = 2
    #             else:
    #                 if break_high:
    #                     self.trade_order(context, code, False,
    #                                      False, True, 'stop')  # clear
    #                     self.ST_trade[code] = []  # exit trade
    #                 elif break_low:
    #                     self.ST_trade[code][0] = 2
    #         elif state == 2:
    #             target_1st = targets[0]
    #             if long_short:
    #                 break_even_low = self.close < trade_price*(1+FEE)
    #                 break_premium_high = self.close > (
    #                     2*target_1st + trade_price)/3
    #                 if break_even_low:
    #                     self.trade_order(context, code, False,
    #                                      False, True, 'even')  # clear
    #                     self.ST_trade[code] = []  # exit trade
    #                 elif break_premium_high:
    #                     self.ST_trade[code][0] = 3
    #             else:
    #                 break_even_high = self.close > trade_price*(1-FEE)
    #                 break_premium_low = self.close < (
    #                     2*target_1st + trade_price)/3
    #                 if break_even_high:
    #                     self.trade_order(context, code, False,
    #                                      False, True, 'even')  # clear
    #                     self.ST_trade[code] = []  # exit trade
    #                 elif break_premium_low:
    #                     self.ST_trade[code][0] = 3
    #         elif state == 3:
    #             target_1st = targets[0]
    #             if long_short:
    #                 break_premium_low = self.close < (
    #                     2*target_1st + trade_price)/3
    #                 if break_premium_low:
    #                     self.trade_order(context, code, False,
    #                                      False, True, 'pnl-tp')  # clear
    #                     self.ST_trade[code] = []  # exit trade
    #                 elif self.stop_dir == -1:
    #                     self.trade_order(context, code, False,
    #                                      False, True, 'algo-tp')  # clear
    #                     self.ST_trade[code] = []  # exit trade
    #             else:
    #                 break_premium_high = self.close > (
    #                     2*target_1st + trade_price)/3
    #                 if break_premium_high:
    #                     self.trade_order(context, code, False,
    #                                      False, True, 'pnl-tp')  # clear
    #                     self.ST_trade[code] = []  # exit trade
    #                 elif self.stop_dir == 1:
    #                     self.trade_order(context, code, False,
    #                                      False, True, 'algo-tp')  # clear
    #                     self.ST_trade[code] = []  # exit trade
