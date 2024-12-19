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
from Util.UtilCpt import mkdir
from Util.MemoryAnalyzer import MemoryAnalyzer

from Util.BiLTSM_Pattern_Recog import PatternRecognizer

from Math.Chandelier_Stop import ChandelierIndicator
from Math.ChandeKroll_Stop import ChandeKrollStop
from Math.Parabolic_SAR_Stop import ParabolicSARIndicator
from Math.Adaptive_SuperTrend import AdaptiveSuperTrend
from Math.LorentzianClassifier import LorentzianClassifier

from Math.VolumeWeightedBands import VolumeWeightedBands
from Math.Mini_Entry_Pattern import Mini_Entry_Pattern

from config.cfg_cpt import cfg_cpt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
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

ML_Pattern_Len = 10

class Main_CTA(BaseCtaStrategy):
    def __init__(self, name: str, codes: List[str], period: str, capital: float, plot:bool, train:bool):
        BaseCtaStrategy.__init__(self, name)
        self.__period__ = period
        self.__codes__ = codes
        self.__capital__ = capital
        self.__plot__ = plot
        self.__train__ = train

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
        
        # ML_models
        if not self.__train__:
            self.recognizer = PatternRecognizer(
                sequence_length=ML_Pattern_Len, 
                shape_features=2, 
                additional_features=3,
                price_normalization='standard',
                additional_features_normalization='standard'
            )
            self.recognizer.load_model(cfg_cpt.model_path)
            self.recognizer.compile_model()

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
        self.chandekroll_stop: Dict[str, ChandeKrollStop] = {}
        self.parabola_sar_stop: Dict[str, ParabolicSARIndicator] = {}
        self.adaptive_supertrend: Dict[str, AdaptiveSuperTrend] = {}
        self.lorentzian_classifier: Dict[str, LorentzianClassifier] = {}
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
            # 1M (short memory indicators, which would still work for 5hrs trend/patterns because of algo)
            self.chandelier_stop[code] = ChandelierIndicator(length=300, atr_period=2, mult=0.2)
            self.chandekroll_stop[code] = ChandeKrollStop(atr_length=10, atr_coef=1, stop_len=9)
            self.parabola_sar_stop[code] = ParabolicSARIndicator(acceleration=0.001, max_acceleration=0.005, initial_acceleration=0)
            # 5M
            self.adaptive_supertrend[code] = AdaptiveSuperTrend(atr_len=100, factor=7, lookback=100)
            self.lorentzian_classifier[code] = LorentzianClassifier()
            # 15M
            self.volume_weighted_bands[code] = VolumeWeightedBands(window_size=int(2000/15), window_size_atr=int(240/15)) # match with 2hrs median holding time(based on our research)
            self.mini_entry_pattern[code] = Mini_Entry_Pattern(self.kl_datas[code][KL_TYPE.K_1M].bi_list)

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
            K = K = klu_dict[KL_TYPE.K_1M][-1]
            self.ts = K.time.ts

            # process Chan elements (generates Bi)
            self.chan_snapshot[code].trigger_load(klu_dict)

            # process PA elements
            for lv in self.lv_list:
                self.kl_datas[code][lv].PA_Core.parse_dynamic_bi_list()

            # update indicators
            self.c_long_switch, self.c_short_switch = self.chandelier_stop[code].update(K.high, K.low, K.close, self.ts)
            self.k_long_switch, self.k_short_switch = self.chandekroll_stop[code].update(K.high, K.low, K.close, self.ts)
            self.p_long_switch, self.p_short_switch = self.parabola_sar_stop[code].update(K.high, K.low, self.ts)
            
            if KL_TYPE.K_5M in klu_dict:
                K = klu_dict[KL_TYPE.K_5M][-1]
                self.s_long_switch, self.s_short_switch = self.adaptive_supertrend[code].update(K.high, K.low, K.close, self.ts,)
                self.l_long_switch, self.l_short_switch = self.lorentzian_classifier[code].update(K.high, K.low, K.close, K.volume)
                
            if KL_TYPE.K_15M in klu_dict:
                K = klu_dict[KL_TYPE.K_15M][-1]
                [self.vwap, self.dev_mult, _, _] = self.volume_weighted_bands[code].update(K.high, K.low, K.close, K.volume, self.ts,)
                
            # update bsp
            self.long_switch = self.c_long_switch
            self.short_switch = self.c_short_switch
            
            # indicator guard
            if self.barnum < 2*24*60: # need 2 days of 1M data to prepare indicators
                return
            
            # strategy analysis
            if self.__train__:
                self.ST_Train(context, code)
            else:
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
        if self.__plot__:
            from Chan.Plot.PlotDriver import ChanPlotter
            from Util.plot.plot_fee_grid import plot_fee_grid
            from Util.plot.plot_show import plot_show
            
            for code in self.__codes__:
                indicators = [
                    self.chandelier_stop[code],
                    self.chandekroll_stop[code],
                    self.parabola_sar_stop[code],
                    self.adaptive_supertrend[code],
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
        
        if self.__train__:                
            # Generate some test data
            print('Generating Traning data')
            for code in self.__codes__:
                train_sequences = []
                train_features  = []
                train_labels    = []
                for signal in tqdm(self.ST_signals[code]):
                    signal_state = signal[0]
                    bi_pattern = signal[1]
                    long_short = signal[2]
                    elasticity = signal[3]
                    entry_price = signal[4]
                    label = signal[5]
                    if signal[0] == 1:
                        train_sequences.append(bi_pattern)
                        train_features.append([
                            int(long_short),
                            elasticity,
                            entry_price,
                        ])
                        train_labels.append(label)

                train_sequences = np.array(train_sequences)
                train_features  = np.array(train_features )
                train_labels    = np.array(train_labels   )
                
                # Initialize and train model
                recognizer = PatternRecognizer(
                    sequence_length=ML_Pattern_Len, 
                    shape_features=2, 
                    additional_features=len(train_features[-1]),
                    price_normalization='standard',
                    additional_features_normalization='standard'
                )
                history = recognizer.train(
                    train_sequences, 
                    train_features, 
                    train_labels,
                    epochs=50, 
                    batch_size=32
                )
                recognizer.compile_model()
                
                recognizer.save_model(mkdir(cfg_cpt.model_path))
                
                # Make predictions
                predictions = recognizer.predict(train_sequences, train_features)

                # Print results
                for i, (pred, true_label) in enumerate(zip(predictions, train_labels)):
                    print(f"Sample {i}: Prediction={pred[0]:.2f}, True Label={true_label}")

    def ST_SR_reversal(self, context: CtaContext, code: str):  # -> buy, sell, clear
        """
        Strategy: Support & Resistance Reversal
            15M: S&R analysis
            1M: entry pattern
        """
        # NOTE: we are doing reversal here
        # when do you expect price to reverse?
        # yes, when price deviate from MA and momentum vanishes
        # better, if this happens around S&R regions
        # entry? when we see some mini-structures forming in lower level candles (e.g. 1M)

        # ST_signals: [[signal_state, long_short, targets, m1_bi_index, entry_price, pullback, type], ...]
        # ST_trade: [exec_state, long_short, trade_price, targets, type]
        PA = self.kl_datas[code][KL_TYPE.K_15M].PA_Core
        bi_list_m1 = self.kl_datas[code][KL_TYPE.K_1M].bi_list
        bi_list_m15 = self.kl_datas[code][KL_TYPE.K_15M].bi_list
        kl_list = self.kl_datas[code][self.lv_list[-1]]
        if len(bi_list_m15) < 3:  # make sure we have both sup and res
            return
        if self.ST_trade[code] == []:  # no ongoing trade
            # get new potential trade
            #   1. premium region of a relatively strong zone @t1
            #   2. entry mini-pattern formed (trend is not strong) @t1
            match, lower_targets, upper_targets, hint = \
            PA.PA_Liquidity.check_sup_res(self.close)
            
            found = False
            if match:
                long_short =  self.vwap > self.close
                if long_short:
                    targets = upper_targets
                else:
                    targets = lower_targets
                self.ind_ts.append(bi_list_m1[-1].get_end_klu().time.ts)
                self.ind_value.append(self.close)
                self.ind_text.append((long_short,hint))

                if len(targets) > 0 and bi_list_m1[-1].is_sure:
                    # found, idx_list, type, entry_price, pullback = self.mini_entry_pattern[code].check_patterns(long_short)
                    long_short =  self.vwap > self.close
                    elasticity =  self.dev_mult
                    sequence = np.array([(bi.get_klu_cnt(), bi.get_end_val()) for bi in bi_list_m1[-ML_Pattern_Len:]])
                    feature = np.array([
                        long_short,
                        elasticity,
                        self.close,
                    ])
                    # sequence = sequence.reshape(1, *sequence.shape)
                    # feature = feature.reshape(1, -1)
                    label = self.recognizer.predict_single_sample(sequence, feature)
                    if label > 0.51:
                        self.ST_signals[code].append(
                            [0, long_short, targets, self.close, '',]
                        )
                        if cfg_cpt.dump_ind:
                            idx = len(self.ind_text)
                            self.ST_signals[code][-1].append(idx)
                            self.ind_ts.append([bi.get_end_klu().time.ts for bi in bi_list_m1[-6:]])
                            self.ind_value.append([bi.get_end_val() for bi in bi_list_m1[-6:]])
                            self.ind_text.append((long_short,''))

            # confirm potential trades
            #   3. pull-away from FX in the right direction to avoid fake bi(FX) @t3
            #   4. valid candle pattern(FX strength) @t3
            #   5. valid volume character @t3
            if len(self.ST_signals[code]) > 0:
                for signal in self.ST_signals[code][:]:
                    signal_state = signal[0]
                    long_short = signal[1]
                    targets = signal[2]
                    entry_price = signal[3]
                    type = signal[4]
                    # if found and (m1_bi_index not in idx_list):
                    #     # signal is even out of pattern checking range, remove
                    #     self.ST_signals[code].remove(signal)
                    #     continue
                    if signal_state == 0:
                        if cfg_cpt.dump_ind:
                            idx = signal[-1]

                        # # pull-away safety check
                        # #if abs(m1_bi_index-m1_bi_list[-1].idx) > 1:
                        # if m1_bi_index!=bi_list_m1[-1].idx:
                        #     # not pulling back in time
                        #     self.ST_signals[code].remove(signal)
                        #     continue
                        # pull_away_check = False
                        # if long_short and self.close > (entry_price + pullback):
                        #     pull_away_check = True
                        # elif not long_short and self.close < (entry_price - pullback):
                        #     pull_away_check = True
                        # if not pull_away_check:
                        #     continue
                        # if cfg_cpt.dump_ind:
                        #     self.ind_text[idx] += '1'

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
            break_even_high = self.close > trade_price*(1+cfg_cpt.FEE)
            break_even_low = self.close < trade_price*(1-cfg_cpt.FEE)
            if state == 0:
                if long_short:
                    self.trade_order(context, code, True, False, False, f'long({type})')  # buy
                else:
                    self.trade_order(context, code, False, True, False, f'short({type})')  # sell
                self.ST_trade[code][0] = 1
            elif state == 1:
                stop = False
                if long_short:
                    # if break_even_high:
                    #     self.ST_trade[code][0] = 2
                    if self.short_switch:
                        stop = True
                elif not long_short:
                    # if break_even_low:
                    #     self.ST_trade[code][0] = 2
                    if self.long_switch:
                        stop = True
                if stop:
                    #self.trade_order(context, code, False,False, True, 'stop')  # clear
                    self.trade_order(context, code, False, False, True, 'algo-tp')  # clear
                    self.ST_trade[code] = []  # exit trade
            # elif state == 2:
            #     stop = False
            #     if long_short:
            #         if not break_even_high:
            #             self.trade_order(context, code, False, False, True, 'even')  # clear
            #         elif self.short_switch:
            #             stop = True
            #     elif not long_short:
            #         if not break_even_low:
            #             self.trade_order(context, code, False, False, True, 'even')  # clear
            #         elif self.long_switch:
            #             stop = True
            #     if stop:
            #         self.trade_order(context, code, False, False, True, 'algo-tp')  # clear
            #         self.ST_trade[code] = []  # exit trade

    def ST_Train(self, context: CtaContext, code: str):
        """
        Strategy: Training ML model
        """
        PA = self.kl_datas[code][KL_TYPE.K_15M].PA_Core
        bi_list_m1 = self.kl_datas[code][KL_TYPE.K_1M].PA_Core.bi_list
        bi_list_m15 = self.kl_datas[code][KL_TYPE.K_15M].PA_Core.bi_list
        kl_list = self.kl_datas[code][self.lv_list[-1]]
        if len(bi_list_m15) < 3:  # make sure we have both sup and res
            return

        match, lower_targets, upper_targets, hint = \
        PA.PA_Liquidity.check_sup_res(self.close)
        
        if match:
            long_short =  self.vwap > self.close
            elasticity =  self.dev_mult
            if long_short:
                targets = upper_targets
            else:
                targets = lower_targets
            if len(targets) > 0: # this is usually not a problem, just to be safe
                if bi_list_m1[-1].is_sure:
                    self.ST_signals[code].append([
                        # status
                        0, # 0: opened 1: closed with feature and label
                        
                        # features
                        [(bi.get_klu_cnt(), bi.get_end_val()) for bi in bi_list_m1[-ML_Pattern_Len:]], # pattern
                        long_short, # dir of trade (long/short properties may be asymmetric)
                        elasticity,
                        self.close, # entry_price
                        # TODO: e.g. dst to 1st OB, trend, etc
                        
                        # label(pnl)
                        0.0,
                        ])

            for signal in self.ST_signals[code][:]:
                signal_state = signal[0]
                bi_pattern = signal[1]
                long_short = signal[2]
                elasticity = signal[3]
                entry_price = signal[4]
                label = signal[5]

                stop = False
                dir = 0
                if signal_state == 0:
                    if long_short:
                        if self.short_switch:
                            stop = True
                            dir = 1
                    elif not long_short:
                        if self.long_switch:
                            stop = True
                            dir = -1
                    if stop:
                        signal[0] = 1
                        pnl = dir*(self.close - entry_price)/entry_price
                        pnl = pnl + 0.5 # [-1, 0 ,1] -> [-0.5, 0.5, 1.5]
                        if pnl < 0:
                            pnl = 0
                        elif pnl > 1:
                            pnl = 1
                        signal[5] = pnl
                        if signal[5] > 0.01:
                            symbol = "v" if long_short else "^"
                            print(f'{self.date}-{self.time:>4} {symbol} label:{signal[5]:>3.2f}')