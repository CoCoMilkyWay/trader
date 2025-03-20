import time
import torch
from tqdm import tqdm
from typing import List, Dict, Callable

from wtpy import WtBtEngine, EngineType
from wtpy import SelContext
from wtpy import BaseSelStrategy

from config.cfg_stk import cfg_stk

from .TimeSeriesAnalysis_Core import TimeSeriesAnalysis


def stdio(str):
    print(str)
    return str


class Parallel_Process_Worker():
    """
    Because WTCPP/WTPY is mainly a single-asset CTA-style trading engine
    For efficient market-wide back-testing with server CPUs,
    it is easiest to instance N WTCPP worker threads then manage data-coherence manually
    disable all other CTA-style features offered by WTCPP
    """

    def __init__(self, id: int, code_info: Dict[str, Dict[str, int]], shared_tensor: torch.Tensor, is_dummy=False):
        self.__id__ = id
        self.__codes__ = [code for code in code_info.keys()]
        self.__code_idxes__ = [code_info[code]['idx']
                               for code in self.__codes__]

        self.shared_tensor = shared_tensor

        self.inited = False
        self.barnum = 0

        # cross-section data
        self.cs_signal = None
        self.cs_value = None

        # TA core
        self.tech_analysis: Dict[str, TimeSeriesAnalysis] = {}

        for idx, code in enumerate(self.__codes__):
            plot = idx == 0 and self.__id__ == 0
            if plot:
                print(f'TA cores Initiated, back-test ready...')
            self.tech_analysis[code] = TimeSeriesAnalysis(
                code=code, code_idx=self.__code_idxes__[idx], shared_tensor=shared_tensor, plot=plot)
            if idx == 0:
                self.feature_names = self.tech_analysis[code].feature_names
                self.feature_types = self.tech_analysis[code].feature_types
                self.label_names = self.tech_analysis[code].label_names
                self.scaling_methods = self.tech_analysis[code].scaling_methods

        if is_dummy:
            return

        # WTCPP initialization
        self.engine = WtBtEngine(
            EngineType.ET_SEL, logCfg='./config/logcfg.yaml')
        self.engine.init(folder='.', cfgfile='./config/configbt.yaml')
        self.engine.configBacktest(cfg_stk.start, cfg_stk.end)
        self.engine.commitBTConfig()

        str_name = f'bt_stock'

        # Register callback functions for WonderTrader CPP interface
        callback_functions = [
            self.on_init,
            self.on_tick,
            self.on_bar,
            self.on_calculate,
            self.on_backtest_end,
        ]

        straInfo = Strategy_Interface(
            name=str_name,
            callback_functions=callback_functions,
        )
        self.engine.set_sel_strategy(
            straInfo,
            date=0, time=cfg_stk.n, period=cfg_stk.period_u,
            isRatioSlp=False)

    def run(self):
        self.start_time = time.time()
        self.engine.run_backtest()

    def on_init(self, context: SelContext):
        self.context = context  # export environment
        if self.__id__ == 0:
            print('Preparing Bars in DDR...')
            self.pbar = tqdm(total=len(self.__codes__))

        for idx, code in enumerate(self.__codes__):
            context.stra_prepare_bars(code, cfg_stk.wt_period_l, 1)
            if self.__id__ == 0:
                self.pbar.update(1)
                self.pbar.set_description(f'Init: {code}', True)
        return

    def on_tick(self, context: SelContext, code: str, newTick: dict):
        return

    def on_bar(self, context: SelContext, code: str, period: str, newBar: dict):
        # multi-level k bar generation
        TA = self.tech_analysis[code]
        TA.analyze(newBar['open'], newBar['high'], newBar['low'],
                   newBar['close'], newBar['vol'], newBar['time'])

        # # indicator guard (prepare and align)
        # if not self.inited:
        #     self.barnum += 1
        #     if self.barnum > 1*24*60: # need 1 day(s) of 1M data
        #         self.inited = True
        #     return

        # strategy
        # self.ST_Train(context, code)
        return

    def on_calculate(self, context: SelContext):
        """
        TimeSeries   (cpu1):                    |   on_bar_A(T0)    on_bar_A(T1)    ... 
        TimeSeries   (cpu2):                    |   on_bar_B(T0)    on_bar_B(T1)    ... 
        TimeSeries   (... ):                ->  |   ...          -> ...          -> ... 
        CrossSection (cpu0):    on_calc(skip)   V   on_calc(init)   on_calc(T0)     ... (CS is delaying TS by 1 for better compute efficiency)
        """
        return

    def on_backtest_end(self, context: SelContext):
        self.elapsed_time = time.time() - self.start_time
        if self.__id__ == 0:
            print(f'main BT loop time elapsed: {self.elapsed_time:2f}s')

        # # feature distribution analysis
        # if cfg_stk.stat and self.__id__ == 0:
        #     from Util.CheckDist import CheckDist
        #     import pandas as pd
        #     df = pd.DataFrame(
        #         self.shared_tensor[:, :, 0].to(torch.float32).numpy())
        #     df.columns = pd.Index(self.feature_names + self.label_names)
        #     CheckDist(df, [self.feature_types])
        return


class Strategy_Interface(BaseSelStrategy):
    def __init__(self, name: str, callback_functions: List[Callable]):
        BaseSelStrategy.__init__(self, name)
        [self.cb_on_init,
         self.cb_on_tick,
         self.cb_on_bar,
         self.cb_on_calculate,
         self.cb_on_backtest_end,
         ] = callback_functions

    def on_init(self, *args, **kwargs):
        self.cb_on_init(*args, **kwargs)

    def on_tick(self, *args, **kwargs):
        self.cb_on_tick(*args, **kwargs)
        return

    def on_bar(self, *args, **kwargs):
        self.cb_on_bar(*args, **kwargs)
        return

    def on_calculate(self, *args, **kwargs):
        self.cb_on_calculate(*args, **kwargs)
        return

    def on_backtest_end(self, *args, **kwargs):
        self.cb_on_backtest_end(*args, **kwargs)
        return
