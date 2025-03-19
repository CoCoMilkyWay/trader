import torch
from typing import List, Dict

from config.cfg_stk import cfg_stk

from .TechnicalAnalysis_Core import TechnicalAnalysis_Core
from .Wondertrader_Thread import Wondertrader_Thread


class Parallel_Process_Worker():
    def __init__(self, id: int, code_info: Dict[str, Dict[str, int]], shared_tensor: torch.Tensor):
        self.__id__ = id
        self.__codes__ = [code for code in code_info.keys()]
        self.__code_idxes__ = [code_info[code]['idx']
                               for code in self.__codes__]

        self.shared_tensor = shared_tensor

        self.inited = False
        self.barnum = 0

        # Register callback functions for WonderTrader CPP interface
        callback_functions = [
            self.on_init,
            self.on_tick,
            self.on_bar,
            self.on_calculate,
            self.on_backtest_end,
        ]
        self.wt = Wondertrader_Thread(self.__codes__, callback_functions)

        # TA core
        self.tech_analysis: Dict[str, TechnicalAnalysis_Core] = {}

        for idx, code in enumerate(self.__codes__):
            plot = idx == 0 and self.__id__ == 0
            if plot:
                print(f'TA cores Initiated, back-test ready...')
            self.tech_analysis[code] = TechnicalAnalysis_Core(
                code=code, code_idx=self.__code_idxes__[idx], shared_tensor=shared_tensor, plot=plot)
            if idx == 0:
                self.feature_names = self.tech_analysis[code].feature_names
                self.feature_types = self.tech_analysis[code].feature_types
                self.label_names = self.tech_analysis[code].label_names
                self.scaling_methods = self.tech_analysis[code].scaling_methods

    def on_init(self):
        return

    def on_tick(self):
        return

    def on_bar(self):
        return

    def on_calculate(self):
        return

    def on_backtest_end(self):
        return

    # def on_bar(self, code: str, open: float, high: float, low: float, close: float, vol: float, time: int):
    #     # multi-level k bar generation
    #     TA = self.tech_analysis[code]
    #     TA.analyze(open, high, low, close, vol, time)
# 
    # def on_backtest_end(self):
    #     if cfg_stk.stat:
    #         from Util.CheckDist import CheckDist
    #         from Util.UtilCpt import mkdir
    #         import pandas as pd
    #         df = pd.DataFrame(
    #             self.shared_tensor[:, :, 0].to(torch.float32).numpy())
    #         df.columns = pd.Index(self.feature_names + self.label_names)
    #         CheckDist(df, [self.feature_types])
# 
    #     return
