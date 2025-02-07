from time import time
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional

from config.cfg_cpt import cfg_cpt

from Util.UtilCpt import mkdir
from .Parallel_Process import SharedData
from .TechnicalAnalysis_Core import TechnicalAnalysis_Core

class Main_Alpha_Core():
    def __init__(self, id:int, codes: List[str]):
        self.__id__ = id
        self.__codes__ = codes
        
        self.inited = False
        self.barnum = 0

        # TA core
        self.tech_analysis: Dict[str, TechnicalAnalysis_Core] = {}
        
        for idx, code in enumerate(self.__codes__):
            plot = idx == 0 and self.__id__ == 0
            self.tech_analysis[code] = TechnicalAnalysis_Core(code=code, train=cfg_cpt.train, plot=plot)
            
        if self.__id__ == 0:
            print(f'TA cores Initiated, back-test begin...')
            
    def on_bar(self, code:str, open:float, high:float, low:float, close:float, vol:float, time:int):
        # multi-level k bar generation
        TA = self.tech_analysis[code]
        TA.analyze(open, high, low, close, vol, time)
        
        # indicator guard (prepare and align)
        if not self.inited:
            self.barnum += 1
            if self.barnum > 1*24*60: # need 1 day(s) of 1M data
                self.inited = True
            return
        
        # strategy
        # self.ST_Train(context, code)
        
    def on_backtest_end(self):
        for idx, code in enumerate(self.__codes__):
            df, scaling_methods = self.tech_analysis[code].get_features_df()
            df.to_parquet(mkdir(f'{cfg_cpt.ML_MODEL_DIR}/data/ts_{code}_{cfg_cpt.start}_{cfg_cpt.end}.parquet'))

            if idx == 0:
                print(df.shape)
                print(df.describe())
                print(df.info())
                # from .Model import train
                # train(df, scaling_methods)

        if cfg_cpt.plot and self.__id__ == 0:
            from Chan.Plot.PlotDriver import ChanPlotter
            from Util.plot.plot_fee_grid import plot_fee_grid
            from Util.plot.plot_show import plot_show

            code = self.__codes__[0]
            TA = self.tech_analysis[code]
            TA.AdaptiveSuperTrend.get_stats(code)
            
            # get labels
            ts, closes, labels1, labels2 = TA.ts_label.get_labels()
            indicators = [
                TA.AdaptiveSuperTrend,
                ts,
                closes,
                labels1,
                labels2,
                ]
            fig = ChanPlotter().plot(
                TA.kl_datas, [], indicators)
            fig = plot_fee_grid(fig, dtick=TA.closes[0][-1]*cfg_cpt.FEE)
            plot_show(fig)
