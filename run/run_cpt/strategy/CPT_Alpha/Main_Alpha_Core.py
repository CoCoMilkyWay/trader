import torch
from typing import List, Dict

from config.cfg_cpt import cfg_cpt

from .TechnicalAnalysis_Core import TechnicalAnalysis_Core

class Main_Alpha_Core():
    def __init__(self, id:int, code_info:Dict[str, Dict[str, int]], shared_tensor:torch.Tensor):
        self.__id__ = id
        self.__codes__ = [code for code in code_info.keys()]
        self.__code_idxes__ = [code_info[code]['idx'] for code in self.__codes__]
        
        self.shared_tensor = shared_tensor
        
        self.inited = False
        self.barnum = 0
        
        # TA core
        self.tech_analysis: Dict[str, TechnicalAnalysis_Core] = {}
        
        for idx, code in enumerate(self.__codes__):
            plot = idx == 0 and self.__id__ == 0
            if plot:
                print(f'TA cores Initiated, back-test ready...')
            self.tech_analysis[code] = TechnicalAnalysis_Core(code=code, code_idx=self.__code_idxes__[idx], shared_tensor=shared_tensor, plot=plot)
            if idx == 0:
                self.feature_names = self.tech_analysis[code].feature_names
                self.label_names = self.tech_analysis[code].label_names
                self.scaling_methods = self.tech_analysis[code].scaling_methods
                
    def on_bar(self, code:str, open:float, high:float, low:float, close:float, vol:float, time:int):
        # multi-level k bar generation
        TA = self.tech_analysis[code]
        TA.analyze(open, high, low, close, vol, time)
        
        # # indicator guard (prepare and align)
        # if not self.inited:
        #     self.barnum += 1
        #     if self.barnum > 1*24*60: # need 1 day(s) of 1M data
        #         self.inited = True
        #     return
        
        # strategy
        # self.ST_Train(context, code)
        
    def on_backtest_end(self):
        # from Util.CheckDist import CheckDist
        # for idx, code in enumerate(self.__codes__):
        #     df, scaling_methods = self.tech_analysis[code].get_features_df()
        #     df.to_parquet(mkdir(f'{cfg_cpt.ML_MODEL_DIR}/data/ts_{code}_{cfg_cpt.start}_{cfg_cpt.end}.parquet'))
        #     
        #     if idx == 0:
        #         CheckDist(df)
        #         
        #         # print(df.shape)
        #         # print(df.describe())
        #         # print(df.info())
        #         # from .Model import train
        #         # train(df, scaling_methods)

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
