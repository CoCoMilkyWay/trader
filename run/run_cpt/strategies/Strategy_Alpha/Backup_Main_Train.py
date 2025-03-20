import os
import sys
import math
import numpy as np
from time import time
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional

from wtpy import CtaContext
from wtpy import BaseCtaStrategy

from Util.UtilCpt import mkdir

from ..CPT_Train.TimeSeriesAnalysis import TimeSeriesAnalysis

from config.cfg_cpt import cfg_cpt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
sys.path.append(os.path.join(os.path.dirname(__file__), "../../app"))

def stdio(str):
    print(str)
    return str

class Main_Train(BaseCtaStrategy):
    def __init__(self, name: str, codes: List[str], period: str):
        BaseCtaStrategy.__init__(self, name)
        self.__period__ = period
        self.__codes__ = codes
        
        # stats
        self.barnum = 0
        self.inited = False
        self.last_price: Dict[str, float] = {}
        self.last_ts: Dict[str, float] = {}
        self.start_time = time()
        self.date = None
        
        # TA core
        self.tech_analysis: Dict[str, TimeSeriesAnalysis] = {}
        
        # ML_models
        
        # bsp:
        # (1: long, -1: short, 0: no hold, position open period)
        # self.holds: Dict[str, List[int]] = {}
        self.markers: Dict[str, List[Tuple]] = {}
        
        # self.ST_signals: Dict[str, List[List]] = {}  # multiple signals
        # self.ST_trade: Dict[str, List] = {}  # 1 on-going trades at a time
        
    def on_init(self, context: CtaContext):
        print('Preparing Bars in DDR...')
        self.pbar = tqdm(total=len(self.__codes__))
        
        for idx, code in enumerate(self.__codes__):
            r = context.stra_prepare_bars(
                code, self.__period__, 1, isMain=idx == 0)
            # only 1 series is registered as 'Main', which works as clock, more registrations would result in fault
            # on_calculate is triggered once main bar is closed
            # if hook is installed, on_calculate_done would be triggered(for RL)
            
            self.init_new_code(code)
            self.last_price[code] = 0.0
            self.last_ts[code] = 0.0
            # self.holds[code] = [0, 0]
            self.markers[code] = []
            
            # ST(Strategies): liquidity
            # self.ST_signals[code] = []
            # self.ST_trade[code] = []
            self.pbar.update(1)
            self.pbar.set_description(f'init: {code}', True)
            
        self.pbar.close()
        context.stra_log_text(stdio("Strategy Initiated"))
        
    def init_new_code(self, code: str):
        # initiate new code specific models/structs
        self.tech_analysis[code] = TimeSeriesAnalysis(code=code, train=cfg_cpt.train)
        
    def on_calculate(self, context: CtaContext):
        # all sub-ed bars closed (main/non-main) at this period
        self.barnum += 1
        
        date = context.get_date()
        time = context.stra_get_time()
        if date!=self.date:
            print(date)
        self.date = date
        self.time = time
        
        for idx, code in enumerate(self.__codes__):
            try:
                np_bars = context.stra_get_bars(code, self.__period__, 1, isMain=idx == 0)
            except Exception as e:
                print(f'{code}: {e}')
                continue
            if not np_bars:
                print(f'no bars: {code}')
                continue
            # print(np_bars.ndarray)
            # multi-level k bar generation
            TA = self.tech_analysis[code]
            TA.analyze(np_bars)
            
            # indicator guard (prepare and align)
            if not self.inited:
                if self.barnum < 1*24*60: # need 1 day(s) of 1M data
                    self.inited = True
                continue
            
            # strategy
            # self.ST_Train(context, code)
            
    def on_backtest_end(self, context: CtaContext):
        self.elapsed_time = time() - self.start_time
        print(f'main BT loop time elapsed: {self.elapsed_time:2f}s')
        
        for idx, code in enumerate(self.__codes__):
            df, scaling_methods = self.tech_analysis[code].get_features_df()
            df.to_parquet(mkdir(f'{cfg_cpt.ML_MODEL_DIR}/data/ts_{code}_{cfg_cpt.start}_{cfg_cpt.end}.parquet'))
            
            if idx == 0:
                print(df.shape)
                print(df.describe())
                print(df.info())
                # from .Model import train
                # train(df, scaling_methods)
        
        if cfg_cpt.dump_ind:
            from Chan.Plot.PlotDriver import ChanPlotter
            from Util.plot.plot_fee_grid import plot_fee_grid
            from Util.plot.plot_show import plot_show
            
            for code in self.__codes__:
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
                    TA.kl_datas, self.markers[code], indicators)
                fig = plot_fee_grid(fig, dtick=TA.closes[0][-1]*cfg_cpt.FEE)
                plot_show(fig)
                break
        return
    
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
            model = BiLTSM_Model(
                sequence_length=10, 
                shape_features=2, 
                additional_features=len(train_features[-1]),
                price_normalization='standard',
                additional_features_normalization='standard'
            )
            history = model.train(
                train_sequences, 
                train_features, 
                train_labels,
                epochs=50, 
                batch_size=32
            )
            model.compile_model()
            
            model.save_model(mkdir(cfg_cpt.model_path))
            
            # Make predictions
            predictions = model.predict(train_sequences, train_features)
            
            # Print results
            for i, (pred, true_label) in enumerate(zip(predictions, train_labels)):
                print(f"Sample {i}: Prediction={pred[0]:.2f}, True Label={true_label}")

    # def ST_Train(self, context: CtaContext, code: str):
    #     """
    #     Strategy: Training ML model
    #     """
    #     PA = self.kl_datas[code][KL_TYPE.K_15M].PA_Core
    #     bi_list_m1 = self.kl_datas[code][KL_TYPE.K_1M].PA_Core.bi_list
    #     bi_list_m15 = self.kl_datas[code][KL_TYPE.K_15M].PA_Core.bi_list
    #     kl_list = self.kl_datas[code][self.lv_list[-1]]
    #     if len(bi_list_m15) < 3:  # make sure we have both sup and res
    #         return
    # 
    #     match, lower_targets, upper_targets, hint = \
    #     PA.PA_Liquidity.check_sup_res(self.close)
    #     
    #     if match:
    #         long_short =  self.vwap > self.close
    #         elasticity =  self.dev_mult
    #         if long_short:
    #             targets = upper_targets
    #         else:
    #             targets = lower_targets
    #         if len(targets) > 0: # this is usually not a problem, just to be safe
    #             if bi_list_m1[-1].is_sure:
    #                 self.ST_signals[code].append([
    #                     # status
    #                     0, # 0: opened 1: closed with feature and label
    #                     
    #                     # features
    #                     [(bi.get_klu_cnt(), bi.get_end_val()) for bi in bi_list_m1[-10:]], # pattern
    #                     long_short, # dir of trade (long/short properties may be asymmetric)
    #                     elasticity,
    #                     self.close, # entry_price
    #                     # TODO: e.g. dst to 1st OB, trend, etc
    #                     
    #                     # label(pnl)
    #                     0.0,
    #                     ])
    # 
    #         for signal in self.ST_signals[code][:]:
    #             signal_state = signal[0]
    #             bi_pattern = signal[1]
    #             long_short = signal[2]
    #             elasticity = signal[3]
    #             entry_price = signal[4]
    #             label = signal[5]
    # 
    #             stop = False
    #             dir = 0
    #             if signal_state == 0:
    #                 if long_short:
    #                     if self.short_switch:
    #                         stop = True
    #                         dir = 1
    #                 elif not long_short:
    #                     if self.long_switch:
    #                         stop = True
    #                         dir = -1
    #                 if stop:
    #                     signal[0] = 1
    #                     pnl = dir*(self.close - entry_price)/entry_price
    #                     pnl = pnl + 0.5 # [-1, 0 ,1] -> [-0.5, 0.5, 1.5]
    #                     if pnl < 0:
    #                         pnl = 0
    #                     elif pnl > 1:
    #                         pnl = 1
    #                     signal[5] = pnl
    #                     if signal[5] > 0.01:
    #                         symbol = "v" if long_short else "^"
    #                         print(f'{self.date}-{self.time:>4} {symbol} label:{signal[5]:>3.2f}')