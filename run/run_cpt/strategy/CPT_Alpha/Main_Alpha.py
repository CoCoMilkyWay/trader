from time import time
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional

from wtpy import SelContext
from wtpy import BaseSelStrategy

from Util.UtilCpt import mkdir
from Util.Parallel import Parallel

from .TechnicalAnalysis_Core import TechnicalAnalysis_Core

from config.cfg_cpt import cfg_cpt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def stdio(str):
    print(str)
    return str

def parallel_worker(worker_id:int, data):
    print(worker_id, data)
    
class Main_Alpha(BaseSelStrategy):
    def __init__(self, name: str, codes: List[str], period: str):
        BaseSelStrategy.__init__(self, name)
        
        self.__period__ = period
        self.__codes__ = codes
        
        # stats
        self.inited = False
        self.barnum = 0
        self.date = None
        self.time = None
        self.code_info: Dict[str, Dict] = {}
        
        # TA core
        # self.tech_analysis: Dict[str, TechnicalAnalysis_Core] = {}
        
        # bsp:
        # (1: long, -1: short, 0: no hold, position open period)
        self.markers: Dict[str, List[Tuple]] = {}
        
    def on_init(self, context: SelContext):
        print('Preparing Bars in DDR...')
        self.pbar = tqdm(total=len(self.__codes__))
        
        for idx, code in enumerate(self.__codes__):
            r = context.stra_prepare_bars(code, self.__period__, 1)
            
            self.code_info[code] = {
                'idx':idx,
                'date':None,
                'hour':None,
                'min':None,
            }
            
            self.init_new_code(code)
            self.markers[code] = []
            
            self.pbar.update(1)
            self.pbar.set_description(f'init: {code}', True)
        self.pbar.close()
        
        self.P = Parallel()
        self.P.parallel_init(self.code_info, parallel_worker)
        
        context.stra_log_text(stdio("Strategy Initiated, timer started..."))
        self.start_time = time()
        
    def init_new_code(self, code: str):
        # initiate new code specific models/structs
        # self.tech_analysis[code] = TechnicalAnalysis_Core(code=code, train=cfg_cpt.train)
        pass
    
    def on_tick(self, context:SelContext, code:str, newTick:dict):
        print(code, 'newTick')
        return
    
    def on_bar(self, context:SelContext, code:str, period:str, newBar:dict):
        date = str(newBar['date'])
        hour = str(newBar['time'])[-4:-2]
        min = str(newBar['time'])[-2:]
        if date != self.code_info[code]['date']:
            self.code_info[code]['date'] = date
            self.code_info[code]['hour'] = hour
            self.code_info[code]['min'] = min
            if code == 'Binance.UM.BTCUSDT':
                print(code, self.code_info[code], newBar)
        
        self.P.parallel_feed(code, period, newBar)
        
        return
    
    def on_calculate(self, context:SelContext):
        """
        on_calc(T0) ->  |   on_bar_A(T1) -> on_bar_A(T2)
                        |   on_bar_B(T1)    on_bar_B(T2)
                        V   on_calc(T1)     on_calc(T2)
        """
        self.P.parallel_block()
        # curTime = context.stra_get_time()
        # np_bars = context.stra_get_bars('Binance.UM.BTCUSDT', self.__period__, 1,)
        # print('on_calculate: ', curTime, np_bars.closes[-1])
    
    # def on_calculate(self, context: SelContext):
    #     # all sub-ed bars closed (main/non-main) at this period
    #     self.barnum += 1
    #     
    #     date = context.get_date()
    #     time = context.stra_get_time()
    #     print(date, time)
    #     if date!=self.date:
    #         print(date)
    #     self.date = date
    #     self.time = time
    #     
    #     for idx, code in enumerate(self.__codes__):
    #         try:
    #             np_bars = context.stra_get_bars(code, self.__period__, 1,)
    #             print(code, np_bars)
    #         except Exception as e:
    #             print(f'{code}: {e}')
    #             continue
    #         if not np_bars:
    #             print(f'no bars: {code}')
    #             continue
    #         # print(np_bars.ndarray)
    #         # multi-level k bar generation
    #         # TA = self.tech_analysis[code]
    #         # TA.analyze(np_bars)
    #         
    #         # indicator guard (prepare and align)
    #         if not self.inited:
    #             if self.barnum < 1*24*60: # need 1 day(s) of 1M data
    #                 self.inited = True
    #             continue
    #         
    #         # strategy
    #         # self.ST_Train(context, code)
            
    def on_backtest_end(self, context: SelContext):
        self.P.parallel_close()
        self.elapsed_time = time() - self.start_time
        print(f'main BT loop time elapsed: {self.elapsed_time:2f}s')
        return
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
