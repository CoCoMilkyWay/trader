import os
import sys
import json
import glob
from time import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from datetime import datetime
from tqdm import tqdm

from wtpy import CtaContext, BaseCtaStrategy
from Chan.Chan import CChan
from Chan.ChanConfig import CChanConfig
from Chan.Common.CEnum import KL_TYPE
from Chan.Common.kbar_parser import KLineHandler
from Chan.KLine.KLine_List import CKLine_List

# Import indicators
from Math.VolumeWeightedBands import VolumeWeightedBands

from Math.Chandelier_Stop import ChandelierIndicator
from Math.ChandeKroll_Stop import ChandeKrollStop
from Math.Parabolic_SAR_Stop import ParabolicSARIndicator
from Math.Adaptive_SuperTrend import AdaptiveSuperTrend
from Math.LorentzianClassifier import LorentzianClassifier

from strategy.CPT_Statistics.IndicatorAnalyzer import IndicatorAnalyzer, long_short_analyzer, distribution_info, analyze_distribution_shape

from config.cfg_cpt import cfg_cpt

@dataclass
class BarData:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float

class YearlyStats:
    def __init__(self):
        self.bars: List[BarData] = []
        # self.indicator_results = {
        #     'chandelier': defaultdict(list),
        #     'chandekroll': defaultdict(list),
        #     'parabolic_sar': defaultdict(list),
        #     'supertrend': defaultdict(list),
        #     'lorentzian': defaultdict(list),
        #     'chan': defaultdict(list)
        # }

    def to_arrays(self):
        """Convert bar data to numpy arrays for efficient batch processing"""
        bars = np.array([(b.timestamp, b.open, b.high, b.low, b.close, b.volume) for b in self.bars])
        return {
            'timestamp': bars[:, 0],
            'open': bars[:, 1],
            'high': bars[:, 2],
            'low': bars[:, 3],
            'close': bars[:, 4],
            'volume': bars[:, 5],
        }

class Main_Stats(BaseCtaStrategy):
    def __init__(self, name: str, codes: List[str], period: str):
        super().__init__(name)
        self.__period__ = period
        self.__codes__ = codes
        
        # Basic setup
        self.config = CChanConfig()
        self.current_year = None
        self.current_month = None
        self.current_day = None
        self.yearly_data = defaultdict(lambda: {code: YearlyStats() for code in codes})
        
        # Chan Elements
        self.lv_list = [lv[0] for lv in self.config.lv_list]
        
        # Indicator configurations
        self.indicator_configs = {
            # MA/BI indicators (Cycle Analysis)
            'vwma':  [
                {'window_size': l}
                for l in [10, 20, 50, 100, 200, 400, 800]
            ],
            
            # TP/SL indicators (TP/SL Analysis)
            'chandelier': [
                {'length': l, 'atr_period': a, 'mult': m}
                for l in [5, 20, 100, 300]
                for a in [1, 2, 4, 6, 8, 10, 15, 20]
                for m in [0.2, 0.4, 0.8, 1, 2, 3]
            ],
            'chandekroll': [
                {'atr_length': l, 'atr_coef': a, 'stop_len': m}
                for l in [50, 100, 200]
                for a in [0.2, 0.4, 0.8, 1, 2, 3]
                for m in [4, 8, 16, 32]
            ],
            'parabolic_sar': [
                {'acceleration': a, 'max_acceleration': a * m, 'initial_acceleration': a * n}
                for a in [0.0005, 0.001, 0.002, 0.004, 0.01]
                for m in [1, 5, 10, 20, 100]
                for n in [0, 10, 50, 100]
            ],
            'supertrend': [
                {'atr_len': a, 'factor': f, 'lookback': l}
                for a in [50, 100]
                for f in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                for l in [50, 100]
            ],
            'lorentzian': [
                {'kernel_lookback': n}
                for n in [8]
            ],
        }
        
        # check if results already exist
        self.json_files = glob.glob(f"{cfg_cpt.stats_result}/*.json")
        
        self.barnum = 0
        self.date = None
        self.start_time = time()
        self.initialize_year_structures()

    def initialize_year_structures(self):
        """Initialize or reinitialize all structures for a new year"""
        print('Wiping all data structure, re-calculating for new year...')
        
        self.KLineHandler: Dict[str, KLineHandler] = {}
        self.kl_datas: Dict[str, Dict[KL_TYPE, CKLine_List]] = {}
        self.chan_snapshot: Dict[str, CChan] = {}
        self.last_price: Dict[str, float] = {}
        self.last_ts: Dict[str, float] = {}
        
        # Initialize structures for each code
        for code in self.__codes__:
            self.KLineHandler[code] = KLineHandler(self.config.lv_list)
            self.kl_datas[code] = {
                level: CKLine_List(level, conf=self.config)
                for level in [lv[0] for lv in self.config.lv_list]
            }
            
            self.chan_snapshot[code] = CChan(
                code=code,
                kl_datas=self.kl_datas[code],
                lv_list=[lv[0] for lv in self.config.lv_list]
            )
            
            self.last_price[code] = 0.0
            self.last_ts[code] = 0.0

    def on_init(self, context: CtaContext):
        """Initial setup when strategy starts"""
        print('Initializing Strategy...')
        
        # Prepare bars for each code
        self.pbar = tqdm(total=len(self.__codes__), desc='Preparing Bars in DDR...')
        for idx, code in enumerate(self.__codes__):
            context.stra_prepare_bars(code, self.__period__, 1, isMain=idx == 0)

    def batch_process_indicators(self, code: str, bar_arrays: Dict[str, np.ndarray]) -> Dict:
        """Process entire year's data through indicators at once"""
        results = {
            # PA Analysis
            'bi':{},
            'vwma':{},
            
            # TP/SL Analysis
            'chandelier': {},
            'chandekroll': {},
            'parabolic_sar': {},
            'supertrend': {},
            'lorentzian': {},
        }
        # ===========================================
        # Process multi-level bi
        def calculate_minutes_per_bar(lv_list):
            minutes_per_bar = []
            # Start from the bottom level (1M)
            current_minutes = 1
            # Iterate through levels from bottom to top
            for level in reversed(lv_list):
                multiplier:int = level[1]
                current_minutes *= multiplier
                minutes_per_bar.append(current_minutes)
            minutes_per_bar.reverse()
            return minutes_per_bar
        bi_period = [] # in hours
        bi_delta = [] # % change relative to start price
        n_min_per_bar = calculate_minutes_per_bar(self.config.lv_list)
        for lv_idx, lv in enumerate(self.lv_list):
            lv_str = str(lv).split('.')[1]
            kl_list = self.kl_datas[code][lv]
            PA = kl_list.PA_Core
            bi_list = kl_list.bi_list
            for bi in bi_list:
                bi_period.append(bi.get_klu_cnt()*n_min_per_bar[lv_idx]/60)
                bi_delta.append(bi.amp()/bi.get_begin_val()*100)
            results['bi'][lv_str] = {
                'lv': lv_str,
                'period': distribution_info(bi_period),
                'delta': distribution_info(bi_delta),
                'period_shape': analyze_distribution_shape(bi_period),
                'delta_shape': analyze_distribution_shape(bi_delta),
            }

        # Process VolumeWeightedBands
        for idx, params in tqdm(enumerate(self.indicator_configs['vwma']), desc="VolumeWeightedBands..."):
            ind = VolumeWeightedBands(**params)
            last_cross_dir = None # 1: price cross up; -1: price cross down
            last_cross_ts = None # timestamp
            max_vwdev = 0.0
            long_period = []
            short_period = []
            long_max_dev = []
            short_max_dev = []
            for i in range(len(bar_arrays['timestamp'])):
                vwma, vwdev = ind.update(
                    bar_arrays['high'][i],
                    bar_arrays['low'][i],
                    bar_arrays['close'][i],
                    bar_arrays['volume'][i],
                    bar_arrays['timestamp'][i]
                )
                cross_up = bar_arrays['close'][i] > vwma
                max_vwdev = max(abs(vwdev), max_vwdev)
                if not last_cross_dir or not last_cross_ts:
                    last_cross_dir = 1 if cross_up else -1
                    last_cross_ts = bar_arrays['timestamp'][i]
                    max_vwdev = vwdev
                else:
                    if last_cross_dir == 1 and not cross_up:
                        long_period.append((bar_arrays['timestamp'][i]-last_cross_ts)/3600)
                        long_max_dev.append(max_vwdev)
                        last_cross_dir = -1
                        last_cross_ts = bar_arrays['timestamp'][i]
                        max_vwdev = 0.0
                    elif last_cross_dir == -1 and cross_up:
                        short_period.append((bar_arrays['timestamp'][i]-last_cross_ts)/3600)
                        short_max_dev.append(-max_vwdev)
                        last_cross_dir = 1
                        last_cross_ts = bar_arrays['timestamp'][i]
                        max_vwdev = 0.0
            long_period = distribution_info(long_period)
            short_period = distribution_info(short_period)
            long_max_dev = distribution_info(long_max_dev)
            short_max_dev = distribution_info(short_max_dev)
            period_shape = analyze_distribution_shape(long_period+short_period) # type: ignore
            max_dev_shape = analyze_distribution_shape(long_max_dev+short_max_dev) # type: ignore
            if long_period and short_period:
                print(f"({params}):period:{(long_period[0]+short_period[0])*0.5:03.2f}h({period_shape}/{max_dev_shape})")
            results['vwma'][str(idx)] = {
                'params': params,
                'long_period': long_period,
                'short_period': short_period,
                'long_max_dev': long_max_dev,
                'short_max_dev': short_max_dev,
                'period_shape': period_shape,
                'max_dev_shape': max_dev_shape,
            }

        # ===========================================
        analyzer = long_short_analyzer()
        
        # Process Chandelier
        for idx, params in tqdm(enumerate(self.indicator_configs['chandelier']), desc="chandelier..."):
            ind = ChandelierIndicator(**params)
            analyzer.init()
            for i in range(len(bar_arrays['timestamp'])):
                long_switch, short_switch = ind.update(
                    bar_arrays['high'][i],
                    bar_arrays['low'][i],
                    bar_arrays['close'][i],
                    bar_arrays['timestamp'][i]
                )
                analyzer.update(long_switch,short_switch,bar_arrays,i)
            [long_switchs,short_switchs,long_profits,short_profits,long_holds,short_holds,profit_shape,hold_shape] = analyzer.get_stats(params)
            results['chandelier'][str(idx)] = {
                'params': params,
                'long_switchs': long_switchs,
                'short_switchs': short_switchs,
                'long_profits': long_profits,
                'short_profits': short_profits,
                'long_holds': long_holds,
                'short_holds': short_holds,
                'profit_shape': profit_shape,
                'hold_shape': hold_shape,
            }

        # Process ChandeKroll
        for idx, params in tqdm(enumerate(self.indicator_configs['chandekroll']), desc="chandekroll..."):
            ind = ChandeKrollStop(**params)
            analyzer.init()
            for i in range(len(bar_arrays['timestamp'])):
                long_switch, short_switch = ind.update(
                    bar_arrays['high'][i],
                    bar_arrays['low'][i],
                    bar_arrays['close'][i],
                    bar_arrays['timestamp'][i]
                )
                analyzer.update(long_switch,short_switch,bar_arrays,i)
            [long_switchs,short_switchs,long_profits,short_profits,long_holds,short_holds,profit_shape,hold_shape] = analyzer.get_stats(params)
            results['chandekroll'][str(idx)] = {
                'params': params,
                'long_switchs': long_switchs,
                'short_switchs': short_switchs,
                'long_profits': long_profits,
                'short_profits': short_profits,
                'long_holds': long_holds,
                'short_holds': short_holds,
                'profit_shape': profit_shape,
                'hold_shape': hold_shape,
            }
        
        # Process Parabolic SAR
        for idx, params in tqdm(enumerate(self.indicator_configs['parabolic_sar']), desc=f"parabolic_sar..."):
            ind = ParabolicSARIndicator(**params)
            analyzer.init()
            for i in range(len(bar_arrays['timestamp'])):
                long_switch, short_switch = ind.update(
                    bar_arrays['high'][i],
                    bar_arrays['low'][i],
                    bar_arrays['timestamp'][i]
                )
                analyzer.update(long_switch,short_switch,bar_arrays,i)
            [long_switchs,short_switchs,long_profits,short_profits,long_holds,short_holds,profit_shape,hold_shape] = analyzer.get_stats(params)
            results['parabolic_sar'][str(idx)] = {
                'params': params,
                'long_switchs': long_switchs,
                'short_switchs': short_switchs,
                'long_profits': long_profits,
                'short_profits': short_profits,
                'long_holds': long_holds,
                'short_holds': short_holds,
                'profit_shape': profit_shape,
                'hold_shape': hold_shape,
            }

        # Process SuperTrend
        for idx, params in tqdm(enumerate(self.indicator_configs['supertrend']), desc="supertrend..."):
            ind = AdaptiveSuperTrend(**params)
            analyzer.init()
            for i in range(len(bar_arrays['timestamp'])):
                long_switch, short_switch = ind.update(
                    bar_arrays['high'][i],
                    bar_arrays['low'][i],
                    bar_arrays['close'][i],
                    bar_arrays['timestamp'][i]
                )
                analyzer.update(long_switch,short_switch,bar_arrays,i)
            [long_switchs,short_switchs,long_profits,short_profits,long_holds,short_holds,profit_shape,hold_shape] = analyzer.get_stats(params)
            results['supertrend'][str(idx)] = {
                'params': params,
                'long_switchs': long_switchs,
                'short_switchs': short_switchs,
                'long_profits': long_profits,
                'short_profits': short_profits,
                'long_holds': long_holds,
                'short_holds': short_holds,
                'profit_shape': profit_shape,
                'hold_shape': hold_shape,
            }

        # Process Lorentzian
        for idx, params in tqdm(enumerate(self.indicator_configs['lorentzian']), desc="lorentzian..."):
            ind = LorentzianClassifier(**params)
            analyzer.init()
            for i in tqdm(range(len(bar_arrays['timestamp']))):
                long_switch, short_switch = ind.update(
                    bar_arrays['high'][i],
                    bar_arrays['low'][i],
                    bar_arrays['close'][i],
                    bar_arrays['volume'][i]
                )
                analyzer.update(long_switch,short_switch,bar_arrays,i)
            [long_switchs,short_switchs,long_profits,short_profits,long_holds,short_holds,profit_shape,hold_shape] = analyzer.get_stats(params)
            results['lorentzian'][str(idx)] = {
                'params': params,
                'long_switchs': long_switchs,
                'short_switchs': short_switchs,
                'long_profits': long_profits,
                'short_profits': short_profits,
                'long_holds': long_holds,
                'short_holds': short_holds,
                'profit_shape': profit_shape,
                'hold_shape': hold_shape,
            }
            
        return results
    
    def process_year_end(self, year: int):
        """Process and save all accumulated data for the ending year"""
        print(f"Processing year {year} data...")
        
        serializable_data = {}
        for code in self.__codes__:
            # Convert bars to arrays for efficient processing
            yearly_stats = self.yearly_data[year][code]
            bar_arrays = yearly_stats.to_arrays()
            
            # Batch process all indicators
            serializable_data[code] = self.batch_process_indicators(code, bar_arrays)
            
        # Save to file
        output_path = f"{cfg_cpt.stats_result}/{year}_{cfg_cpt.FEE}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        
        # Clear data and reinitialize
        del self.yearly_data[year]
        self.initialize_year_structures()
        print(f"Year {year} processed and structures reinitialized.")

    def on_calculate(self, context: CtaContext):
        """Process each new bar"""
        self.barnum += 1
        
        if self.json_files:
            return
        
        # Check for year transition
        date = context.get_date()
        dt = datetime.strptime(str(date), '%Y%m%d')
        year = dt.year
        month = dt.month
        day = dt.day
        
        if self.current_year is None:
            self.current_year = year
        elif year != self.current_year:
            self.process_year_end(self.current_year)
            self.current_year = year

        if self.current_month is None:
            self.current_month = month
        elif month != self.current_month:
            self.current_month = month

        if self.current_day is None:
            self.current_day = day
        elif day != self.current_day:
            print(date)
            self.current_day = day

        # Update progress bar on first bar
        if self.barnum == 1:
            self.pbar.update(1)
            if self.barnum == len(self.__codes__):
                self.pbar.close()

        # Process each code
        for idx, code in enumerate(self.__codes__):
            np_bars = context.stra_get_bars(code, self.__period__, 1, isMain=idx==0)
            
            # Process multi-level k bars
            klu_dict = self.KLineHandler[code].process_bar(np_bars)
            ts = klu_dict[KL_TYPE.K_1M][-1].time.ts

            # Store bar data for batch processing
            bar_data = BarData(
                timestamp=ts,
                open=np_bars.opens[-1],
                high=np_bars.highs[-1],
                low=np_bars.lows[-1],
                close=np_bars.closes[-1],
                volume=np_bars.volumes[-1]
            )
            self.yearly_data[year][code].bars.append(bar_data)

            # Update Chan analysis
            self.chan_snapshot[code].trigger_load(klu_dict)
            for lv in [lv[0] for lv in self.config.lv_list]:
                self.kl_datas[code][lv].PA_Core.parse_dynamic_bi_list()

    def on_backtest_end(self, context: CtaContext):
        """Clean up at the end of backtest"""
        if self.current_year is not None:
            self.process_year_end(self.current_year)
            
        self.elapsed_time = time() - self.start_time
        print(f'Main BT loop time elapsed: {self.elapsed_time:.2f}s')
        
        analyzer = IndicatorAnalyzer()
        
        analyzer.plot_performance("indicator_perf.html")