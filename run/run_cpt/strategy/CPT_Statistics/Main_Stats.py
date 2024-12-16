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
from Math.Chandelier_Stop import ChandelierIndicator
from Math.Parabolic_SAR_Stop import ParabolicSARIndicator
from Math.Adaptive_SuperTrend import AdaptiveSuperTrend

from strategy.CPT_Statistics.IndicatorAnalyzer import IndicatorAnalyzer, long_short_analyzer

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
        self.indicator_results = {
            'chandelier': defaultdict(list),
            'parabolic_sar': defaultdict(list),
            'supertrend': defaultdict(list),
            'chan': defaultdict(list)
        }

    def to_arrays(self):
        """Convert bar data to numpy arrays for efficient batch processing"""
        bars = np.array([(b.timestamp, b.high, b.low, b.close) for b in self.bars])
        return {
            'timestamp': bars[:, 0],
            'high': bars[:, 1],
            'low': bars[:, 2],
            'close': bars[:, 3]
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
        
        # Indicator configurations
        self.indicator_configs = {
            'chandelier': [
                {'length': l, 'atr_period': a, 'mult': m}
                for l in [5, 20, 100, 300]
                for a in [1, 2, 3, 4]
                for m in [1, 2, 3, 4]
            ],
            'parabolic_sar': [
                {'acceleration': a, 'max_acceleration': a * m, 'initial_acceleration': a * n}
                for a in [0.0005, 0.001, 0.002, 0.004, 0.01]
                for m in [1, 5, 10, 20, 100]
                for n in [0, 10, 50, 100]
            ],
            'supertrend': [
                {'atr_len': a, 'factor': f, 'lookback': l}
                for a in [5, 50, 100]
                for f in [1, 2, 3, 4, 5, 6, 7, 8]
                for l in [50, 100]
            ]
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
            'chandelier': {},
            'parabolic_sar': {},
            'supertrend': {}
        }
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
            [long_switchs,short_switchs,long_profits,short_profits,long_holds,short_holds] = analyzer.get_stats(params)
            results['chandelier'][str(idx)] = {
                'params': params,
                'long_switchs': long_switchs,
                'short_switchs': short_switchs,
                'long_profits': long_profits,
                'short_profits': short_profits,
                'long_holds': long_holds,
                'short_holds': short_holds,
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
            [long_switchs,short_switchs,long_profits,short_profits,long_holds,short_holds] = analyzer.get_stats(params)
            results['parabolic_sar'][str(idx)] = {
                'params': params,
                'long_switchs': long_switchs,
                'short_switchs': short_switchs,
                'long_profits': long_profits,
                'short_profits': short_profits,
                'long_holds': long_holds,
                'short_holds': short_holds,
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
            [long_switchs,short_switchs,long_profits,short_profits,long_holds,short_holds] = analyzer.get_stats(params)
            results['supertrend'][str(idx)] = {
                'params': params,
                'long_switchs': long_switchs,
                'short_switchs': short_switchs,
                'long_profits': long_profits,
                'short_profits': short_profits,
                'long_holds': long_holds,
                'short_holds': short_holds,
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
        stats_list = analyzer.load_data(cfg_cpt.stats_result)
        analyzer.plot_performance(stats_list, "indicator_perf.html")