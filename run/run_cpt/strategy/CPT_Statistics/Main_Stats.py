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

class TimeframeStats: # yearly/monthly/...
    def __init__(self):
        self.bars: List[List[BarData]] = []
        for lv_idx, lv in enumerate([lv[0] for lv in CChanConfig().lv_list]):
            self.bars.append([])
        
    def to_arrays(self) -> List[Dict[str, np.ndarray]]:
        """Convert bar data to numpy arrays for efficient batch processing"""
        result = []
        for lv in range(len(self.bars)):
            bars = np.array([(b.timestamp, b.open, b.high, b.low, b.close, b.volume) 
                            for b in self.bars[lv]])
            result.append({
                'timestamp': bars[:, 0],
                'open': bars[:, 1],
                'high': bars[:, 2],
                'low': bars[:, 3],
                'close': bars[:, 4],
                'volume': bars[:, 5],
            })
        return result

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
        self.timeframe_data: Dict[int, Dict[int, Dict[str, TimeframeStats]]] = {}
        
        # Chan Elements
        self.lv_list = [lv[0] for lv in self.config.lv_list]
        
        # Indicator configurations
        self.indicator_configs = {
            # MA/BI indicators (Cycle Analysis)
            'vwma':  [
                {'window_size': l, 'window_size_atr': a}
                for l in [50, 200, 400]
                for a in [100, 200]
            ],
            
            # TP/SL indicators (TP/SL Analysis)
            'chandelier': [
                {'length': l, 'atr_period': a, 'mult': m}
                for l in [50, 100, 200]
                for a in [4, 8, 16, 32]
                for m in [1, 2, 4, 8, 16]
            ],
            'chandekroll': [
                {'atr_length': l, 'atr_coef': a, 'stop_len': m}
                for l in [100, 200]
                for a in [0.2, 0.4, 0.8]
                for m in [8, 16]
            ],
            'parabolic_sar': [
                {'acceleration': a, 'max_acceleration': a * m, 'initial_acceleration': a * n}
                for a in [0.0005, 0.001, 0.002]
                for m in [10, 20]
                for n in [0, 5]
            ],
            'supertrend': [
                {'atr_len': a, 'factor': f}
                for a in [50, 100]
                for f in [1, 1.5, 2, 2.5, 3]
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

    def batch_process_indicators(self, code: str, bar_arrays: List[Dict[str, np.ndarray]]) -> Dict:
        """Process entire year's data through indicators at once"""
        results = {
            # Volatility Location Analysis
            'fourier': {},
            
            # PA Analysis
            'bi':{},
            
            # Elasticity Analysis
            'vwma':{},
            
            # TP/SL Analysis
            'chandelier': {},
            'chandekroll': {},
            'parabolic_sar': {},
            'supertrend': {},
            'lorentzian': {},
        }
        
        # bar_arrays: -1:1M -2:5M etc.
        
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
        
        if cfg_cpt.analyze_bi:
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
        if cfg_cpt.analyze_vwma:
            n = -2 # 5M bar
            for idx, params in tqdm(enumerate(self.indicator_configs['vwma']), desc="VolumeWeightedBands..."):
                ind = VolumeWeightedBands(**params)
                last_cross_dir = None # 1: price cross up; -1: price cross down
                last_cross_ts = None # timestamp
                max_vwdev = 0.0
                long_period = []
                short_period = []
                long_max_dev = []
                short_max_dev = []
                for i in range(len(bar_arrays[n]['timestamp'])):
                    [vwma,vwdev,_,_] = ind.update(
                        bar_arrays[n]['high'][i],
                        bar_arrays[n]['low'][i],
                        bar_arrays[n]['close'][i],
                        bar_arrays[n]['volume'][i],
                        bar_arrays[n]['timestamp'][i]
                    )
                    cross_up = bar_arrays[n]['close'][i] > vwma
                    max_vwdev = max(abs(vwdev), max_vwdev)
                    if not last_cross_dir or not last_cross_ts:
                        last_cross_dir = 1 if cross_up else -1
                        last_cross_ts = bar_arrays[n]['timestamp'][i]
                        max_vwdev = vwdev
                    else:
                        if last_cross_dir == 1 and not cross_up:
                            long_period.append((bar_arrays[n]['timestamp'][i]-last_cross_ts)/3600)
                            long_max_dev.append(max_vwdev)
                            last_cross_dir = -1
                            last_cross_ts = bar_arrays[n]['timestamp'][i]
                            max_vwdev = 0.0
                        elif last_cross_dir == -1 and cross_up:
                            short_period.append((bar_arrays[n]['timestamp'][i]-last_cross_ts)/3600)
                            short_max_dev.append(-max_vwdev)
                            last_cross_dir = 1
                            last_cross_ts = bar_arrays[n]['timestamp'][i]
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
        if cfg_cpt.analyze_longshort:
            analyzer = long_short_analyzer()
            n = -1 # 1M bar
            # Process Chandelier
            for idx, params in tqdm(enumerate(self.indicator_configs['chandelier']), desc="chandelier..."):
                ind = ChandelierIndicator(**params)
                analyzer.init()
                for i in range(len(bar_arrays[n]['timestamp'])):
                    long_switch, short_switch = ind.update(
                        bar_arrays[n]['high'][i],
                        bar_arrays[n]['low'][i],
                        bar_arrays[n]['close'][i],
                        bar_arrays[n]['timestamp'][i]
                    )
                    analyzer.update(long_switch,short_switch,bar_arrays[n],i)
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

            n = -1 # 1M bar
            # Process ChandeKroll
            for idx, params in tqdm(enumerate(self.indicator_configs['chandekroll']), desc="chandekroll..."):
                ind = ChandeKrollStop(**params)
                analyzer.init()
                for i in range(len(bar_arrays[n]['timestamp'])):
                    long_switch, short_switch = ind.update(
                        bar_arrays[n]['high'][i],
                        bar_arrays[n]['low'][i],
                        bar_arrays[n]['close'][i],
                        bar_arrays[n]['timestamp'][i]
                    )
                    analyzer.update(long_switch,short_switch,bar_arrays[n],i)
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

            n = -1 # 1M bar
            # Process Parabolic SAR
            for idx, params in tqdm(enumerate(self.indicator_configs['parabolic_sar']), desc=f"parabolic_sar..."):
                ind = ParabolicSARIndicator(**params)
                analyzer.init()
                for i in range(len(bar_arrays[n]['timestamp'])):
                    long_switch, short_switch = ind.update(
                        bar_arrays[n]['high'][i],
                        bar_arrays[n]['low'][i],
                        bar_arrays[n]['timestamp'][i]
                    )
                    analyzer.update(long_switch,short_switch,bar_arrays[n],i)
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

            n = -2 # 5M bar
            # Process SuperTrend
            for idx, params in tqdm(enumerate(self.indicator_configs['supertrend']), desc="supertrend..."):
                ind = AdaptiveSuperTrend(**params)
                analyzer.init()
                for i in range(len(bar_arrays[n]['timestamp'])):
                    long_switch, short_switch = ind.update(
                        bar_arrays[n]['high'][i],
                        bar_arrays[n]['low'][i],
                        bar_arrays[n]['close'][i],
                        bar_arrays[n]['timestamp'][i]
                    )
                    analyzer.update(long_switch,short_switch,bar_arrays[n],i)
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

            n = -2 # 5M bar
            # Process Lorentzian
            for idx, params in tqdm(enumerate(self.indicator_configs['lorentzian']), desc="lorentzian..."):
                ind = LorentzianClassifier(**params)
                analyzer.init()
                for i in tqdm(range(len(bar_arrays[n]['timestamp']))):
                    long_switch, short_switch = ind.update(
                        bar_arrays[n]['high'][i],
                        bar_arrays[n]['low'][i],
                        bar_arrays[n]['close'][i],
                        bar_arrays[n]['volume'][i]
                    )
                    analyzer.update(long_switch,short_switch,bar_arrays[n],i)
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

        # Process Frourier Transforms
        if cfg_cpt.analyze_fourier:
            from scipy import signal
            from scipy.fft import rfft, rfftfreq  # Using rfft instead of fft
            from scipy.signal import lombscargle
            from scipy.integrate import simpson
            n = -1 # 1M bar
            print('Fourier Transform...')
            
            y = []
            for i in range(len(bar_arrays[n]['timestamp'])):
                y.append(bar_arrays[n]['close'][i])
            # Perform FFT
            sample_rate = 60 # min/hours
            nyquist = sample_rate / 2
            min_period=1/nyquist * 1.1 # 10% margin for nyquist limit
            max_period=10
            num_periods=1000
            y = np.array(y)
            t = np.arange(len(y)) / sample_rate  # Time in hours
            # Create evenly spaced periods
            periods = np.linspace(min_period, max_period, num_periods)
            # Convert to frequencies (will be unevenly spaced)
            frequencies = 1/periods
            angular_frequencies = 2 * np.pi * frequencies
            # Apply bandpass filter first (Avoid aliasing)
            
            btype = 'band'
            wn = [1/(max_period * nyquist), 1/(min_period * nyquist)]
            # wn = 1/(max_period * nyquist)
            order = 4
            # b_elp, a_elp = signal.ellip(
            #     order,
            #     rp=0.01,     # Extremely small passband ripple
            #     rs=100,      # Very high stopband attenuation
            #     Wn=wn,
            #     btype=btype,
            #     analog=False)
            b_but, a_but = signal.butter(
                order,
                wn,
                btype=btype,
                analog=False)
            # y_intermediate = signal.filtfilt(b_elp, a_elp, y)
            y_filtered = signal.filtfilt(b_but, a_but, y)
            # y_filtered = y
            # Method 1: Lomb-Scargle Periodogram
            # lombscargle_power = lombscargle(t, y_filtered - np.mean(y_filtered), angular_frequencies)
            # Convert power to amplitude
            # lombscargle_amplitude = 2 * np.sqrt(lombscargle_power/len(t))
            # Method 2: Direct DFT calculation at specific frequencies
            def direct_dft(t, y, freq):
                """Compute DFT at a specific frequency"""
                complex_exp = np.exp(-2j * np.pi * freq * t)
                return np.abs(np.sum(y * complex_exp)) / len(t)
            direct_amplitude = np.zeros(len(frequencies))
            for i, freq in enumerate(frequencies):
                direct_amplitude[i] = 2 * direct_dft(t, y_filtered, freq)
                

            # Smooth both spectra while preserving area
            window_size = int(num_periods/10)  # 10% window
            sort_idx = np.argsort(periods)  # Get indices that would sort periods
            periods_sorted = periods[sort_idx]

            # # Smooth and normalize Lomb-Scargle
            # original_area_ls = simpson(lombscargle_amplitude, periods)
            # lombscargle_smoothed = signal.savgol_filter(lombscargle_amplitude, window_size, 3)
            # lombscargle_smoothed = np.maximum(lombscargle_smoothed, 0)  # Ensure no negatives
            # new_area_ls = simpson(lombscargle_smoothed, periods)
            # lombscargle_smoothed *= (original_area_ls / new_area_ls)

            # Smooth and normalize Direct DFT
            dft_amp_sorted = direct_amplitude[sort_idx]
            original_area_dft = simpson(y=dft_amp_sorted, x=periods_sorted)
            
            direct_amplitude_smoothed = signal.savgol_filter(direct_amplitude, window_size, 3)
            direct_amplitude_smoothed = np.maximum(direct_amplitude_smoothed, 0)
            new_area_dft = simpson(y=direct_amplitude_smoothed[sort_idx], x=periods_sorted)
            direct_amplitude_smoothed *= (original_area_dft / new_area_dft)

            results['fourier'] = {
                'periods': periods.tolist(),
                # 'frequencies': frequencies.tolist(),
                # 'lombscargle_amplitude': lombscargle_amplitude.tolist(),
                'direct_dft_amplitude': direct_amplitude.tolist(),
                'direct_dft_amplitude_smoothed': direct_amplitude_smoothed.tolist(),
                # 'filtered_timeseries': y_filtered.tolist()
            }

        return results

    def process_on_period_end(self, year: int, month: int, day: int):
        """Process and save accumulated data for the ending period based on n-month intervals"""
        # Only process if we've reached an n-month interval
        if month % cfg_cpt.analyze_n_month == 0:
            start_month = month - cfg_cpt.analyze_n_month + 1
            print(f"Processing data from {year}-{start_month:02d} to {year}-{month:02d}")

            serializable_data = {}

            # Process data for the n-month period
            for code in self.__codes__:
                monthly_stats = self.timeframe_data[year][month][code]
                bar_arrays = monthly_stats.to_arrays()
                serializable_data[f"{code}"] = self.batch_process_indicators(code, bar_arrays)

            # Save to file
            if serializable_data:
                output_path = f"{cfg_cpt.stats_result}/{year}_{month}_{cfg_cpt.analyze_n_month}_{cfg_cpt.FEE}.json"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(serializable_data, f, indent=4)

            # Clear all processed data up to current month
            for m in range(start_month, month + 1):
                if m > 0 and year in self.timeframe_data and m in self.timeframe_data[year]:
                    del self.timeframe_data[year][m]

            if year in self.timeframe_data and not self.timeframe_data[year]:
                del self.timeframe_data[year]

            self.initialize_year_structures()
            print(f"Period {year}-{start_month:02d} to {year}-{month:02d} processed and structures reinitialized.")

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

        # Initialize timeframe data structure if needed
        if year not in self.timeframe_data:
            self.timeframe_data[year] = {}
        if month not in self.timeframe_data[year]:
            self.timeframe_data[year][month] = {code: TimeframeStats() for code in self.__codes__}

        # Check for period change and process data
        if self.current_year and self.current_month and self.current_day:
            if (month != self.current_month or year != self.current_year):
                print(f"{year}-{month}")
                if self.current_month % cfg_cpt.analyze_n_month == 0:
                    self.process_on_period_end(self.current_year, self.current_month, self.current_day)


        self.current_year = year
        self.current_month = month
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
            for lv_idx, lv in enumerate(self.lv_list):
                if lv in klu_dict:
                    K = klu_dict[lv][-1]
                    # Store bar data for batch processing
                    bar_data = BarData(
                        timestamp   =K.time.ts,
                        open        =K.open,
                        high        =K.high,
                        low         =K.low,
                        close       =K.close,
                        volume      =K.volume,
                    )
                    self.timeframe_data[year][month][code].bars[lv_idx].append(bar_data)

            # Update Chan analysis
            self.chan_snapshot[code].trigger_load(klu_dict)
            for lv in [lv[0] for lv in self.config.lv_list]:
                self.kl_datas[code][lv].PA_Core.parse_dynamic_bi_list()

    def on_backtest_end(self, context: CtaContext):
        """Clean up at the end of backtest"""
        
        self.elapsed_time = time() - self.start_time
        print(f'Main BT loop time elapsed: {self.elapsed_time:.2f}s')
        
        analyzer = IndicatorAnalyzer()
        
        analyzer.plot_performance("indicator_perf.html")