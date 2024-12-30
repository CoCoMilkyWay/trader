import array
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from wtpy.WtDataDefs import WtNpKline

from Chan.Common.CEnum import KL_TYPE, DATA_FIELD
from Chan.Common.CTime import CTime
from Chan.KLine.KLine_Unit import CKLine_Unit

from Math.performance.log_return import logreturn
from Math.candles.candlestrength import candlestrength
from Math.cycle.timely import timely
from Math.overlap.ema import ema
from Math.overlap.ma import ma
from Math.overlap.bband import bband
from Math.overlap.donchian import donchian
from Math.overlap.keltner import keltner
from Math.volatility.stddev import stddev
from Math.volatility.atr import atr
from Math.volatility.massi import massi
from Math.volatility.rvi import rvi
from Math.volatility.gk import gk
from Math.momentum.rsi import rsi
from Math.momentum.stoch_rsi import stoch_rsi
from Math.momentum.macd import macd
from Math.momentum.cci import cci
from Math.momentum.tsi_trend import tsi_trend
from Math.momentum.tsi_true import tsi_true
from Math.momentum.roc import roc
from Math.momentum.fisher import fisher
from Math.momentum.cmo import cmo
from Math.momentum.adx import adx
from Math.momentum.squeeze import squeeze
from Math.momentum.uo import uo
from Math.momentum.kst import kst
from Math.momentum.williams_r import williams_r
from Math.volume.aobv import aobv
from Math.volume.avwap import avwap
from Math.volume.eom import eom

# Define indicator categories and their outputs
@dataclass
class IndicatorSpec:
    name: str
    category: str
    outputs: list[str]  # List of output field names

class TechnicalAnalysis:
    
    # explicitly declare which attributes a class can have
    # Primarily used to optimize memory usage and access time for class instances
    # Restrictions:
    #   1. can't add new attributes that weren't declared in __slots__
    #   2. can't use __dict__ unless explicitly include it in __slots__
    # __slots__ = ('levels', 'multiplier', 'periods', 'counts', 'timestamp',
    #              '_bar_template', '_max_size', '_train',
    #              'opens', 'highs', 'lows', 'closes', 'volumes', 
    #              )
    
    def __init__(self, lv_list: list, train:bool = False):
        """
        Initialize with a maximum size to prevent unbounded growth
        max_size: Maximum number of bars to store per level
        """
        self._train = train
        self.levels = tuple(lv[0] for lv in reversed(lv_list))
        self.multiplier = tuple(lv[1] for lv in reversed(lv_list))
        self.periods = dict(zip(self.levels, self.multiplier))
        
        n_levels = len(self.levels)
        # Use array for fixed-size counters
        self.counts = array.array('I', [0] * n_levels)
        
        # Initialize fixed-size arrays for each level
        self.opens = [array.array('d', [0.0]) for _ in range(n_levels)]
        self.highs = [array.array('d', [0.0]) for _ in range(n_levels)]
        self.lows = [array.array('d', [0.0]) for _ in range(n_levels)]
        self.closes = [array.array('d', [0.0]) for _ in range(n_levels)]
        self.volumes = [array.array('L', [0]) for _ in range(n_levels)]
        self.timestamp = 0.0
        
        # ================== indicators ===========================
        # 0:1M, 1:5M, 2:15M, ...
        # overlay
        self.ma20           = ma(self.closes[0], 20)
        self.ema9           = ema(self.closes[0], 9)
        self.ema12          = ema(self.closes[0], 12)
        self.ema20          = ema(self.closes[0], 20)
        self.ema26          = ema(self.closes[0], 26)
        self.vema10         = ema(self.volumes[0], 10)
        
        # volatility
        self.stddev20       = stddev(self.closes[0], self.ma20.ma, 20)
        self.atr10          = atr(self.highs[0], self.lows[0], self.closes[0], 10)
        self.massi25        = massi(self.highs[0], self.lows[0], 9, 25)
        self.rvi10          = rvi(self.closes[0], self.highs[0], self.lows[0], 10, 7)
        self.gk10           = gk(self.opens[0], self.highs[0], self.lows[0], self.closes[0], 10, 252)

        self.bband20        = bband(self.ma20.ma, self.stddev20.stddev, 2)
        self.donchian20     = donchian(self.highs[0], self.lows[0], 20)
        self.keltner20      = keltner(self.ema20.ema, self.atr10.atr, 2)

        # performance
        self.logreturn      = logreturn(self.closes[0])
        self.candlestrength = candlestrength(self.opens[0], self.highs[0], self.lows[0], self.closes[0], self.volumes[0], self.vema10.ema, self.atr10.atr)
        self.timely         = timely(self.timestamp)

        # momentum
        self.rsi14          = rsi(self.closes[0], 14)
        self.stoch_rsi14    = stoch_rsi(self.closes[0], self.rsi14.rsi, 3, 3)
        self.macd           = macd(self.closes[0], self.ema12.ema, self.ema26.ema, 9)
        self.cci20          = cci(self.highs[0], self.lows[0], self.closes[0], self.ma20.ma, 20, 0.015)
        self.tsi_trend20    = tsi_trend(self.closes[0], self.ma20.ma, 20)
        self.tsi_true25     = tsi_true(self.closes[0], 25, 13)
        self.roc10          = roc(self.closes[0], 10)
        self.roc15          = roc(self.closes[0], 15)
        self.roc20          = roc(self.closes[0], 20)
        self.roc30          = roc(self.closes[0], 30)
        self.fisher9        = fisher(self.highs[0], self.lows[0], 9)
        self.cmo9           = cmo(self.closes[0], 9)
        self.adx14          = adx(self.highs[0], self.lows[0], self.closes[0], 14, 14)
        self.squeeze20      = squeeze(self.closes[0], self.bband20.upper_band, self.bband20.lower_band,
                                      self.keltner20.upper_band, self.keltner20.lower_band, 20)
        self.uo             = uo(self.closes[0], self.highs[0], self.lows[0], 7, 14, 28, 4, 2, 1)
        self.kst            = kst(self.roc10.roc, self.roc15.roc, self.roc20.roc, self.roc30.roc, [10,15,20,30], [1,2,3,4], 9)
        self.william_r14    = williams_r(self.highs[0], self.lows[0], self.closes[0], 14)
        self.william_r50    = williams_r(self.highs[0], self.lows[0], self.closes[0], 50)
        self.william_r200   = williams_r(self.highs[0], self.lows[0], self.closes[0], 200)
        
        # volume
        self.aobv           = aobv(self.opens[0], self.highs[0], self.lows[0], self.closes[0], self.volumes[0], 13)
        self.avwap          = avwap(self.highs[0], self.lows[0], self.closes[0], self.volumes[0], self.atr10.atr, 20, 0)
        self.eom            = eom(self.highs[0], self.lows[0], self.volumes[0], 14)

        self.feature_specs = {
            # Overlay indicators
            'ma20': {'instance': self.ma20, 'features': []},
            'ema9': {'instance': self.ema9, 'features': []},
            'ema12': {'instance': self.ema12, 'features': []},
            'ema20': {'instance': self.ema20, 'features': []},
            'ema26': {'instance': self.ema26, 'features': []},
            'vema10': {'instance': self.vema10, 'features': []},

            # Volatility indicators
            'stddev20': {'instance': self.stddev20, 'features': [('stddev', -1)]},
            'atr10': {'instance': self.atr10, 'features': [('atr', -1)]},
            'massi25': {'instance': self.massi25, 'features': [('mass_index', -1)]},
            'rvi10': {'instance': self.rvi10, 'features': [('rvi', -1)]},
            'gk10': {'instance': self.gk10, 'features': [('volatility', -1)]},
            'bband20': {'instance': self.bband20, 'features': []},  # Used by squeeze20
            'donchian20': {'instance': self.donchian20, 'features': []},
            'keltner20': {'instance': self.keltner20, 'features': []},  # Used by squeeze20

            # Performance indicators
            'logreturn': {
                'instance': self.logreturn,
                'features': [('log_returns', -1), ('log_returns', -2), ('log_returns', -3),
                            ('log_returns', -4), ('log_returns', -5)]
            },
            'candlestrength': {
                'instance': self.candlestrength,
                'features': [('strength', -1), ('strength', -2), ('strength', -3),
                            ('strength', -4), ('strength', -5),
                            ('tr_mult', -1), ('tr_mult', -2), ('tr_mult', -3),
                            ('tr_mult', -4), ('tr_mult', -5),
                            ('v_mult', -1), ('v_mult', -2), ('v_mult', -3),
                            ('v_mult', -4), ('v_mult', -5)]
            },
            'timely': {
                'instance': self.timely,
                'features': [('month_of_year', None), ('day_of_week', None), ('hour_of_day', None)]
            },

            # Momentum indicators
            'rsi14': {'instance': self.rsi14, 'features': [('rsi', -1)]},
            'stoch_rsi14': {'instance': self.stoch_rsi14, 'features': [('histogram', -1)]},
            'macd': {'instance': self.macd, 'features': [('histogram', -1)]},
            'cci20': {'instance': self.cci20, 'features': [('cci', -1)]},
            'tsi_trend20': {'instance': self.tsi_trend20, 'features': [('tsi', -1)]},
            'tsi_true25': {'instance': self.tsi_true25, 'features': [('tsi', -1)]},
            'roc10': {'instance': self.roc10, 'features': []},  # Used by kst
            'roc15': {'instance': self.roc15, 'features': [('roc', -1)]},  # Direct feature + used by kst
            'roc20': {'instance': self.roc20, 'features': []},  # Used by kst
            'roc30': {'instance': self.roc30, 'features': []},  # Used by kst
            'fisher9': {'instance': self.fisher9, 'features': [('fisher', -1)]},
            'cmo9': {'instance': self.cmo9, 'features': [('cmo', -1)]},
            'adx14': {'instance': self.adx14, 'features': [('adx', -1)]},
            'squeeze20': {
                'instance': self.squeeze20,
                'features': [('squeeze_rating', -1), ('momentum', -1)]
            },
            'uo': {'instance': self.uo, 'features': [('uo', -1)]},
            'kst': {'instance': self.kst, 'features': [('histogram', -1)]},
            'william_r14': {'instance': self.william_r14, 'features': [('wpr', -1)]},
            'william_r50': {'instance': self.william_r50, 'features': [('wpr', -1)]},
            'william_r200': {'instance': self.william_r200, 'features': [('wpr', -1)]},

            # Volume indicators
            'aobv': {'instance': self.aobv, 'features': [('histogram', -1)]},
            'avwap': {'instance': self.avwap, 'features': [('deviation', -1)]},
            'eom': {'instance': self.eom, 'features': [('emv', -1)]}
        }
        # Count total features and create array
        self.n_features = sum(len(spec['features']) for spec in self.feature_specs.values())
        self.current_row = 0
        if self._train:
            self.features_history = np.zeros((1_000_000, self.n_features), dtype=np.float32)
        else:
            self.features = np.zeros(self.n_features, dtype=np.float32)

        # =========================================================
        
        # Reusable bar dict template
        self._bar_template = {
            DATA_FIELD.FIELD_TIME: CTime(1990,1,1,0,0),
            DATA_FIELD.FIELD_OPEN: 0.0,
            DATA_FIELD.FIELD_HIGH: 0.0,
            DATA_FIELD.FIELD_LOW: 0.0,
            DATA_FIELD.FIELD_CLOSE: 0.0,
            DATA_FIELD.FIELD_VOLUME: 0
        }
        self.init = False
        
    def analyze(self, np_bars: WtNpKline):
        # if not isinstance(np_bars, WtNpKline):
        #     raise TypeError("Expected WtNpKline instance")

        klu_dict = self.parse_kline(np_bars)
        self.update_features()
        return klu_dict

    def update_features(self):
        # Simple iteration will follow the insertion order of the dictionary
        for spec in self.feature_specs.values():
            spec['instance'].update()
            
        if not self.init:
            if len(self.closes[0]) <= 5:
                return
            else:
                self.init = True
                
        if self._train:
            if self.current_row >= len(self.features_history):
                new_array = np.zeros((len(self.features_history) * 2, self.n_features), dtype=np.float32)
                new_array[:self.current_row] = self.features_history
                self.features_history = new_array
            
        feature_idx = 0
        for spec in self.feature_specs.values():
            instance = spec['instance']
            for attr_name, idx in spec['features']:
                value = getattr(instance, attr_name)
                # print(attr_name, idx, value)
                if self._train:
                    self.features_history[self.current_row, feature_idx] = value[idx] if idx is not None else value
                else:
                    self.features[feature_idx] = value[idx] if idx is not None else value

                feature_idx += 1
                
        self.current_row += 1
        # formatted_numbers = [f"{x:.5f}" for x in features]
        # print(formatted_numbers)
        
    def parse_kline(self, np_bars: WtNpKline) -> dict[KL_TYPE, List[CKLine_Unit]]:
        # def debug(i):
        #     print(i, curr_open, curr_high, curr_low, curr_close, curr_vol)
        # if not np_bars or len(np_bars.bartimes) == 0:
        #     raise ValueError("Empty bars data")
        
        # Extract bar data
        time_str = str(np_bars.bartimes[-1])
        time = CTime(
            int(time_str[:4]),
            int(time_str[4:6]),
            int(time_str[6:8]),
            int(time_str[8:10]),
            int(time_str[10:12]),
            auto=False
        )
        
        curr_open = np_bars.opens[-1]
        curr_high = np_bars.highs[-1]
        curr_low = np_bars.lows[-1]
        curr_close = np_bars.closes[-1]
        curr_vol = int(np_bars.volumes[-1])
        self.timestamp = time.ts
        
        # Reuse bar dict
        self._bar_template[DATA_FIELD.FIELD_TIME] = time
        self._bar_template[DATA_FIELD.FIELD_OPEN] = curr_open
        self._bar_template[DATA_FIELD.FIELD_HIGH] = curr_high
        self._bar_template[DATA_FIELD.FIELD_LOW] = curr_low
        self._bar_template[DATA_FIELD.FIELD_CLOSE] = curr_close
        self._bar_template[DATA_FIELD.FIELD_VOLUME] = curr_vol
        
        # Pre-allocate results with singleton list
        results = {
            KL_TYPE.K_1M: [CKLine_Unit(self._bar_template.copy(), autofix=True)]
        }
        
        # Use local references for speed
        opens, highs, lows, closes, volumes = self.opens, self.highs, self.lows, self.closes, self.volumes
        
        for i, level in enumerate(self.levels):
            count = self.counts[i]
            
            if count == 0:
                # Append new values and manage size
                opens[i].append(curr_open)
                highs[i].append(curr_high)
                lows[i].append(curr_low)
                closes[i].append(curr_close)
                volumes[i].append(curr_vol)
                
                # Maintain fixed size for all arrays
                LEN = 100
                if len(opens[i]) > 2*LEN: # optimize less updates
                    del opens[i][:-LEN]
                    del highs[i][:-LEN]
                    del lows[i][:-LEN]
                    del closes[i][:-LEN]
                    del volumes[i][:-LEN]
            else:
                # Update existing values without growing arrays
                highs[i][-1] = max(highs[i][-1], curr_high)
                lows[i][-1] = min(lows[i][-1], curr_low)
                closes[i][-1] = curr_close
                volumes[i][-1] += curr_vol
            
            count += 1
            
            if count >= self.multiplier[i]:
                # Update bar dict in place
                self._bar_template[DATA_FIELD.FIELD_TIME] = time
                self._bar_template[DATA_FIELD.FIELD_OPEN] = opens[i][-1]
                self._bar_template[DATA_FIELD.FIELD_HIGH] = highs[i][-1]
                self._bar_template[DATA_FIELD.FIELD_LOW] = lows[i][-1]
                self._bar_template[DATA_FIELD.FIELD_CLOSE] = closes[i][-1]
                self._bar_template[DATA_FIELD.FIELD_VOLUME] = volumes[i][-1]
                
                results[level] = [CKLine_Unit(self._bar_template.copy(), autofix=True)]
                
                # Update values for next level
                curr_open = opens[i][-1]
                curr_high = highs[i][-1]
                curr_low = lows[i][-1]
                curr_close = closes[i][-1]
                curr_vol = volumes[i][-1]
                
                # Reset counter and prepare for next bar
                self.counts[i] = 0
                # Reset volume for next bar
                if i < len(volumes) - 1:  # Only if there's a next level
                    volumes[i+1].append(0)  # Initialize next level's volume

            else:
                self.counts[i] = count
                break  # Changed from return to break to allow volume initialization
        return results
    
    def get_features_df(self):
        # Generate column names from feature specs
        feature_names = []
        for name, spec in self.feature_specs.items():
            for attr_name, idx in spec['features']:
                if idx is not None:
                    # For indexed features (like t-1, t-2, etc.)
                    feature_names.append(f"{name}_{attr_name}_{abs(idx)}")
                else:
                    # For direct features (like timely's hour_of_day)
                    feature_names.append(f"{name}_{attr_name}")

        return pd.DataFrame(
            self.features_history[:self.current_row],
            columns=feature_names
        )
