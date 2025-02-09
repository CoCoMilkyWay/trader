import copy
import array
import torch
from typing import List, Dict, Any, Union, Set, Optional

from Chan.Chan import CChan
from Chan.ChanConfig import CChanConfig
from Chan.Common.CEnum import KL_TYPE, DATA_FIELD, FX_TYPE, BI_DIR
from Chan.Common.CTime import CTime
from Chan.KLine.KLine_Unit import CKLine_Unit
from Chan.KLine.KLine_List import CKLine_List

from Math.Adaptive_SuperTrend import AdaptiveSuperTrend

from .Labels import ts_label, NUM_LABELS
from .TechnicalAnalysis_Rules import TechnicalAnalysis_Rules, IndicatorManager, ParamType, IndicatorArg, ScalingMethod

class TechnicalAnalysis_Core:
    def __init__(self, code:str, code_idx:int, shared_tensor:torch.Tensor, plot:bool=False):
        self._code = code
        self._code_idx = code_idx
        self._timestamp_idx = 0
        self.shared_tensor = shared_tensor
        self._plot = plot
        self.config: CChanConfig = CChanConfig()
        self.lv_list = [lv[0] for lv in self.config.lv_list]
        self.levels = tuple(reversed(self.lv_list))
        self.multiplier = tuple(lv[1] for lv in reversed(self.config.lv_list))
        
        if self._plot:
            # Timer
            self.day = None
            self.hour = None
            self.minute = None
        
        # Initialize data storage
        n_levels = len(self.levels)
        self.counts = array.array('I', [0] * n_levels)
        self.opens = [array.array('d', [0.0]) for _ in range(n_levels)]
        self.highs = [array.array('d', [0.0]) for _ in range(n_levels)]
        self.lows = [array.array('d', [0.0]) for _ in range(n_levels)]
        self.closes = [array.array('d', [0.0]) for _ in range(n_levels)]
        self.volumes = [array.array('d', [0]) for _ in range(n_levels)]
        self.timestamp = array.array('d', [0.0])
        
        # Initialize core components
        self.kl_datas: Dict[KL_TYPE, CKLine_List] = {
            level: CKLine_List(level, conf=self.config) for level in self.lv_list
        }
        self.chan_snapshot = CChan(
            code=code,
            kl_datas=self.kl_datas,
            lv_list=self.lv_list,
        )
        
        # Reusable bar dict template
        self._bar_template = {
            DATA_FIELD.FIELD_TIME: CTime(1990,1,1,0,0),
            DATA_FIELD.FIELD_OPEN: 0.0,
            DATA_FIELD.FIELD_HIGH: 0.0,
            DATA_FIELD.FIELD_LOW: 0.0,
            DATA_FIELD.FIELD_CLOSE: 0.0,
            DATA_FIELD.FIELD_VOLUME: 0
        }
        
        # 0:1M, 1:5M, 2:15M, ...
        i = 0
        self.i = i
        self.lv = self.levels[i]
        self.init = False
        
        self._init_indicators()
        self._init_features_labels_and_scalers()
        
        # non-feature indicators
        self.AdaptiveSuperTrend = AdaptiveSuperTrend(atr_len=60, factor=10) # 1 hours
        
        # labels (NOTE: int are not mutable)
        tensor_pointer = {
            'x': lambda: self._timestamp_idx,
            'y': lambda: self.n_features,
            'z': lambda: self._code_idx,}
        self.ts_label = ts_label(self.shared_tensor, tensor_pointer)
        
    def _init_indicators(self):
        """Define and initialize all technical indicators"""
        # Indicator management system
        self.indicator_manager = IndicatorManager(self, self._code_idx)
        self.indicator_manager.process_definitions(TechnicalAnalysis_Rules.indicator_definitions)
        
    def _init_features_labels_and_scalers(self):
        # Generate column names and scaling methods from feature specs
        self.n_features = sum(len(spec['features']) for spec in self.indicator_manager.feature_specs.values())
        self.n_labels = NUM_LABELS
        self.feature_names = []
        self.feature_map = []
        self.scaling_methods = {} # use dict just to be safe
        for indicator_name, spec in self.indicator_manager.feature_specs.items():
            # Get scaler once per indicator (assumes all features from same indicator use same scaler)
            scaler = spec.get('scaler', ScalingMethod.NONE)
            for attr_name, idx in spec['features']:
                # Create descriptive column name
                if idx is not None:
                    col_name = f"{indicator_name}_{attr_name}_{abs(idx)}"
                else:
                    col_name = f"{indicator_name}_{attr_name}"
                
                self.feature_names.append(col_name)
                self.feature_map.append((spec['instance'], attr_name, idx if idx else None))
                self.scaling_methods[col_name] = scaler  # Associate scaling method with column
                
        # Create label column names
        self.label_names = [f'label_{i+1}' for i in range(self.n_labels)]
        
        # Create template for fast instance
        self.list_template = [0.0 for i in range(self.n_features + self.n_labels)]
        
    def analyze(self, open:float, high:float, low:float, close:float, vol:float, time:int):
        """Main analysis workflow"""
        klu_dict = self.parse_kline(open, high, low, close, vol, time)
        
        # process Chan elements (generates Bi)
        self.chan_snapshot.trigger_load(klu_dict)
        
        # process PA elements
        for lv in self.lv_list:
            self.kl_datas[lv].PA_Core.parse_dynamic_bi_list()
            
        if self.lv in klu_dict.keys():
            self.update_features_and_labels()
            
        # update timestamp idx
        self._timestamp_idx += 1
            
    def update_features_and_labels(self):
        """Update all features and labels"""
        # Update all indicators
        for instance in self.indicator_manager.indicator_registry.values():
            instance.update()
            
        if not self.init:
            if len(self.closes[self.i]) <= 5:
                return
            else:
                self.init = True
                
        # Store features
        result_list = copy.deepcopy(self.list_template)
        for idx, (instance, attr_name, delay) in enumerate(self.feature_map):
            feature_time_series = getattr(instance, attr_name)
            result_list[idx] = feature_time_series[delay] if delay else feature_time_series
        for i in range(self.n_labels):
            result_list[self.n_features + i] = 0.0
            
        # Update the selected slice (features, codes, timestamps)
        result_vector = torch.tensor(result_list, dtype=torch.float16)
        self.shared_tensor[self._timestamp_idx, :, self._code_idx,] = result_vector
        
        # formatted_result = [f"{x:.5f}" for x in result_list]
        # print(formatted_result)
        
        s_long_switch, s_short_switch = self.AdaptiveSuperTrend.update(self.highs[0][-1], self.lows[0][-1], self.closes[0][-1], self.timestamp[-1])
        
        # if self._train:
        self.ts_label.update(self.timestamp[-1], self.closes[0][-1], self.atr_10.atr[-1], s_long_switch, s_short_switch) # type: ignore
        
    def parse_kline(self, open:float, high:float, low:float, close:float, vol:float, time:int) -> dict[KL_TYPE, List[CKLine_Unit]]:
        # Extract bar data
        time_str = str(time)
        # df['date'] = df['bartime'].astype(str).str[:8].astype(int)
        # df['time'] = df['bartime']-199000000000
        year    = int(time_str[-10:-8]) + 1990
        month   = int(time_str[-8:-6])
        day     = int(time_str[-6:-4])
        hour    = int(time_str[-4:-2])
        minute  = int(time_str[-2:])
        ctime = CTime(year, month, day, hour, minute, auto=False)
        
        if self._plot:
            if hour != self.hour:
                self.day = day
                self.hour = hour
                self.minute  = minute 
                print(f'{self._code:<22}:{year:04}-{month:>02}-{day:>02}-{hour:>02}-{minute:>02}')
                
        self.timestamp[-1] = ctime.ts
        
        # Reuse bar dict
        self._bar_template[DATA_FIELD.FIELD_TIME] = ctime
        self._bar_template[DATA_FIELD.FIELD_OPEN] = open
        self._bar_template[DATA_FIELD.FIELD_HIGH] = high
        self._bar_template[DATA_FIELD.FIELD_LOW] = low
        self._bar_template[DATA_FIELD.FIELD_CLOSE] = close
        self._bar_template[DATA_FIELD.FIELD_VOLUME] = vol
        
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
                opens[i].append(open)
                highs[i].append(high)
                lows[i].append(low)
                closes[i].append(close)
                volumes[i].append(vol)
                
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
                highs[i][-1] = max(highs[i][-1], high)
                lows[i][-1] = min(lows[i][-1], low)
                closes[i][-1] = close
                volumes[i][-1] += vol
                
            count += 1
            
            if count >= self.multiplier[i]:
                # Update bar dict in place
                self._bar_template[DATA_FIELD.FIELD_TIME] = ctime
                self._bar_template[DATA_FIELD.FIELD_OPEN] = opens[i][-1]
                self._bar_template[DATA_FIELD.FIELD_HIGH] = highs[i][-1]
                self._bar_template[DATA_FIELD.FIELD_LOW] = lows[i][-1]
                self._bar_template[DATA_FIELD.FIELD_CLOSE] = closes[i][-1]
                self._bar_template[DATA_FIELD.FIELD_VOLUME] = volumes[i][-1]
                
                results[level] = [CKLine_Unit(self._bar_template.copy(), autofix=True)]
                
                # Update values for next level
                open = opens[i][-1]
                high = highs[i][-1]
                low = lows[i][-1]
                close = closes[i][-1]
                vol = volumes[i][-1]
                
                # Reset counter and prepare for next bar
                self.counts[i] = 0
                # Reset volume for next bar
                if i < len(volumes) - 1:  # Only if there's a next level
                    volumes[i+1].append(0)  # Initialize next level's volume
            else:
                self.counts[i] = count
                break  # Changed from return to break to allow volume initialization
        return results