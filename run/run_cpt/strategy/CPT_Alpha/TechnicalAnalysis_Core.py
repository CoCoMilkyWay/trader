import array
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Set, Optional

from wtpy.WtDataDefs import WtNpKline
from Chan.Chan import CChan
from Chan.ChanConfig import CChanConfig
from Chan.Common.CEnum import KL_TYPE, DATA_FIELD, FX_TYPE, BI_DIR
from Chan.Common.CTime import CTime
from Chan.KLine.KLine_Unit import CKLine_Unit
from Chan.KLine.KLine_List import CKLine_List

from Math.Adaptive_SuperTrend import AdaptiveSuperTrend

from .Labels import ts_label
from .TechnicalAnalysis_Rules import TechnicalAnalysis_Rules, IndicatorManager, ParamType, IndicatorArg, ScalingMethod

class TechnicalAnalysis_Core:
    def __init__(self, code: str, train: bool = False, plot: bool = False):
        self._code = code
        self._train = train
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
        
        # Indicator management system
        self.indicator_manager = IndicatorManager(self)
        self._init_indicators()
        
        # Initialize feature storage
        self.n_features = sum(len(spec['features']) for spec in self.indicator_manager.feature_specs.values())
        self.n_labels = 1
        self._init_feature_storage()
        
        # special purpose indicators
        self.AdaptiveSuperTrend = AdaptiveSuperTrend(atr_len=60, factor=10) # 1 hours

        # labels
        self.ts_label = ts_label()

    def _init_indicators(self):
        """Define and initialize all technical indicators"""
        
        self.indicator_manager.process_definitions(TechnicalAnalysis_Rules.indicator_definitions)
        
    def _init_feature_storage(self):
        """Initialize feature storage based on specs"""
        if self._train:
            self.features_history = np.zeros((100_000, self.n_features + self.n_labels), dtype=np.float32)
        else:
            self.features = np.zeros(self.n_features, dtype=np.float32)
        self.current_row = 0

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
        feature_idx = 0
        for spec in self.indicator_manager.feature_specs.values():
            instance = spec['instance']
            for attr_name, idx in spec['features']:
                value = getattr(instance, attr_name)
                if self._train:
                    self.features_history[self.current_row, feature_idx] = value[idx] if idx else value
                else:
                    self.features[feature_idx] = value[idx] if idx else value
                feature_idx += 1

        self.current_row += 1
        # formatted_numbers = [f"{x:.5f}" for x in features]
        # print(formatted_numbers)

        s_long_switch, s_short_switch = self.AdaptiveSuperTrend.update(self.highs[0][-1], self.lows[0][-1], self.closes[0][-1], self.timestamp[-1])
        
        if self._train:
            self.ts_label.update(self.timestamp[-1], self.closes[0][-1], self.atr_10.atr[-1], s_long_switch, s_short_switch) # type: ignore

    def parse_kline(self, open:float, high:float, low:float, close:float, vol:float, time:int) -> dict[KL_TYPE, List[CKLine_Unit]]:
        # def debug(i):
        #     print(i, curr_open, curr_high, curr_low, curr_close, curr_vol)
        # if not np_bars or len(np_bars.bartimes) == 0:
        #     raise ValueError("Empty bars data")
        
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
                print(f'{self._code}:{day}-{hour}-{minute}')
            
        curr_open = open
        curr_high = high
        curr_low = low
        curr_close = close
        curr_vol = vol
        self.timestamp[-1] = ctime.ts
        
        # Reuse bar dict
        self._bar_template[DATA_FIELD.FIELD_TIME] = ctime
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
                self._bar_template[DATA_FIELD.FIELD_TIME] = ctime
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
        # Generate column names and scaling methods from feature specs
        feature_names = []
        scaling_methods = {}

        for indicator_name, spec in self.indicator_manager.feature_specs.items():
            # Get scaler once per indicator (assumes all features from same indicator use same scaler)
            scaler = spec.get('scaler', ScalingMethod.NONE)  # Default to no scaling

            for attr_spec in spec['features']:
                # Handle attribute/index specification
                attr_name, idx = attr_spec if isinstance(attr_spec, tuple) else (attr_spec, None)

                # Create descriptive column name
                if idx is not None:
                    col_name = f"{indicator_name}_{attr_name}|{abs(idx)}"
                else:
                    col_name = f"{indicator_name}_{attr_name}"

                feature_names.append(col_name)
                scaling_methods[col_name] = scaler  # Associate scaling method with column

        # Create label column names
        label_names = [f'label_{i+1}' for i in range(self.n_labels)]

        # Create DataFrame with proper typing
        df = pd.DataFrame(
            data=self.features_history[:self.current_row, :self.n_features],  # Exclude labels from features
            columns=feature_names,
            dtype=np.float32
        )

        # Add labels if in training mode
        if self._train and self.n_labels > 0:
            labels = self.features_history[:self.current_row, self.n_features:]
            df[label_names] = labels.astype(np.float32)

        return df, scaling_methods

# Example usage
if __name__ == "__main__":
    ta = TechnicalAnalysis_Core("TEST_CODE", train=True)
    # Use with real data...
