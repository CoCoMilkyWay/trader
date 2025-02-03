import copy
from enum import Enum
from pprint import pprint
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Any, Union, Set, Tuple, Optional

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
from Math.momentum.aroon import aroon
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
from Math.momentum.td_seq import td_seq
from Math.volume.aobv import aobv
from Math.volume.avwap import avwap
from Math.volume.eom import eom

from Math.models.pytorch_model import \
    ScalingMethod, SplitMethod, \
    DataCheckResult, ModelType, GeneralizedModel, \
    CNN, Recurrent, Transformer, Ensemble


class ParamType(Enum):
    SOURCE = 1       # Data source like closes[0]
    REFERENCE = 2    # Other indicator reference
    LITERAL = 3      # Literal value


@dataclass
class IndicatorArg:
    type: ParamType
    value: Any
    attr: Optional[str] = None


class TechnicalAnalysis_Rules:
    def __init__(self):
        """
        dummy indicator(only instantiated after called as dependency):
            1. with param = None
            2. empty features
        """
        # Technical indicator configurations
        self.indicator_definitions = [
            # Overlay indicators
            {
                'param0': None,
                'name': ('ma', 'param0'),
                'constructor': ma,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': []
            },
            {
                'param0': None,
                'name': ('ema', 'param0'),
                'constructor': ema,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': []
            },
            {
                'param0': None,
                'name': ('vema', 'param0'),
                'constructor': ema,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('volumes', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': []
            },

            # Volatility indicators
            {
                'param0': 20,
                'name': ('stddev', 'param0'),
                'constructor': stddev,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.REFERENCE, ('ma', 'param0'), 'ma'),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': [('stddev', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': 10,
                'name': ('atr', 'param0'),
                'constructor': atr,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': [('atr', -1)],
                'scaler': ScalingMethod.ROBUST
            },
            {
                'param0': 9,
                'param1': 25,
                'name': ('massi', 'param0', 'param1'),
                'constructor': massi,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                    IndicatorArg(ParamType.LITERAL, 'param1'),
                ],
                'features': [('mass_index', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': 10,
                'param1': 7,
                'name': ('rvi', 'param0', 'param1'),
                'constructor': rvi,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                    IndicatorArg(ParamType.LITERAL, 'param1'),
                ],
                'features': [('rvi', -1)],
                'scaler': ScalingMethod.ROBUST
            },
            {
                'param0': 10,
                'name': ('gk', 'param0'),
                'constructor': gk,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('opens', 0)),
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                    IndicatorArg(ParamType.LITERAL, 252),
                ],
                'features': [('volatility', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': None, #20,
                'param1': None, #2,
                'name': ('bband', 'param0', 'param1'),
                'constructor': bband,
                'args': [
                    IndicatorArg(ParamType.REFERENCE, ('ma', 'param0'), 'ma'),
                    IndicatorArg(ParamType.REFERENCE, ('stddev', 'param0'), 'stddev'),
                    IndicatorArg(ParamType.LITERAL, 'param1'),
                ],
                'features': []
            },
            {
                'param0': None, #20,
                'name': ('donchian', 'param0'),
                'constructor': donchian,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': []
            },
            {
                'param0': None, #20,
                'param1': None, #2,
                'name': ('keltner', 'param0', 'param1'),
                'constructor': keltner,
                'args': [
                    IndicatorArg(ParamType.REFERENCE, ('ema', 'param0'), 'ema'),
                    IndicatorArg(ParamType.REFERENCE, ('atr', 10), 'atr'),
                    IndicatorArg(ParamType.LITERAL, 'param1'),
                ],
                'features': []
            },

            # Performance indicators
            {
                'name': ('logreturn',),
                'constructor': logreturn,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                ],
                'features': [
                    ('log_returns', -1), ('log_returns', -2), ('log_returns', -3),
                    ('log_returns', -4), ('log_returns', -5)
                ],
                'scaler': ScalingMethod.ROBUST
            },
            {
                'param0': 10,
                'name': ('candlestrength', 'param0'),
                'constructor': candlestrength,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('opens', 0)),
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.SOURCE, ('volumes', 0)),
                    IndicatorArg(ParamType.REFERENCE, ('vema', 'param0'), 'ema'),
                    IndicatorArg(ParamType.REFERENCE, ('atr', 'param0'), 'atr'),
                ],
                'features': [
                    ('strength', -1), ('strength', -2), ('strength', -3),
                    ('strength', -4), ('strength', -5),
                    ('tr_mult', -1), ('tr_mult', -2), ('tr_mult', -3),
                    ('tr_mult', -4), ('tr_mult', -5),
                    ('v_mult', -1), ('v_mult', -2), ('v_mult', -3),
                    ('v_mult', -4), ('v_mult', -5)
                ],
                'scaler': ScalingMethod.ROBUST
            },
            {
                'name': ('timely',),
                'constructor': timely,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('timestamp',)),
                ],
                'features': [('day_of_week', None), ('hour_of_day', None)],
                'scaler': ScalingMethod.STANDARD
            },

            # Momentum indicators
            {
                'param0': 14,
                'name': ('rsi', 'param0'),
                'constructor': rsi,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': [('rsi', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': 3,
                'name': ('stoch_rsi', 14),
                'constructor': stoch_rsi,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.REFERENCE, ('rsi', 14), 'rsi'),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': [('histogram', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': 9,
                'name': ('macd', 'param0'),
                'constructor': macd,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.REFERENCE, ('ema', 12), 'ema'),
                    IndicatorArg(ParamType.REFERENCE, ('ema', 26), 'ema'),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': [('histogram', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': 25,
                'name': ('aroon', 'param0'),
                'constructor': aroon,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': [('aroon_up', -1), ('aroon_down', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': 20,
                'name': ('cci', 'param0'),
                'constructor': cci,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.REFERENCE, ('ma', 'param0'), 'ma'),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                    IndicatorArg(ParamType.LITERAL, 0.015),
                ],
                'features': [('cci', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': 20,
                'name': ('tsi_trend', 'param0'),
                'constructor': tsi_trend,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.REFERENCE, ('ma', 'param0'), 'ma'),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': [('tsi', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': 25,
                'param1': 13,
                'name': ('tsi_true', 'param0', 'param1'),
                'constructor': tsi_true,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                    IndicatorArg(ParamType.LITERAL, 'param1'),
                ],
                'features': [('tsi', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': None,
                'name': ('roc', 'param0'),
                'constructor': roc,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': []
            },
            {
                'param0': 9,
                'name': ('fisher', 'param0'),
                'constructor': fisher,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': [('fisher', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': 9,
                'name': ('cmo', 'param0'),
                'constructor': cmo,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': [('cmo', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': 14,
                'name': ('adx', 'param0'),
                'constructor': adx,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': [('adx', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': 20,
                'name': ('squeeze', 'param0'),
                'constructor': squeeze,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.REFERENCE, ('bband', 'param0', 2), 'upper_band'),
                    IndicatorArg(ParamType.REFERENCE, ('bband', 'param0', 2), 'lower_band'),
                    IndicatorArg(ParamType.REFERENCE, ('keltner', 'param0', 2), 'upper_band'),
                    IndicatorArg(ParamType.REFERENCE, ('keltner', 'param0', 2), 'lower_band'),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': [('squeeze_rating', -1), ('momentum', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'name': ('uo',),
                'constructor': uo,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.LITERAL, 7 ),
                    IndicatorArg(ParamType.LITERAL, 14),
                    IndicatorArg(ParamType.LITERAL, 28),
                    IndicatorArg(ParamType.LITERAL, 4 ),
                    IndicatorArg(ParamType.LITERAL, 2 ),
                    IndicatorArg(ParamType.LITERAL, 1 ),
                ],
                'features': [('uo', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'name': ('kst',),
                'constructor': kst,
                'args': [
                    IndicatorArg(ParamType.REFERENCE, ('roc', 10), 'roc'),
                    IndicatorArg(ParamType.REFERENCE, ('roc', 15), 'roc'),
                    IndicatorArg(ParamType.REFERENCE, ('roc', 20), 'roc'),
                    IndicatorArg(ParamType.REFERENCE, ('roc', 30), 'roc'),
                    IndicatorArg(ParamType.LITERAL, [10, 15, 20, 30]),
                    IndicatorArg(ParamType.LITERAL, [1, 2, 3, 4]),
                    IndicatorArg(ParamType.LITERAL, 9),
                ],
                'features': [('histogram', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': 14,
                'name': ('william_r', 'param0'),
                'constructor': williams_r,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': [('wpr', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': 50,
                'name': ('william_r', 'param0'),
                'constructor': williams_r,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': [('wpr', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': 200,
                'name': ('william_r', 'param0'),
                'constructor': williams_r,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': [('wpr', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'name': ('td_seq',),
                'constructor': td_seq,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                ],
                'features': [('setup_index', None)],
                'scaler': ScalingMethod.STANDARD
            },

            # Volume indicators
            {
                'param0': 13,
                'name': ('aobv', 'param0'),
                'constructor': aobv,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('opens', 0)),
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.SOURCE, ('volumes', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': [('histogram', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': 20,
                'name': ('avwap', 'param0'),
                'constructor': avwap,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.SOURCE, ('closes', 0)),
                    IndicatorArg(ParamType.SOURCE, ('volumes', 0)),
                    IndicatorArg(ParamType.REFERENCE, ('atr', 10), 'atr'),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                    IndicatorArg(ParamType.LITERAL, 0),
                ],
                'features': [('deviation', -1)],
                'scaler': ScalingMethod.STANDARD
            },
            {
                'param0': 14,
                'name': ('eom', 'param0'),
                'constructor': eom,
                'args': [
                    IndicatorArg(ParamType.SOURCE, ('highs', 0)),
                    IndicatorArg(ParamType.SOURCE, ('lows', 0)),
                    IndicatorArg(ParamType.SOURCE, ('volumes', 0)),
                    IndicatorArg(ParamType.LITERAL, 'param0'),
                ],
                'features': [('emv_osc', -1)],
                'scaler': ScalingMethod.STANDARD
            },
        ]

class IndicatorManager:
    """
    Manages technical indicator creation and dependencies

    2 types:
        1. dummy constructor: with None in name, only define a constructor with no real instance 
            until called as an dependency with parameters overwriting None
        2. real constructor: with real parameter in name, thus instantiate immediately (dependencies first, then itself)

        note that both can be instanced with different parameters
    """

    def __init__(self, parent):
        self.parent = parent  # where the indicators are instanced
        self.indicator_registry: Dict[str, Any] = {}
        self.feature_specs: Dict[str, Dict] = {}

    def process_definitions(self, definitions: List[Dict]) -> None:
        """Process indicator definitions, separating templates and concrete indicators"""

        # Process concrete definitions with topological sort
        real_definitions = {} # definitions with valid parameters and features
        dummy_definitions = {} # all definitions with empty parameters
        for defn in definitions:
            real_name, dummy_name, real = self._get_indicator_name(defn['name'], defn)
            
            if real:
                real_definitions[real_name] = copy.deepcopy(defn)
            
            if dummy_name not in dummy_definitions.keys():
                defn = copy.deepcopy(defn)
                for key in list(defn.keys()):
                    if key.startswith('param'):
                        defn[key] = None # dummify parameters
                dummy_definitions[dummy_name] = defn
            
        # build the complete real dependency tree
        dependencies: Dict[str, Set[str]] = defaultdict(set)
        reverse_deps: Dict[str, Set[str]] = defaultdict(set)

        # Check if a definition has dependencies and add them recursively
        def add_dependencies(name_str: str, defn: Dict):
            for arg in defn['args']:
                if arg.type == ParamType.REFERENCE:
                    real_dep, dummy_dep, real = self._get_indicator_name(arg.value, defn)
                    # print(f'add: {name_str} -> {real_dep}')
                    if not real:
                        raise ValueError(
                        f"{name_str}: Check Indicator Dependencies Definition")
                    dependencies[name_str].add(real_dep)
                    reverse_deps[real_dep].add(name_str)
                    
                    # check if the dependencies are defined
                    if real_dep not in real_definitions.keys():
                        # instance from the dummy definitions
                        real_definitions[real_dep] = copy.deepcopy(dummy_definitions[dummy_dep])
                        # pass parameters
                        for key in list(real_definitions[real_dep].keys()):
                            if key.startswith('param'):
                                try:
                                    i = int(key.lstrip("param"))
                                    real_definitions[real_dep][key] = int(real_dep.split(sep='_')[i+1])
                                except Exception as e:
                                    raise KeyError(
                                    f"Parameter({key}) mismatch building: {name_str} -> {real_dep}")
                                    
                    # Recursively add dependencies of the dependency if it exists in real_definitions
                    add_dependencies(real_dep, real_definitions[real_dep])

        for real_name, defn in list(real_definitions.items()):  # iterate over copy as it dynamically changes
            add_dependencies(real_name, defn)

        # roots: real indicators with no dep
        roots = {name_str for name_str in real_definitions.keys()
                 if not dependencies[name_str]}
        
        # print(f'dummy_definitions:'); pprint(dummy_definitions.keys())
        print(f'real_definitions:'); pprint(real_definitions.keys())
        # print(f'Dependency tree:'); pprint(dependencies)
        # print(f'Reverse_deps tree:'); pprint(reverse_deps)
        # print(f'Roots:'); pprint(roots)
        
        processed = set()
        queue = list(roots)  # Start with root nodes
        
        while queue: # bottom(roots) -> top method
            current_node = queue.pop(0)
            if current_node in processed:
                continue  # Skip if already processed
            
            # Get pre-resolved definition from real_definitions
            defn = real_definitions[current_node]
            
            # Instance the indicator
            self._create_indicator(current_node, defn)
            processed.add(current_node)
            
            # Check dependents(top) have all dependencies(bottom) instanced
            for dependent in reverse_deps.get(current_node, []):
                if all(dep in processed for dep in dependencies[dependent]):
                    if dependent not in processed and dependent not in queue:
                        queue.append(dependent)
        
        # Verify all real_definitions were processed
        missing = set(real_definitions.keys()) - processed
        if missing:
            raise ValueError(f"Failed to process indicators: {missing}. Circular dependency or missing definition.")
        
    def _get_indicator_name(self, name_tuple, defn: Dict) -> Tuple[str, str, bool]:
        """Generate unique indicator name from name specification"""
        is_real = True
        if isinstance(name_tuple, tuple):
            base, params_raw = name_tuple[0], name_tuple[1:]
            params = []
            for param_raw in params_raw:
                if type(param_raw) == str: # "param0", "param1", ...
                    param_parsed = defn[param_raw]
                    if param_parsed is None: # "param0" == None
                        is_real = False # this is just a constructor
                        continue
                    else:
                        params.append(param_parsed)
                else:
                    params.append(param_raw)
                    
            if len(params) > 0:
                real_name = f"{base}_{"_".join(map(str, params))}"
            else:
                real_name = f"{base}"
            dummy_name = f"{base}"
            return real_name, dummy_name, is_real
        return str(name_tuple), str(name_tuple), is_real

    def _create_indicator(self, real_name: str, definition: Dict) -> None:
        """Create indicator instance with resolved parameters"""
        # 1. Resolve indicator name with actual parameters
        if real_name in self.indicator_registry:
            return

        # 2. Resolve arguments with parameter substitution
        resolved_args = []
        for arg in definition['args']:
            if arg.type == ParamType.LITERAL:
                # Handle parameter references like 'param0'
                value = arg.value
                if isinstance(value, str) and value.startswith('param'):
                    resolved_value = definition[value]
                else:
                    resolved_value = int(value) if type(value) is str else value
                resolved_args.append(resolved_value)
            elif arg.type == ParamType.REFERENCE:
                # Dependency already exists due to topological order
                real_dep, dummy_dep, real = self._get_indicator_name(arg.value, definition)
                resolved_args.append(getattr(self.indicator_registry[real_dep], arg.attr))
            elif arg.type == ParamType.SOURCE:
                # Directly access parent data series
                if len(arg.value) == 2: # 0:1min, 1:5min
                    src, idx = arg.value
                    resolved_args.append(getattr(self.parent, src)[idx])
                elif len(arg.value) == 1:
                    src = arg.value
                    resolved_args.append(getattr(self.parent, src[0]))
                else:
                    raise ValueError(
                        f"Failed to parse source arg {real_name}: {arg}")
            else:
                raise ValueError(f"Unknown argument type {arg.type}")

        # 3. Instantiate and register
        # print(f'creating {real_name}: {resolved_args}')
        instance = definition['constructor'](*resolved_args)
        self.indicator_registry[real_name] = instance
        setattr(self.parent, real_name, instance)

        # 4. Store feature configuration
        if 'features' in definition:
            self.feature_specs[real_name] = {
                'instance': instance,
                'features': definition['features'],
                'scaler': definition.get('scaler')
                }