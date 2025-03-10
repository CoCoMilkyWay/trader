import copy
import torch
from torch import Tensor
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from Mining.Expression.Dimension import Dimension, Dimension_Map
from Mining.Util.Data_Util import list_timestamps
from Mining.Util.Data_Util import count_nan_and_inf


class Data:
    # path = f"{os.path.dirname(__file__)}/Data/Example/TimeSeries"
    def __init__(
        self,
        path: str,
        max_past: int,
        max_future: int,
        init: bool = False,
    ):
        self.path = path
        self.max_past = max_past
        self.max_future = max_future
        if init:
            self.init()

    def init(self):
        path: str = self.path
        max_past: int = self.max_past
        max_future: int = self.max_future
        # (N_timestamps, N_columns, N_codes)
        tensor = torch.load(f'{path}/tensor.pt', weights_only=True)
        num_nan, num_inf = count_nan_and_inf(tensor)
        print(f"Features and Labels: nan:{num_nan} inf:{num_inf}")
        tensor[torch.isinf(tensor)] = 0
        meta = torch.load(f'{path}/meta.pt', weights_only=True)
        self.device: torch.device = torch.device("cpu:0")
        self.dtype = torch.float

        # from pprint import pprint
        # pprint(meta)

        timestamps_key = 'timestamps'
        features_key = 'features'
        labels_key = 'labels'
        codes_key = 'codes'

        # actually not necessary, will affect metric accuracy
        # if you are testing higher timeframe (e.g. timestamps/weeks)
        # mostly be okay if you are testing on minute-bar
        self.max_past = max_past
        self.max_future = max_future

        shape = tensor.shape
        self.n_timestamps = shape[0] - self.max_past - self.max_future
        self.n_columns = shape[1]
        self.n_codes = shape[2]
        self.n_features = len(meta[features_key])
        self.n_labels = len(meta[labels_key])

        assert (self.n_columns == self.n_features + self.n_labels)
        assert (self.n_codes == len(meta[codes_key]))

        self.timestamps:List[datetime] = []
        self.codes:List[str] = []
        self.features:List[str] = []
        self.labels:List[str] = []
        self.dimensions:List[Dimension] = []
        self.scalar:List[str] = []

        n_timestamps, start, end = meta[timestamps_key]
        self.timestamps = list_timestamps(start, end, True)
        assert (n_timestamps == len(self.timestamps) == (
            self.n_timestamps+self.max_past+self.max_future))
        for feature, dimension, scalar in meta[features_key]:
            self.features.append(feature)
            self.dimensions.append(Dimension_Map(dimension))
            self.scalar.append(scalar)

        for label, in meta[labels_key]:
            self.labels.append(label)

        self.codes = meta[codes_key]

        # strip labels
        self.features_tensor: Tensor = \
            tensor[:,:self.n_features, :].to(self.dtype)
        self.labels_tensor: Tensor = \
            tensor[:, self.n_features:self.n_columns, :].to(self.dtype)

        print(f"Data Start:{start}, End:{end}")
        print(f"Num of Features:{self.n_features}, Labels:{self.n_labels}")
        print(self.codes)

    def __getitem__(self, slc: slice) -> "Data":
        "Get a subview of the data given a date slice or an index slice."
        if slc.step is not None:
            raise ValueError("Only support slice with step=None")
        if isinstance(slc.start, str):
            return self[self.find_datetime_slice(slc.start, slc.stop)]
        expanded_start, expanded_stop = slc.start, slc.stop
        # reserve the past/future data
        expanded_start = (
            expanded_start if expanded_start is not None else 0) - self.max_past
        expanded_stop = (
            expanded_stop if expanded_stop is not None else self.n_timestamps) + self.max_future
        expanded_start = max(0, expanded_start)
        expanded_stop = min(self.n_timestamps, expanded_stop)
        idx_range = slice(expanded_start, expanded_stop)
        # (N_timestamps, N_columns, N_codes)
        features_tensor = self.features_tensor[idx_range]
        # (N_timestamps, N_columns, N_codes)
        labels_tensor = self.labels_tensor[idx_range]
        # This is for removing code that is non-existent in the period of choice
        # (NaN value on all features/labels)
        remaining = features_tensor.isnan().\
            reshape(-1, features_tensor.shape[-1]).\
            all(dim=0).logical_not().nonzero().flatten()
        features_tensor = features_tensor[:, :, remaining]
        labels_tensor = labels_tensor[:, :, remaining]
        real_start = expanded_start + self.max_past
        real_end = expanded_stop - self.max_future - 1
        return self.deepcopy_with_time_index(real_start, real_end, features_tensor, labels_tensor)

    def find_datetime_slice(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> slice:
        """
        Find a slice of indices corresponding to the given date range.
        For the input, both ends are inclusive. The output is a normal left-closed right-open slice.
        """
        start = None if start_time is None else self.find_date_index(
            start_time)
        stop = None if end_time is None else self.find_date_index(
            end_time, exclusive=False)
        return slice(start, stop)

    def find_date_index(self, time: str, exclusive: bool = False) -> int:
        dt = datetime.strptime(time, "%Y-%m-%d")
        idx: int = self.timestamps.searchsorted(dt)  # type: ignore
        if exclusive and self.timestamps[idx] == dt:
            idx += 1
        if idx < 0 or idx > self.n_timestamps:
            raise ValueError(
                f"Datetime {time} is out of range: available [{self.timestamps[0]}, {self.timestamps[-1]}]")
        return idx

    def deepcopy_with_time_index(self, start: int, end: int, features_tensor: Tensor, labels_tensor: Tensor):
        """Create a deepcopy of the instance, excluding specified attributes."""
        # Create a new instance with just the relevant attributes
        new_instance = self.__class__(
            self.path, self.max_past, self.max_future)
        exclude_attrs = ['n_timestamps', 'timestamps',
                         'features_tensor', 'labels_tensor']
        # Copying the remaining attributes
        for attr, value in self.__dict__.items():
            if attr not in exclude_attrs:
                setattr(new_instance, attr, copy.deepcopy(value))
        new_instance.timestamps = self.timestamps[start:end]
        new_instance.n_timestamps = len(new_instance.timestamps)
        new_instance.features_tensor = features_tensor
        new_instance.labels_tensor = labels_tensor
        return new_instance


"""
meta = {
 'codes': [('Binance.UM.BTCUSDT',),
           ('Binance.UM.ETHUSDT',),
           ('Binance.UM.BCHUSDT',),
           ('Binance.UM.XRPUSDT',),
           ('Binance.UM.EOSUSDT',),
           ('Binance.UM.LTCUSDT',),
           ('Binance.UM.TRXUSDT',),
           ('Binance.UM.ETCUSDT',),
           ('Binance.UM.LINKUSDT',),
           ('Binance.UM.XLMUSDT',)],
 'features': [('eom_14_emv_osc_1', 'ratio', 'ScalingMethod.STANDARD'),
              ('logreturn_log_returns_1', 'ratio', 'ScalingMethod.ROBUST'),
              ('gk_10_volatility_1', 'misc', 'ScalingMethod.STANDARD'),
              ('rsi_14_rsi_1', 'oscillator', 'ScalingMethod.STANDARD'),
              ('td_seq_setup_index', 'oscillator', 'ScalingMethod.STANDARD'),
              ('massi_9_25_mass_index_1',
               'oscillator',
               'ScalingMethod.STANDARD'),
              ('roc_10_roc_1', 'oscillator', 'None'),
              ('aroon_25_aroon_up_1', 'oscillator', 'ScalingMethod.STANDARD'),
              ('aroon_25_aroon_down_1', 'oscillator', 'ScalingMethod.STANDARD'),
              ('william_r_14_wpr_1', 'oscillator', 'ScalingMethod.STANDARD'),
              ('vema_20_ema_1', 'volume', 'None'),
              ('william_r_60_wpr_1', 'oscillator', 'ScalingMethod.STANDARD'),
              ('uo_uo_1', 'oscillator', 'ScalingMethod.STANDARD'),
              ('adx_14_adx_1', 'misc', 'ScalingMethod.STANDARD'),
              ('rvi_10_7_rvi_1', 'oscillator', 'ScalingMethod.ROBUST'),
              ('roc_15_roc_1', 'oscillator', 'None'),
              ('roc_30_roc_1', 'oscillator', 'None'),
              ('tsi_true_25_13_tsi_1', 'oscillator', 'ScalingMethod.STANDARD'),
              ('william_r_7_wpr_1', 'oscillator', 'ScalingMethod.STANDARD'),
              ('timely_day_of_week', 'oscillator', 'ScalingMethod.STANDARD'),
              ('timely_hour_of_day', 'oscillator', 'ScalingMethod.STANDARD'),
              ('ema_12_ema_1', 'price', 'None'),
              ('ema_5_ema_1', 'price', 'None'),
              ('ema_26_ema_1', 'price', 'None'),
              ('roc_14_roc_1', 'oscillator', 'None'),
              ('donchian_20_upper_band_1', 'price', 'None'),
              ('donchian_20_middle_band_1', 'price', 'None'),
              ('donchian_20_lower_band_1', 'price', 'None'),
              ('atr_10_atr_1', 'misc', 'ScalingMethod.ROBUST'),
              ('roc_20_roc_1', 'oscillator', 'None'),
              ('fisher_9_fisher_1', 'ratio', 'ScalingMethod.STANDARD'),
              ('cmo_9_cmo_1', 'oscillator', 'ScalingMethod.STANDARD'),
              ('william_r_30_wpr_1', 'oscillator', 'ScalingMethod.STANDARD'),
              ('aobv_13_histogram_1', 'ratio', 'ScalingMethod.STANDARD'),
              ('vema_10_ema_1', 'volume', 'None'),
              ('vema_5_ema_1', 'volume', 'None'),
              ('donchian_5_upper_band_1', 'price', 'None'),
              ('donchian_5_middle_band_1', 'price', 'None'),
              ('donchian_5_lower_band_1', 'price', 'None'),
              ('ema_20_ema_1', 'price', 'None'),
              ('stoch_rsi_14_histogram_1',
               'oscillator',
               'ScalingMethod.STANDARD'),
              ('stddev_5_stddev_1', 'misc', 'ScalingMethod.STANDARD'),
              ('macd_9_histogram_1', 'misc', 'ScalingMethod.STANDARD'),
              ('keltner_5_2_upper_band_1', 'price', 'None'),
              ('keltner_5_2_lower_band_1', 'price', 'None'),
              ('avwap_20_deviation_1', 'ratio', 'ScalingMethod.STANDARD'),
              ('kst_histogram_1', 'ratio', 'ScalingMethod.STANDARD'),
              ('candlestrength_10_strength_1',
               'oscillator',
               'ScalingMethod.ROBUST'),
              ('candlestrength_10_tr_mult_1', 'ratio', 'ScalingMethod.ROBUST'),
              ('candlestrength_10_v_mult_1', 'ratio', 'ScalingMethod.ROBUST'),
              ('keltner_20_2_upper_band_1', 'price', 'None'),
              ('keltner_20_2_lower_band_1', 'price', 'None'),
              ('cci_20_cci_1', 'misc', 'ScalingMethod.STANDARD'),
              ('tsi_trend_20_tsi_1', 'ratio', 'ScalingMethod.STANDARD'),
              ('stddev_20_stddev_1', 'misc', 'ScalingMethod.STANDARD'),
              ('bband_5_4_upper_band_1', 'price', 'None'),
              ('bband_5_4_lower_band_1', 'price', 'None'),
              ('bband_5_2_upper_band_1', 'price', 'None'),
              ('bband_5_2_lower_band_1', 'price', 'None'),
              ('bband_20_4_upper_band_1', 'price', 'None'),
              ('bband_20_4_lower_band_1', 'price', 'None'),
              ('bband_20_2_upper_band_1', 'price', 'None'),
              ('bband_20_2_lower_band_1', 'price', 'None'),
              ('squeeze_20_squeeze_rating_1',
               'ratio',
               'ScalingMethod.STANDARD'),
              ('squeeze_20_momentum_1', 'misc', 'ScalingMethod.STANDARD')],
 'labels': [('label_1',)],
 'timestamps': [84960, 202101010000, 202103010000]}
"""
