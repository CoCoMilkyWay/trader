from enum import Enum, auto
from typing import Literal


class KL_TYPE(Enum):
    K_1M      = 1
    K_DAY     = 2
    K_WEEK    = 3
    K_MON     = 4
    K_YEAR    = 5
    K_5M      = 6
    K_15M     = 7
    K_30M     = 8
    K_60M     = 9
    K_3M      = 10
    K_QUARTER = 11


class KLINE_DIR(Enum):
    UP = auto()
    DOWN = auto()
    COMBINE = auto()
    INCLUDED = auto()


class FX_TYPE(Enum):
    BOTTOM = auto()
    TOP = auto()
    UNKNOWN = auto()


class BI_DIR(Enum):
    UP = auto()
    DOWN = auto()


class BI_TYPE(Enum):
    UNKNOWN = auto()
    STRICT = auto()
    SUB_VALUE = auto()  # 次高低点成笔
    TIAOKONG_THRED = auto()
    DAHENG = auto()
    TUIBI = auto()
    UNSTRICT = auto()
    TIAOKONG_VALUE = auto()


class FX_CHECK_METHOD(Enum):
    STRICT = auto()
    LOSS = auto()
    HALF = auto()
    TOTALLY = auto()


class DATA_FIELD:
    FIELD_TIME = "time_key"
    FIELD_OPEN = "open"
    FIELD_HIGH = "high"
    FIELD_LOW = "low"
    FIELD_CLOSE = "close"
    FIELD_VOLUME = "volume"  # 成交量
    FIELD_TURNOVER = "turnover"  # 成交额
    FIELD_TURNRATE = "turnover_rate"  # 换手率


TRADE_INFO_LST = [DATA_FIELD.FIELD_VOLUME,
                  DATA_FIELD.FIELD_TURNOVER, DATA_FIELD.FIELD_TURNRATE]
