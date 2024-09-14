from typing import NamedTuple, Tuple, List

from Chan.ChanConfig import CChanConfig
from Chan.KLine.KLine_Unit import CKLine_Unit

class MetadataIn(NamedTuple):
    idx: int
    code: str
    date: int
    curTime: int
    bar: CKLine_Unit
    rebalance: bool

class MetadataOut(NamedTuple):
    cpu_id: int
    idx: int
    code: str
    date: int
    curTime: int
    buy: bool
    sell: bool
    
bt_config = CChanConfig({
    "trigger_step"      : True,
    "skip_step"         : 0,
    # Bi
    "bi_algo"           : "normal",
    "bi_strict"         : True,
    "bi_fx_check"       : "strict", # 突破够强，回撤不大(滤掉绞肉机行情)
})