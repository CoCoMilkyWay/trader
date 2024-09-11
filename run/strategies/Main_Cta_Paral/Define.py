from typing import NamedTuple, Tuple, List
from Chan.KLine.KLine_Unit import CKLine_Unit

class MetadataIn(NamedTuple):
    idx: int
    code: str
    date: int
    curTime: int
    bar: CKLine_Unit
    rebalance: bool