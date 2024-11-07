from typing import Dict, Generic, List, Optional, TypeVar, Union

from Chan.Bi.Bi import CBi
from Chan.Common.CEnum import BSP_TYPE
from Chan.Seg.Seg import CSeg

LINE_TYPE = TypeVar('LINE_TYPE', CBi, CSeg)

class CBS_Point(Generic[LINE_TYPE]):
    def __init__(self, bi: LINE_TYPE, is_buy, bs_type: BSP_TYPE, relate_bsp1: Optional['CBS_Point'], feature_dict=None):
        self.bi: LINE_TYPE = bi
        self.klu = bi.get_end_klu()
        self.is_buy = is_buy
        self.type: List[BSP_TYPE] = [bs_type]
        self.relate_bsp1 = relate_bsp1

        self.bi.bsp = self  # type: ignore

        self.is_segbsp = False

    def add_type(self, bs_type: BSP_TYPE):
        self.type.append(bs_type)

    def type2str(self):
        return ",".join([x.value for x in self.type])

    def add_another_bsp_prop(self, bs_type: BSP_TYPE, relate_bsp1):
        self.add_type(bs_type)
        if self.relate_bsp1 is None:
            self.relate_bsp1 = relate_bsp1
        elif relate_bsp1 is not None:
            assert self.relate_bsp1.klu.idx == relate_bsp1.klu.idx
