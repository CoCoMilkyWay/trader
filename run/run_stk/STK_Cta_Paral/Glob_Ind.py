from typing import List, Dict
from Chan.KLine.KLine_Unit import CKLine_Unit

class Glob_Ind:
    def __init__(self, klu_dict:Dict[str, CKLine_Unit]) -> None:
        pass
    
    def calc(self):
        # 行业资金流向
        # 全市场趋势强度排行: rank((MACD.DIFF - EMA慢线)/EMA慢线, 全市场)
        # 近期放量：最近40天成交量全年占比，全市场排列（避开微盘股）
        return
