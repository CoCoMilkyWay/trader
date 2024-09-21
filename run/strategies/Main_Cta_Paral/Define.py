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
    "bi_algo"           : "fx",
    "bi_strict"         : False,
    "bi_fx_check"       : "loss", 
})
bt_config.plot_config["plot_bsp"] = False
bt_config.plot_config["plot_marker"] = False
bt_config.plot_config["plot_zs"] = False
bt_config.plot_config["plot_channel"] = False
bt_config.plot_config["plot_mean"] = False
bt_config.plot_config["plot_eigen"] = False
bt_config.plot_config["plot_demark"] = False
bt_config.plot_config["plot_seg"] = False
bt_config.plot_para["seg"]["plot_trendline"] = False
bt_config.plot_config["plot_chart_patterns"] = True
