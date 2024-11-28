from typing import NamedTuple, Tuple, List

from wtpy.WtDataDefs import WtNpTicks, WtNpKline

from Chan.ChanConfig import CChanConfig
from Chan.KLine.KLine_Unit import CKLine_Unit
from Chan.Common.CEnum import DATA_SRC, KL_TYPE, AUTYPE, DATA_FIELD, BSP_TYPE, FX_TYPE

class MetadataIn(NamedTuple):
    idx: int
    code: str
    date: int
    curTime: int
    bars: List[WtNpKline]
    
class MetadataOut(NamedTuple):
    cpu_id: int
    idx: int
    code: str
    date: int
    curTime: int
    buy: bool
    sell: bool

column_name = [
    DATA_FIELD.FIELD_TIME,
    DATA_FIELD.FIELD_OPEN,
    DATA_FIELD.FIELD_HIGH,
    DATA_FIELD.FIELD_LOW,
    DATA_FIELD.FIELD_CLOSE,
    DATA_FIELD.FIELD_VOLUME,
    # DATA_FIELD.FIELD_TURNOVER,
    # DATA_FIELD.FIELD_TURNRATE,
    ]  # 每一列字段

bt_config = CChanConfig({
    "trigger_step"      : True,
    "skip_step"         : 0,
    # Bi
    "bi_algo"           : "fx",
    "bi_strict"         : False,
    "bi_fx_check"       : "loss",   # when use with chart patterns, use "loss"
                                    # if not use "loss", can not capture convergence pattern of volatility
})

# ML config
bt_config.LEARN = True
bt_config.LABEL_METHOD = 'naive_next_bi'

bt_config.plot_para["trend_lines"]["plot_trendline_num"] = 1
bt_config.plot_config["plot_kline"] = False
bt_config.plot_config["plot_bsp"] = False
bt_config.plot_config["plot_marker"] = True
bt_config.plot_config["plot_zs"] = False
bt_config.plot_config["plot_channel"] = False
bt_config.plot_config["plot_mean"] = False
bt_config.plot_config["plot_eigen"] = False
bt_config.plot_config["plot_demark"] = False
bt_config.plot_config["plot_seg"] = True
bt_config.plot_config["plot_trend_lines"] = True
bt_config.plot_config["plot_liquidity"] = True
bt_config.plot_config["plot_chart_patterns"] = True
bt_config.plot_config["plot_volume_profile"] = True