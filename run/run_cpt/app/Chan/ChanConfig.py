from typing import List

from Chan.Bi.BiConfig import CBiConfig
from Chan.Common.ChanException import CChanException, ErrCode
from Chan.Common.func_util import _parse_inf

class CChanConfig:
    def __init__(self, conf=None):
        if conf is None:
            conf = {}
        conf = ConfigWithCheck(conf)
        
        self.plot_config = {
            "plot_kline": False,
            "plot_kline_combine": True,
            "plot_bi": True,
            # "plot_extrainfo": False,
            "plot_marker": False,
            "plot_trend_lines": False,
            "plot_liquidity": False,
            "plot_chart_patterns": False,
            "plot_volume_profile": False,
        }
        self.plot_para = {
            "seg": {
                
            },
            "bi": {
                "show_num": False,
                "disp_end": False, # show vertex price
            },
            "figure": {
                # "x_range":240,
            },
            "marker": {
                "markers": {  # text, position, color
                    # '2024/01/02': ('marker here1', 'up', 'red'),
                    # '2024/02/01': ('marker here2', 'down')
                },
            },
            "animation_pause_time": 0,
            "trend_lines": {
                "plot_trendline_num": 2,
                },
            "chart_patterns": {},
        }

        self.bi_conf = CBiConfig(
            bi_algo=conf.get("bi_algo", "fx"),  
                                                    # normal(按缠论笔定义来算), 
                                                    # fx(顶底分形即成笔)
            is_strict=conf.get("bi_strict", False), #* 是否只用严格笔(bi_algo=normal时有效)，默认为 Ture，其中这里的严格笔只考虑顶底分形之间相隔几个合并K线
            bi_fx_check=conf.get("bi_fx_check", "loss"),    #* 检查笔顶底分形是否成立的方法 (不建议放宽)
                                                            # NO.bsp: loss > half > strict > totally
                                                                # totally: 底分型3元素的最高点必须必顶分型三元素的最低点还低
                                                                # strict (默认) (突破够强,回撤不大,滤掉绞肉机行情) (底分型的最低点必须比顶分型3元素最低点的最小值还低，顶分型反之。
                                                                # half: (突破够强,回撤可以大)对于上升笔，底分型的最低点比顶分型前两元素最低点还低，顶分型的最高点比底分型后两元素高点还高。下降笔反之。
                                                                # loss: (做波段可以考虑打开)底分型的最低点比顶分型中间元素低点还低，顶分型反之。
            gap_as_kl=conf.get("gap_as_kl", True),  # 缺口是否处理成一根K线
            bi_end_is_peak=conf.get("bi_end_is_peak", False), # 笔的尾部是否是整笔中最低/最高 (可以考虑模型学习时放宽)
            bi_allow_sub_peak=conf.get("bi_allow_sub_peak", True),
        )

        self.kl_data_check = conf.get("kl_data_check", True) # 是否需要检验K线数据，检查项包括时间线是否有乱序，大小级别K线是否有缺失；默认为 True
        self.print_warning = conf.get("print_warning", True) # 打印K线不一致的明细，默认为 True
        self.print_err_time = conf.get("print_err_time", True) # 计算发生错误时打印因为什么时间的K线数据导致的，默认为 False
        
        conf.check()

class ConfigWithCheck:
    def __init__(self, conf):
        self.conf = conf

    def get(self, k, default_value=None):
        res = self.conf.get(k, default_value)
        if k in self.conf:
            del self.conf[k]
        return res

    def items(self):
        visit_keys = set()
        for k, v in self.conf.items():
            yield k, v
            visit_keys.add(k)
        for k in visit_keys:
            del self.conf[k]

    def check(self):
        if len(self.conf) > 0:
            invalid_key_lst = ",".join(list(self.conf.keys()))
            raise CChanException(f"invalid CChanConfig: {invalid_key_lst}", ErrCode.PARA_ERROR)
