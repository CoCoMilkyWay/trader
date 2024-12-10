from typing import List

from Chan.Bi.BiConfig import CBiConfig
from Chan.Common.ChanException import CChanException, ErrCode
from Chan.Common.func_util import _parse_inf
from Chan.Common.CEnum import KL_TYPE, DATA_FIELD

Muted_purple = 'rgba(186, 128, 212, 1)'
Sage_green = 'rgba(166, 212, 128, 1)'
Dusty_rose = 'rgba(212, 128, 166, 1)'
Steel_blue = 'rgba(128, 166, 212, 1)'
Classic_gray = 'rgba(128, 128, 128, 1)'


class CChanConfig:
    def __init__(self, conf=None):
        if conf is None:
            conf = {}
        conf = ConfigWithCheck(conf)

        self.lv_list = [  # from high to lowest(1min) levels
            # level,
            # down_sample_multiplier(to lower level),
            # num_klc_cached,
            # bi color,
            # opacity,
            # charts enable
            # liquidity level enable

            [KL_TYPE.K_60M, 2,  100,   Muted_purple,   1.0,   [             ], False,],
            [KL_TYPE.K_30M, 2,  100,   Sage_green,     1.0,   [             ], False,],
            [KL_TYPE.K_15M, 3,  100,   Dusty_rose,     1.0,   [             ], True, ],
            [KL_TYPE.K_5M,  5,  100000,   Steel_blue,     1.0,   ['conv_type', ], False,],
            [KL_TYPE.K_1M,  1,  100,   Classic_gray,   1.0,   [             ], False,],
        ]

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
            "klu": {
                "plot_mode": 'kl',  # kl/open/high/low/close
            },
            "klc": {
                "width": 0.4,
            },
            "bi": {
                "show_num": False,
                "disp_end": False,  # show vertex price
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
        }

        self.bi_conf = CBiConfig(
            bi_algo=conf.get("bi_algo", "normal"),
            # normal(按缠论笔定义来算)(分形之间相隔>= 3 K-lines & Unit Count >= 3)
            # fx(顶底分形即成笔)
            # * 是否只用严格笔(bi_algo=normal时有效)，默认为 Ture，(分形之间相隔>= 4 K-lines & Unit Count >= 3)
            is_strict=conf.get("bi_strict", True),
            # * 检查笔顶底分形是否成立的方法 (不建议放宽)
            bi_fx_check=conf.get("bi_fx_check", "strict"),
            # example:                                             # NO.bsp: loss > half > strict > totally
            #               fx      normal  strict                     # totally: 底分型3元素的最高点必须必顶分型三元素的最低点还低
            #   loss        2915    817     715                        # strict (默认) (突破够强,回撤不大,滤掉绞肉机行情) (底分型的最低点必须比顶分型3元素最低点的最小值还低，顶分型反之。
            #   half        1239    723     639                        # half: (突破够强,回撤可以大)对于上升笔，底分型的最低点比顶分型前两元素最低点还低，顶分型的最高点比底分型后两元素高点还高。下降笔反之。
            #   totally     726     608     558                        # loss: (做波段可以考虑打开)底分型的最低点比顶分型中间元素低点还低，顶分型反之。
            #   strict      771     555     511
            gap_as_kl=conf.get("gap_as_kl", True),  # 缺口是否处理成一根K线
            # 笔的尾部是否是整笔中最低/最高 (可以考虑模型学习时放宽)
            bi_end_is_peak=conf.get("bi_end_is_peak", True),
            bi_allow_sub_peak=conf.get("bi_allow_sub_peak", True), # This has to be True for PA to work
        )

        # 是否需要检验K线数据，检查项包括时间线是否有乱序，大小级别K线是否有缺失；默认为 True
        self.kl_data_check = conf.get("kl_data_check", True)
        self.print_warning = conf.get(
            "print_warning", True)  # 打印K线不一致的明细，默认为 True
        # 计算发生错误时打印因为什么时间的K线数据导致的，默认为 False
        self.print_err_time = conf.get("print_err_time", True)

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
            raise CChanException(f"invalid CChanConfig: {
                                 invalid_key_lst}", ErrCode.PARA_ERROR)
