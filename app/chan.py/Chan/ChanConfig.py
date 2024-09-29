from typing import List

from Chan.Bi.BiConfig import CBiConfig
from Chan.BuySellPoint.BSPointConfig import CBSPointConfig
from Chan.Common.CEnum import TREND_TYPE
from Chan.Common.ChanException import CChanException, ErrCode
from Chan.Common.func_util import _parse_inf
from Chan.Math.BOLL import BollModel
from Chan.Math.Demark import CDemarkEngine
from Chan.Math.KDJ import KDJ
from Chan.Math.MACD import CMACD
from Chan.Math.RSI import RSI
from Chan.Math.TrendModel import CTrendModel
from Chan.Math.PA_Core import PA_Core
from Chan.Seg.SegConfig import CSegConfig
from Chan.ZS.ZSConfig import CZSConfig

class CChanConfig:
    def __init__(self, conf=None):
        if conf is None:
            conf = {}
        conf = ConfigWithCheck(conf)
        
        self.plot_config = {
            "plot_kline": False,
            "plot_kline_combine": True,
            "plot_bi": True,
            "plot_seg": False,
            "plot_eigen": False, # 笔的特征序列
            "plot_zs": True,
            "plot_macd": True,
            "plot_mean": False,
            "plot_channel": False,
            "plot_bsp": True,
            #　"plot_extrainfo": False,
            "plot_demark": False,
            "plot_marker": False,
            "plot_rsi": False,
            "plot_kdj": False,
            "plot_trend_lines": False,
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
        self.seg_conf = CSegConfig(
            seg_algo=conf.get("seg_algo", "chan"), # 线段计算方法
                                                    # chan: 利用特征序列来计算（默认）
                                                    # 1+1: 都业华版本 1+1 终结算法
                                                    # break: 线段破坏定义来计算线段
            left_method=conf.get("left_seg_method", "peak"), # 剩余那些不能归入确定线段的笔如何处理成段
                                                                # all: 收集至最后一个方向正确的笔，成为一段
                                                                # peak: 如果有个靠谱的新的极值，那么分成两段（默认）
        )
        self.zs_conf = CZSConfig(
            need_combine=conf.get("zs_combine", True), # 是否进行中枢合并，默认为 True
            zs_combine_mode=conf.get("zs_combine_mode", "zs"), # 中枢合并模式
                                                                    # zs: 两中枢区间有重叠才合并（默认）
                                                                    # peak: 两中枢有K线重叠就合并
            one_bi_zs=conf.get("one_bi_zs", False), # 是否需要计算只有一笔的中枢（分析趋势时会用到），默认为 False
            zs_algo=conf.get("zs_algo", "auto"), # 中枢算法，涉及到中枢是否允许跨段 (normal, over_seg, auto) (段内中枢/跨段中枢/自动)
                                                    # normal: 段内中枢:
                                                        # 上升线段起始笔为下上下，下降线段起始笔为上下上
                                                        # 中枢一定是奇数笔
                                                        # 中枢不跨段（即便后一段为虚段）
                                                    # over-seg: 跨段中枢:
                                                        # 当一个新的中枢产生时，起始笔会考虑线段方向，上升线段起始笔为下上下，下降线段起始笔为下上下
                                                        # 说明中枢也有所属线段的，属于第一笔所在的线段
                                                        # 中枢延伸时，当前笔和下一笔的高点范围均与中枢的高低范围有交集，那么当前笔就会纳入该中枢
                                                        # 中枢的笔数可能为奇数，也可能为偶数，取决于什么时候有一笔离开中枢
                                                        # 中枢可能跨段
                                                        # 相邻中枢如果分别属于两个段，不会进行合并（因为这种模式下可能会出现合并出一个巨大的中枢，使得后续所有笔都在这个中枢里面）
                                                        # 如果一类买卖点在跨段中枢的中间，背驰判断如上图所示。
                                                    # auto: 对于确定的线段，采用normal算法，不确定部分用over_seg
        )
        self.trigger_step = conf.get("trigger_step", False)
        # 用于逐步回放绘图时使用，此时 CChan 会变成一个生成器，每读取一根新K线就会计算一次当前所有指标，返回当前帧指标状况；常用于返回给 CAnimateDriver 绘图
        self.skip_step = conf.get("skip_step", 0) # trigger_step 为 True 时有效，指定跳过前面几根K线，默认为 0

        self.kl_data_check = conf.get("kl_data_check", True) # 是否需要检验K线数据，检查项包括时间线是否有乱序，大小级别K线是否有缺失；默认为 True
        self.max_kl_misalgin_cnt = conf.get("max_kl_misalgin_cnt", 2) # 在次级别找不到K线最大条数，默认为 2（次级别数据有缺失），kl_data_check 为 True 时生效
        self.max_kl_inconsistent_cnt = conf.get("max_kl_inconsistent_cnt", 5) # 天K线以下（包括）子级别和父级别日期不一致最大允许条数（往往是父级别数据有缺失），默认为 5，kl_data_check 为 True 时生效
        self.auto_skip_illegal_sub_lv = conf.get("auto_skip_illegal_sub_lv", False) # 如果获取次级别数据失败，自动删除该级别（比如指数数据一般不提供分钟线），默认为 False
        self.print_warning = conf.get("print_warning", True) # 打印K线不一致的明细，默认为 True
        self.print_err_time = conf.get("print_err_time", False) # 计算发生错误时打印因为什么时间的K线数据导致的，默认为 False

        self.mean_metrics: List[int] = conf.get("mean_metrics", [10]) # 均线计算周期（用于生成特征及绘图时使用），默认为空[]
        self.trend_metrics: List[int] = conf.get("trend_metrics", [10]) # 计算上下轨道线周期，即 T 天内最高/低价格（用于生成特征及绘图时使用），默认为空[]
        self.macd_config = conf.get("macd", {"fast": 12, "slow": 26, "signal": 9})
        self.cal_demark = conf.get("cal_demark", True) # demark指标
        self.cal_rsi = conf.get("cal_rsi", True) # rsi指标
        self.cal_kdj = conf.get("cal_kdj", True) # kdj指标
        self.rsi_cycle = conf.get("rsi_cycle", 14)
        self.kdj_cycle = conf.get("kdj_cycle", 9)
        self.demark_config = conf.get("demark", {
            'demark_len': 9, # setup完成时长度
            'setup_bias': 4, # setup比较偏移量
            'countdown_bias': 2, # countdown比较偏移量
            'max_countdown': 13, # 最大countdown数
            'tiaokong_st': True, # 序列真实起始位置计算时，如果setup第一根跳空，是否需要取前一根收盘价，默认为True
            'setup_cmp2close': True, # setup计算当前K线的收盘价对比的是setup_bias根K线前的close，如果不是，下跌setup对比的是low，上升对比的是close，默认为True
            'countdown_cmp2close': True, # countdown计算当前K线的收盘价对比的是countdown_bias根K线前的close，如果不是，下跌setup对比的是low，上升对比的是close，默认为True
        })
        self.boll_n = conf.get("boll_n", 20) # 布林线参数 N，整数，默认为 20（用于生成特征及绘图时使用）
        self.cal_charts = conf.get("cal_charts", True) # chart patterns
        self.charts_config = conf.get("chart_patterns", {
            'enable_patterns': [0] * 15,
        })
        self.set_bsp_config(conf)

        conf.check()

    def GetMetricModel(self): # this is updated at klu level
        res: List[CMACD | CTrendModel | BollModel | CDemarkEngine | RSI | KDJ | PA_Core] = [
            CMACD(
                fastperiod=self.macd_config['fast'],
                slowperiod=self.macd_config['slow'],
                signalperiod=self.macd_config['signal'],
            )
        ]
        res.extend(CTrendModel(TREND_TYPE.MEAN, mean_T) for mean_T in self.mean_metrics)

        for trend_T in self.trend_metrics:
            res.append(CTrendModel(TREND_TYPE.MAX, trend_T))
            res.append(CTrendModel(TREND_TYPE.MIN, trend_T))
        res.append(BollModel(self.boll_n))
        if self.cal_demark:
            res.append(CDemarkEngine(
                demark_len=self.demark_config['demark_len'],
                setup_bias=self.demark_config['setup_bias'],
                countdown_bias=self.demark_config['countdown_bias'],
                max_countdown=self.demark_config['max_countdown'],
                tiaokong_st=self.demark_config['tiaokong_st'],
                setup_cmp2close=self.demark_config['setup_cmp2close'],
                countdown_cmp2close=self.demark_config['countdown_cmp2close'],
            ))
        if self.cal_rsi:
            res.append(RSI(self.rsi_cycle))
        if self.cal_kdj:
            res.append(KDJ(self.kdj_cycle))
        return res

    def set_bsp_config(self, conf):
        para_dict = {
            "divergence_rate": float("inf"), #* 1类买卖点背驰比例，即离开中枢的笔的 MACD 指标相对于进入中枢的笔， 默认为 0.9
            "min_zs_cnt": 0, #* 1类买卖点至少要经历几个中枢
            "bsp1_only_multibi_zs": True, # min_zs_cnt 计算的中枢至少 3 笔（少于 3 笔是因为开启了 one_bi_zs 参数），默认为 True；
            "max_bs2_rate": 0.999, #* 2类买卖点那一笔回撤最大比例，默认为 0.618. 如果是 1.0，那么相当于允许回测到1类买卖点的位置
            "macd_algo": "peak", # peak/full_area/area/slope/amp/diff/amount/volume/amount_avg/volume_avg/turnrate_avg/rsi
            "bs1_peak": True, #* 1类买卖点位置是否必须是整个中枢最低点，默认为 True
            "bs_type": "1,1p,2,2s,3a,3b", #*
                # 1,2: 分别表示1，2，3类买卖点
                # 2s: 类二买卖点
                # 1p: 盘整背驰1类买卖点 (可能伴随跨段中枢)
                # 3a: 中枢出现在1类后面的3类买卖点(3-after)
                # 3b: 中枢出现在1类前面的3类买卖点(3-before)
            "bsp2_follow_1": True, #* 2类买卖点是否必须跟在1类买卖点后面（用于小转大时1类买卖点因为背驰度不足没生成），默认为 True
            "bsp3_follow_1": True, #* 3类买卖点是否必须跟在1类买卖点后面（用于小转大时1类买卖点因为背驰度不足没生成），默认为 True
            "bsp3_peak": False, # 3类买卖点突破笔是不是必须突破中枢里面最高/最低的，默认为 False
            "bsp2s_follow_2": False, # 类2买卖点是否必须跟在2类买卖点后面（2类买卖点可能由于不满足 max_bs2_rate 最大回测比例条件没生成），默认为 False
            "max_bsp2s_lv": None, # 类2买卖点最大层级（距离2类买卖点的笔的距离/2），默认为None，不做限制
            "strict_bsp3": False, # 3类买卖点对应的中枢必须紧挨着1类买卖点，默认为 False
        }
        args = {para: conf.get(para, default_value) for para, default_value in para_dict.items()}
        self.bs_point_conf = CBSPointConfig(**args)

        self.seg_bs_point_conf = CBSPointConfig(**args)
        self.seg_bs_point_conf.b_conf.set("macd_algo", "slope")
        self.seg_bs_point_conf.s_conf.set("macd_algo", "slope")
        self.seg_bs_point_conf.b_conf.set("bsp1_only_multibi_zs", False)
        self.seg_bs_point_conf.s_conf.set("bsp1_only_multibi_zs", False)

        for k, v in conf.items():
            if isinstance(v, str):
                v = f'"{v}"'
            v = _parse_inf(v)
            if k.endswith("-buy"):
                prop = k.replace("-buy", "")
                exec(f"self.bs_point_conf.b_conf.set('{prop}', {v})")
            elif k.endswith("-sell"):
                prop = k.replace("-sell", "")
                exec(f"self.bs_point_conf.s_conf.set('{prop}', {v})")
            elif k.endswith("-segbuy"):
                prop = k.replace("-segbuy", "")
                exec(f"self.seg_bs_point_conf.b_conf.set('{prop}', {v})")
            elif k.endswith("-segsell"):
                prop = k.replace("-segsell", "")
                exec(f"self.seg_bs_point_conf.s_conf.set('{prop}', {v})")
            elif k.endswith("-seg"):
                prop = k.replace("-seg", "")
                exec(f"self.seg_bs_point_conf.b_conf.set('{prop}', {v})")
                exec(f"self.seg_bs_point_conf.s_conf.set('{prop}', {v})")
            elif k in args:
                exec(f"self.bs_point_conf.b_conf.set({k}, {v})")
                exec(f"self.bs_point_conf.s_conf.set({k}, {v})")
            else:
                raise CChanException(f"unknown para = {k}", ErrCode.PARA_ERROR)
        self.bs_point_conf.b_conf.parse_target_type()
        self.bs_point_conf.s_conf.parse_target_type()
        self.seg_bs_point_conf.b_conf.parse_target_type()
        self.seg_bs_point_conf.s_conf.parse_target_type()

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
