import copy
import json
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Union
from pprint import pprint

from Chan.BuySellPoint.BS_Point import CBS_Point
from Chan.ChanConfig import CChanConfig
from Chan.ChanModel.Features import CFeatures
from Chan.Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Chan.Common.ChanException import CChanException, ErrCode
from Chan.Common.CTime import CTime
from Chan.Common.func_util import check_kltype_order, kltype_lte_day
from Chan.DataAPI.CommonStockAPI import CCommonStockApi
from Chan.KLine.KLine_List import CKLine_List
from Chan.KLine.KLine_Unit import CKLine_Unit

import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt
model_path = 'xgboost_model.json' # json or ubj(binary)

class CChan:
    def __init__(
        self,
        code,
        begin_time=None,
        end_time=None,
        data_src: Union[DATA_SRC, str] = DATA_SRC.BAO_STOCK,
        lv_list=None,
        config=None,
        autype: AUTYPE = AUTYPE.QFQ,
    ):
        if lv_list is None:
            lv_list = [KL_TYPE.K_DAY, KL_TYPE.K_60M]
        check_kltype_order(lv_list)  # lv_list顺序从高到低
        self.code = code
        self.begin_time = str(begin_time) if isinstance(begin_time, datetime.date) else begin_time
        self.end_time = str(end_time) if isinstance(end_time, datetime.date) else end_time
        self.autype = autype
        self.data_src = data_src
        self.lv_list: List[KL_TYPE] = lv_list
        self.new_bi_start: bool = False # highest level kline bi
        # self.new_seg_start: bool = False # highest level kline seg

        # temp_batch
        self.volume_profile_batch:List[int|List[int]]

        if config is None:
            config = CChanConfig()
        self.conf = config

        self.kl_misalign_cnt = 0
        self.kl_inconsistent_detail = defaultdict(list)

        self.g_kl_iter = defaultdict(list)

        # Machine Learning Features
        self.ML_en = self.conf.ML_en
        self.LEARN = self.conf.LEARN
        self.PREDICT = self.conf.PREDICT
        self.LABEL_METHOD = self.conf.LABEL_METHOD # 'naive_next_bi'
        
        self.features = CFeatures(initFeat=None)
        if self.ML_en and self.PREDICT:
            try:
                self.model = xgb.Booster()
                self.model.load_model(model_path)
                
                with open(model_path, 'r') as f:
                    model_json = json.load(f)
                    pprint(model_json, width=300, depth=5, compact=True)
                
                print('Plotting Model as tree...')
                plot_tree(self.model)
                plt.savefig('model_tree', bbox_inches='tight')
                print('Model saved as ./model_tree.png ...')
            except:
                print('Error getting trained model ...')

        # trade signals:
        self.is_buy:bool = False
        self.is_sell:bool = False
        
        self.do_init()

        if not config.trigger_step:
            for _ in self.load():
                ...

    def __deepcopy__(self, memo):
        cls = self.__class__
        obj: CChan = cls.__new__(cls)
        memo[id(self)] = obj
        obj.code = self.code
        obj.begin_time = self.begin_time
        obj.end_time = self.end_time
        obj.autype = self.autype
        obj.data_src = self.data_src
        obj.lv_list = copy.deepcopy(self.lv_list, memo)
        obj.conf = copy.deepcopy(self.conf, memo)
        obj.kl_misalign_cnt = self.kl_misalign_cnt
        obj.kl_inconsistent_detail = copy.deepcopy(self.kl_inconsistent_detail, memo)
        obj.g_kl_iter = copy.deepcopy(self.g_kl_iter, memo)
        if hasattr(self, 'klu_cache'):
            obj.klu_cache = copy.deepcopy(self.klu_cache, memo)
        if hasattr(self, 'klu_last_t'):
            obj.klu_last_t = copy.deepcopy(self.klu_last_t, memo)
        obj.kl_datas = {}
        for kl_type, ckline in self.kl_datas.items():
            obj.kl_datas[kl_type] = copy.deepcopy(ckline, memo)
        for kl_type, ckline in self.kl_datas.items():
            for klc in ckline:
                for klu in klc.lst:
                    assert id(klu) in memo
                    if klu.sup_kl:
                        memo[id(klu)].sup_kl = memo[id(klu.sup_kl)]
                    memo[id(klu)].sub_kl_list = [memo[id(sub_kl)] for sub_kl in klu.sub_kl_list]
        return obj

    def do_init(self):
        self.kl_datas: Dict[KL_TYPE, CKLine_List] = {}
        for idx in range(len(self.lv_list)):
            self.kl_datas[self.lv_list[idx]] = CKLine_List(self.lv_list[idx], conf=self.conf)

    def load_stock_data(self, stockapi_instance: CCommonStockApi, lv) -> Iterable[CKLine_Unit]:
        for KLU_IDX, klu in enumerate(stockapi_instance.get_kl_data()):
            klu.set_idx(KLU_IDX)
            klu.kl_type = lv
            yield klu

    def get_load_stock_iter(self, stockapi_cls, lv):
        stockapi_instance = stockapi_cls(code=self.code, k_type=lv, begin_date=self.begin_time, end_date=self.end_time, autype=self.autype)
        return self.load_stock_data(stockapi_instance, lv)

    def add_lv_iter(self, lv_idx, iter):
        if isinstance(lv_idx, int):
            self.g_kl_iter[self.lv_list[lv_idx]].append(iter)
        else:
            self.g_kl_iter[lv_idx].append(iter)

    def get_next_lv_klu(self, lv_idx):
        if isinstance(lv_idx, int):
            lv_idx = self.lv_list[lv_idx]
        if len(self.g_kl_iter[lv_idx]) == 0:
            raise StopIteration
        try:
            return self.g_kl_iter[lv_idx][0].__next__()
        except StopIteration:
            self.g_kl_iter[lv_idx] = self.g_kl_iter[lv_idx][1:]
            if len(self.g_kl_iter[lv_idx]) != 0:
                return self.get_next_lv_klu(lv_idx)
            else:
                raise

    def step_load(self):
        assert self.conf.trigger_step
        self.do_init()  # 清空数据，防止再次重跑没有数据
        yielded = False  # 是否曾经返回过结果
        for idx, snapshot in enumerate(self.load(self.conf.trigger_step)):
            if idx < self.conf.skip_step:
                continue
            yield snapshot
            yielded = True
        if not yielded:
            yield self

    def trigger_load(self, klu_dict, batch_volume_profile:List):
        # PA update flow:
        #   1. update new bi
        #   2. update volume profile
        
        # 在已有pickle基础上继续计算新的
        # {type: [klu, ...]}
        if not hasattr(self, 'klu_cache'):
            self.klu_cache: List[Optional[CKLine_Unit]] = [None for _ in self.lv_list]
        if not hasattr(self, 'klu_last_t'):
            self.klu_last_t = [CTime(1980, 1, 1, 0, 0) for _ in self.lv_list]
        for lv_idx, lv in enumerate(self.lv_list):
            if lv not in klu_dict:
                if lv_idx == 0:
                    raise CChanException(f"最高级别{lv}没有传入数据", ErrCode.NO_DATA)
                continue
            assert isinstance(klu_dict[lv], list)
            self.add_lv_iter(lv, iter(klu_dict[lv]))
        for _ in self.load_iterator(lv_idx=0, parent_klu=None, step=False):
            ...
        if not self.conf.trigger_step:  # 非回放模式全部算完之后才算一次中枢和线段
            for lv in self.lv_list:
                self.kl_datas[lv].cal_seg_and_zs()
                
        # update volume_profile for highest level Kline_List
        batch_volume_profile[6] = self.new_bi_start
        # batch_volume_profile[7] = self.new_seg_start
        self.kl_datas[self.lv_list[0]].PA_Core.add_volume_profile(batch_volume_profile, 'batch')
        
        # ML algorithm is called after a new bi is established, so update feature(label) accordingly
        if self.new_bi_start: # bi is sure
            # self.is_sell = False
            self.is_sell = self.is_buy
            self.is_buy = False
            if self.ML_en:
                self.learn_or_predict()

    def get_features(self):
        self.features.refresh_feature_page()
        f = self.features._features
        PA = self.kl_datas[self.lv_list[0]].PA_Core
        PA_S = PA.PA_Shapes
        PA_L = PA.PA_Liquidity
        PA_V = PA.PA_Volume_Profile
        
        new_feature = len(PA.PA_Shapes_active['nexus_type']) > 0
        if new_feature:
            f['PA_CP_exist_nexus'       ]   = float(len(PA.PA_Shapes_active['nexus_type']) > 0)
            f['PA_CP_exist_nexus_mult'  ]   = float(len(PA.PA_Shapes_active['nexus_type']))
            shape = PA.PA_Shapes_active['nexus_type'][0] # oldest and strongest shape
            f['PA_CP_first_entry'       ]   = float(shape.name == 'entry'      ) # just finished a v or ^ with a strong 1st bi
            f['PA_CP_is_channel'        ]   = float('channel'     in shape.name)
            f['PA_CP_is_rect'           ]   = float('rect'        in shape.name)
            f['PA_CP_is_meg_sym'        ]   = float('meg_sym'     in shape.name)
            f['PA_CP_is_meg_brk_far'    ]   = float('meg_brk_far' in shape.name)
            f['PA_CP_is_meg_rev_bak'    ]   = float('meg_rev_bak' in shape.name)
            f['PA_CP_is_tri_sym'        ]   = float('tri_sym'     in shape.name)
            f['PA_CP_is_tri_brk_far'    ]   = float('tri_brk_far' in shape.name)
            f['PA_CP_is_tri_rev_bak'    ]   = float('tri_rev_bak' in shape.name)
            f['PA_CP_is_flag'           ]   = float('flag'        in shape.name)
            
            f['PA_CP_entry_dir'         ]   = float(shape.entry_dir) # which direction the nexus is entered
            f['PA_CP_abs_d1'            ]   = float(shape.abs_d1) # the abs strength of first bi entering nexus
            f['PA_CP_num_vertices'      ]   = float(shape.rising_cnt)
            f['PA_CP_far_cons'          ]   = float(shape.far_cons) # far side of entering bi is consolidating
            f['PA_CP_near_cons'         ]   = float(shape.near_cons) # near side of entering bi is consolidating
            f['PA_CP_top_slope'         ]   = float(shape.top_m)
            f['PA_CP_bot_slope'         ]   = float(shape.bot_m)
            f['PA_CP_top_residue'       ]   = float(shape.top_residue)
            f['PA_CP_bot_residue'       ]   = float(shape.bot_residue)
        return new_feature

    def learn_or_predict(self):
        
        new_feature = self.get_features()
        # instance a label class for this new feature
        
        if self.LEARN: # labels calculation even when new features are not available
            pending_labels = self.features.get_pending_label_updates()
            if pending_labels > 0:
                if self.LABEL_METHOD == 'naive_next_bi':
                    PA = self.kl_datas[self.lv_list[0]].PA_Core
                    label = PA.bi_lst[-1].value - self.kl_datas[self.lv_list[0]][-1][-1].close # PA.bi_lst[-2].value
                    print(label)
                self.features.update_label_list(label) # update label (contain future information)
            
            if new_feature:
                self.features.update_features_array() # update features
                
            # pprint(shape.name)
            # pprint(f)
            # pprint(self.features.feature_history)
            # pprint('================================================================')
        elif self.PREDICT: # predict
            if new_feature: # only predict when new feature comes in
                new_X = xgb.DMatrix(pd.DataFrame([self.features._features]))
                new_Y = self.model.predict(new_X)[0]
                self.is_buy = new_Y > 0.1
                print(new_Y)

    def train(self):
        if self.PREDICT:
            return
        
        # TODO
        pending_labels = self.features.get_pending_label_updates()
        self.features.label_history.extend([0.0] * pending_labels)
        
        features = pd.DataFrame(self.features.feature_history)
        label = pd.Series(self.features.label_history)
        
        # Create XGBoost DMatrix
        dtrain = xgb.DMatrix(features, label=label)
        
        # Train XGBoost model
        print('Training ......................')
        # model = xgb.XGBClassifier()
        # model = xgb.XGBRegressor()
        model = xgb.train({}, dtrain)
        print('Training Ended ................')
        
        model.save_model(model_path)
        print(f'Model Persisted to {model_path} ................')

    def init_lv_klu_iter(self, stockapi_cls):
        # 为了跳过一些获取数据失败的级别
        lv_klu_iter = []
        valid_lv_list = []
        for lv in self.lv_list:
            try:
                lv_klu_iter.append(self.get_load_stock_iter(stockapi_cls, lv))
                valid_lv_list.append(lv)
            except CChanException as e:
                if e.errcode == ErrCode.SRC_DATA_NOT_FOUND and self.conf.auto_skip_illegal_sub_lv:
                    if self.conf.print_warning:
                        print(f"[WARNING-{self.code}]{lv}级别获取数据失败，跳过")
                    del self.kl_datas[lv]
                    continue
                raise e
        self.lv_list = valid_lv_list
        return lv_klu_iter

    def GetStockAPI(self):
        _dict = {}
        if self.data_src == DATA_SRC.BAO_STOCK:
            from Chan.DataAPI.BaoStockAPI import CBaoStock
            _dict[DATA_SRC.BAO_STOCK] = CBaoStock
        elif self.data_src == DATA_SRC.CCXT:
            from Chan.DataAPI.ccxt import CCXT
            _dict[DATA_SRC.CCXT] = CCXT
        elif self.data_src == DATA_SRC.CSV:
            from Chan.DataAPI.csvAPI import CSV_API
            _dict[DATA_SRC.CSV] = CSV_API
        elif self.data_src == DATA_SRC.WT:
            from Chan.DataAPI.wtAPI import WT_API
            _dict[DATA_SRC.WT] = WT_API
        if self.data_src in _dict:
            return _dict[self.data_src]
        assert isinstance(self.data_src, str)
        if self.data_src.find("custom:") < 0:
            raise CChanException("load src type error", ErrCode.SRC_DATA_TYPE_ERR)
        package_info = self.data_src.split(":")[1]
        package_name, cls_name = package_info.split(".")
        exec(f"from Chan.DataAPI.{package_name} import {cls_name}")
        return eval(cls_name)

    def load(self, step=False):
        stockapi_cls = self.GetStockAPI()
        try:
            stockapi_cls.do_init()
            for lv_idx, klu_iter in enumerate(self.init_lv_klu_iter(stockapi_cls)):
                self.add_lv_iter(lv_idx, klu_iter)
            self.klu_cache: List[Optional[CKLine_Unit]] = [None for _ in self.lv_list]
            self.klu_last_t = [CTime(1980, 1, 1, 0, 0) for _ in self.lv_list]

            yield from self.load_iterator(lv_idx=0, parent_klu=None, step=step)  # 计算入口
            if not step:  # 非回放模式全部算完之后才算一次中枢和线段
                for lv in self.lv_list:
                    self.kl_datas[lv].cal_seg_and_zs()
        except Exception:
            raise
        finally:
            stockapi_cls.do_close()
        if len(self[0]) == 0:
            raise CChanException("最高级别没有获得任何数据", ErrCode.NO_DATA)

    def set_klu_parent_relation(self, parent_klu, kline_unit, cur_lv, lv_idx):
        if self.conf.kl_data_check and kltype_lte_day(cur_lv) and kltype_lte_day(self.lv_list[lv_idx-1]):
            self.check_kl_consitent(parent_klu, kline_unit)
        parent_klu.add_children(kline_unit)
        kline_unit.set_parent(parent_klu)

    def add_new_kl(self, cur_lv: KL_TYPE, kline_unit):
        try:
            self.kl_datas[cur_lv].add_single_klu(kline_unit)
            
            # check if is new bi
            if cur_lv == self.lv_list[0]: # highest level kline
                self.new_bi_start = self.kl_datas[cur_lv].new_bi_start
                # self.new_seg_start = self.kl_datas[cur_lv].new_seg_start
        except Exception:
            if self.conf.print_err_time:
                print(f"[ERROR-{self.code}]在计算{kline_unit.time}K线时发生错误!")
            raise

    def try_set_klu_idx(self, lv_idx: int, kline_unit: CKLine_Unit):
        if kline_unit.idx >= 0:
            return
        if len(self[lv_idx]) == 0:
            kline_unit.set_idx(0)
        else:
            kline_unit.set_idx(self[lv_idx][-1][-1].idx + 1)

    def load_iterator(self, lv_idx, parent_klu, step):
        # K线时间天级别以下描述的是结束时间，如60M线，每天第一根是10点30的
        # 天以上是当天日期
        cur_lv = self.lv_list[lv_idx]
        pre_klu = None
        while True:
            if self.klu_cache[lv_idx]:
                kline_unit = self.klu_cache[lv_idx]
                assert kline_unit is not None
                self.klu_cache[lv_idx] = None
            else:
                try:
                    kline_unit = self.get_next_lv_klu(lv_idx)
                    self.try_set_klu_idx(lv_idx, kline_unit)
                    if not kline_unit.time > self.klu_last_t[lv_idx]:
                        raise CChanException(f"kline time err, cur={kline_unit.time}, last={self.klu_last_t[lv_idx]}", ErrCode.KL_NOT_MONOTONOUS)
                    self.klu_last_t[lv_idx] = kline_unit.time
                except StopIteration:
                    break

            if parent_klu and kline_unit.time > parent_klu.time:
                self.klu_cache[lv_idx] = kline_unit
                break
            kline_unit.set_pre_klu(pre_klu)
            pre_klu = kline_unit
            self.add_new_kl(cur_lv, kline_unit)
            if parent_klu:
                self.set_klu_parent_relation(parent_klu, kline_unit, cur_lv, lv_idx)
            if lv_idx != len(self.lv_list)-1:
                for _ in self.load_iterator(lv_idx+1, kline_unit, step):
                    ...
                self.check_kl_align(kline_unit, lv_idx)
            if lv_idx == 0 and step:
                yield self

    def check_kl_consitent(self, parent_klu, sub_klu):
        if parent_klu.time.year != sub_klu.time.year or \
           parent_klu.time.month != sub_klu.time.month or \
           parent_klu.time.day != sub_klu.time.day:
            self.kl_inconsistent_detail[str(parent_klu.time)].append(sub_klu.time)
            if self.conf.print_warning:
                print(f"[WARNING-{self.code}]父级别时间是{parent_klu.time}，次级别时间却是{sub_klu.time}")
            if len(self.kl_inconsistent_detail) >= self.conf.max_kl_inconsistent_cnt:
                raise CChanException(f"父&子级别K线时间不一致条数超过{self.conf.max_kl_inconsistent_cnt}！！", ErrCode.KL_TIME_INCONSISTENT)

    def check_kl_align(self, kline_unit, lv_idx):
        if self.conf.kl_data_check and len(kline_unit.sub_kl_list) == 0:
            self.kl_misalign_cnt += 1
            if self.conf.print_warning:
                print(f"[WARNING-{self.code}]当前{kline_unit.time}没在次级别{self.lv_list[lv_idx+1]}找到K线！！")
            if self.kl_misalign_cnt >= self.conf.max_kl_misalgin_cnt:
                raise CChanException(f"在次级别找不到K线条数超过{self.conf.max_kl_misalgin_cnt}！！", ErrCode.KL_DATA_NOT_ALIGN)

    def __getitem__(self, n) -> CKLine_List:
        if isinstance(n, KL_TYPE):
            return self.kl_datas[n]
        elif isinstance(n, int):
            return self.kl_datas[self.lv_list[n]]
        else:
            raise CChanException("unspoourt query type", ErrCode.COMMON_ERROR)

    def get_bsp(self, idx=None) -> List[CBS_Point]:
        if idx is not None:
            return sorted(self[idx].bs_point_lst.lst, key=lambda x: x.klu.time)
        assert len(self.lv_list) == 1
        return sorted(self[0].bs_point_lst.lst, key=lambda x: x.klu.time)

    def plot(self, save: bool = True, print: bool = False, animation: bool = False, update_conf: bool = False, conf: CChanConfig = CChanConfig()):
        from Chan.Plot.AnimatePlotDriver import CAnimateDriver
        from Chan.Plot.PlotDriver import CPlotDriver
        if update_conf:
            config = conf
        else:
            config = self.conf
        
        chan:CChan = self
        if not animation:
            plot_driver = CPlotDriver(
                chan, # type: ignore
                plot_config=config.plot_config,
                plot_para=config.plot_para,
                print=print,
            )
            if save:
                plot_driver.save2img(mkdir(f'./outputs_bt/{self.code}.png'))
            else:
                plot_driver.figure.show()
        else:
            CAnimateDriver(
                chan, # type: ignore
                plot_config=config.plot_config,
                plot_para=config.plot_para,
            )
import os

def mkdir(path_str):
    path = os.path.dirname(path_str)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path_str