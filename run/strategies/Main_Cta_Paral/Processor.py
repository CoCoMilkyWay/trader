import math
import os, sys
import numpy as np
import pandas as pd
from typing import List, Dict

from Chan.Chan import CChan
from Chan.ChanConfig import CChanConfig
from Chan.Common.CEnum import DATA_SRC, KL_TYPE, AUTYPE, DATA_FIELD, BSP_TYPE, FX_TYPE
from Chan.DataAPI.wtAPI import parse_time_column
from Chan.KLine.KLine_Unit import CKLine_Unit

from strategies.Main_Cta_Paral.Define import MetadataIn, MetadataOut, bt_config

# Those Processors are largely insulated, try minimize data throughput
UNDEFINED_ID = 999
class n_Processor:
    def __init__(self):
        self.chan_snapshot: Dict[str, CChan] = {}
        self.code_list: List[str] = []
        self.lv_list = [KL_TYPE.K_DAY]
        
        self.num_bsp_T1         = 0
        self.num_bsp_T2         = 0
        self.num_bsp_T3         = 0
        
        self.id:int = UNDEFINED_ID
        # tune Chan config for day bar
        
    def process_slave_task(self, id:int, task:MetadataIn):
        if self.id == UNDEFINED_ID:
            self.id = id
        code = task.code
        date = task.date
        curTime = task.curTime
        combined_klu = task.bar
        if code not in self.code_list:
            self.code_list.append(code)
            self.init_new_chan(code)
            
        trade = False
        while True:
            # feed & calculate
            chan_snapshot = self.chan_snapshot[code]
            chan_snapshot.trigger_load({self.lv_list[0]: [combined_klu]}) # feed day bar
            bsp_list = chan_snapshot.get_bsp()
            
            if not bsp_list:
                break
            last_bsp = bsp_list[-1]
            t = last_bsp.type
            T = [0,0,0]
            if BSP_TYPE.T1 in t or BSP_TYPE.T1P in t:
                self.num_bsp_T1 += 1; T[0] = 1
            if BSP_TYPE.T2 in t or BSP_TYPE.T2S in t:
                self.num_bsp_T2 += 1; T[1] = 1
            if BSP_TYPE.T3A in t or BSP_TYPE.T3B in t:
                self.num_bsp_T3 += 1; T[2] = 1
                
            cur_lv_kline = chan_snapshot[0] # __getitem__: return Kline list of level n
            metrics = cur_lv_kline.metric_model_lst
            #　if last_bsp.klu.klc.idx != cur_lv_kline[-1].idx:
            #　    break
            
            T_sum = 0
            for idx, t in enumerate(T):
                T_sum += t * (idx+1)
                
            Ctime = combined_klu.time
            top = False; bottom = False
            if cur_lv_kline[-2].fx == FX_TYPE.BOTTOM: # and last_bsp.is_buy:
                bottom = True
                bt_config.plot_para["marker"]["markers"][Ctime] = (f'b{T_sum}', 'down', 'red')
            elif cur_lv_kline[-2].fx == FX_TYPE.TOP: # and not last_bsp.is_buy:
                top = True
                bt_config.plot_para["marker"]["markers"][Ctime] = (f's{T_sum}', 'up', 'green')
            # note that for fine data period (e.g. 1m_bar), fx(thus bsp) of the same type would occur consecutively
            
            if T[0] != 1:
                break
            trade = True # trade signal generated
            break
            
        if trade:
            # print(f'cpu:{id}: {code}-{date}-{curTime}: buy:{bottom} sell:{top}')
            return  MetadataOut(
                cpu_id=id,
                idx=task.idx,
                code=task.code,
                date=task.date,
                curTime=task.curTime,
                buy=bottom,
                sell=top,
            )
            
    def init_new_chan(self, code):
        self.chan_snapshot[code] = CChan(
            code=code,
            # begin_time=begin_time,
            # end_time=end_time,
            # data_src=data_src,
            lv_list=self.lv_list,
            config=bt_config,
            # autype=AUTYPE.QFQ,
        )
    
    def on_backtest_end(self):
        try: # self.id == 0:
            code = self.code_list[0]
            print(f'Asset for Analyzing: cpu:{self.id:2} code:{code}')
            print('T1:', self.num_bsp_T1, ' T2:', self.num_bsp_T2, ' T3:', self.num_bsp_T3)
            bt_config.plot_config["plot_bsp"] = False
            bt_config.plot_config["plot_zs"] = False
            bt_config.plot_config["plot_marker"] = False
            bt_config.plot_config["plot_channel"] = False
            bt_config.plot_config["plot_mean"] = False
            bt_config.plot_config["plot_eigen"] = False
            bt_config.plot_config["plot_demark"] = False
            bt_config.plot_config["plot_seg"] = False
            bt_config.plot_para["seg"]["plot_trendline"] = False
            # print(bt_config.plot_para["marker"]["markers"])
            bt_config.plot_config["plot_chart_patterns"] = True
            print('Plotting ...')
            self.chan_snapshot[code].plot(save=True, animation=False, update_conf=True, conf=bt_config)
        except:
            pass