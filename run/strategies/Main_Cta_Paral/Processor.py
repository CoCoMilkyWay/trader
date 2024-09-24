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

from wtpy.WtDataDefs import WtNpTicks, WtNpKline

from db.util import print_class_attributes_and_methods, mkdir
from strategies.Main_Cta_Paral.Define import MetadataIn, MetadataOut, column_name, bt_config

def pause():
    import time
    time.sleep(1000)
    return

UNDEFINED_ID = 999
# Those Processors are largely insulated, try minimize data throughput
class n_Processor:
    REBALANCE_TIME = 1425
    def __init__(self):
        self.chan_snapshot: Dict[str, CChan] = {}
        self.code_list: List[str] = []
        self.lv_list = [KL_TYPE.K_DAY]
        self.resample_buffer: Dict[str, List[WtNpKline]] = {}  # store temp bar to form larger bar
        
        self.num_bsp_T1         = 0
        self.num_bsp_T2         = 0
        self.num_bsp_T3         = 0
        self.id:int = UNDEFINED_ID
        # tune Chan config for day bar
        
    def process_slave_task(self, id:int, tasks:List[MetadataIn]):
        if self.id == UNDEFINED_ID:
            self.id = id
        
        trade = False
        orders:List[MetadataOut] = []
        for task in tasks:
            code = task.code
            date = task.date
            curTime = task.curTime
            kline_batch = task.bars
            
            if code not in self.code_list:
                self.code_list.append(code)
                self.init_new_chan(code)
                self.resample_buffer[code] = []
            
            self.resample_buffer[code].extend(kline_batch)
            rebalance = False
            if curTime == 1425: # new date
                rebalance = True
                combined_klu = self.combine_klu(self.resample_buffer[code])
                self.resample_buffer[code] = []
            else:
                pass
            
            if rebalance:
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
                orders.append(
                    MetadataOut(
                        cpu_id=id,
                        idx=task.idx,
                        code=task.code,
                        date=task.date,
                        curTime=task.curTime,
                        buy=bottom,
                        sell=top,
                        ))
        return orders
                
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
        
    def combine_klu(self, resample_buffer: List[WtNpKline]) -> CKLine_Unit:
        return CKLine_Unit(
            {
                DATA_FIELD.FIELD_TIME: parse_time_column(str(resample_buffer[-1].bartimes[-1])),
                DATA_FIELD.FIELD_OPEN: resample_buffer[0].opens[-1],
                DATA_FIELD.FIELD_HIGH: max(klu.highs[-1] for klu in resample_buffer),
                DATA_FIELD.FIELD_LOW: min(klu.lows[-1] for klu in resample_buffer),
                DATA_FIELD.FIELD_CLOSE: resample_buffer[-1].closes[-1],
                DATA_FIELD.FIELD_VOLUME: sum(klu.volumes[-1] for klu in resample_buffer),
            },
            autofix=True,
        )
        
    def on_backtest_end(self):
        # return
        try: # self.id == 0:
            if self.code_list == []:
                return
            code = self.code_list[0]
            print(f'Asset for Analyzing: cpu:{self.id:2} code:{code}')
            print('T1:', self.num_bsp_T1, ' T2:', self.num_bsp_T2, ' T3:', self.num_bsp_T3)
            print('Plotting ...')
            self.chan_snapshot[code].plot(save=True, animation=False, update_conf=True, conf=bt_config)
        except Exception as e:
            print(f"{type(e).__name__}")
            print(e)
