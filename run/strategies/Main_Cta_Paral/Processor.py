import math
import os, sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from numba import jit, prange

from Chan.Chan import CChan
from Chan.Common.CTime import CTime
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
    # trading_session: 9:35~11:30, 13:05~14:55
    
    # PA: price action
    REBALANCE_TIME:List[int] = [ # bars accumulated per batch before send to processes
        935,
        # 940,
        # 950,
        1000,
        # 1010,
        # 1020,
        1030,
        # 1040,
        # 1050,
        1100,
        # 1110,
        # 1120,
        # 1125,
        1305,
        # 1310,
        # 1320,
        1330,
        # 1340,
        # 1350,
        1400,
        # 1410,
        # 1420,
        1430,
        # 1440,
        1450,
        # 1455,
        ]
    def __init__(self):
        self.chan_snapshot: Dict[str, CChan] = {}
        self.code_list: List[str] = []
        self.price_bin_width: Dict[str, float] = {}
        self.lv_list = [KL_TYPE.K_DAY]
        self.resample_buffer: Dict[str, List[WtNpKline]] = {}  # store temp bar to form larger bar
        
        self.num_bsp_T1 = 0
        self.num_bsp_T2 = 0
        self.num_bsp_T3 = 0
        self.id:int = UNDEFINED_ID
        # tune Chan config for day bar
        
    def process_slave_task(self, id:int, tasks:List[MetadataIn]):
        if self.id == UNDEFINED_ID:
            self.id = id
        
        orders:List[MetadataOut] = []
        for task in tasks:
            trade = False
            code = task.code
            date = task.date
            curTime = task.curTime
            kline_batch = task.bars
            
            if code not in self.code_list:
                self.code_list.append(code)
                self.price_bin_width[code] = self.config_price_bin_width(kline_batch[0].opens[-1])
                self.init_new_chan(code)
                self.resample_buffer[code] = []
            
            self.resample_buffer[code].extend(kline_batch)
            rebalance = False
            if curTime in self.REBALANCE_TIME:
                rebalance = True
                batch_combined_klu, batch_volume_profile = self.process_batch_klu(self.resample_buffer[code], self.price_bin_width[code])
                self.resample_buffer[code] = []
                # print(volume_profile)
            else:
                pass
            
            if rebalance:
                while True:
                    # feed & calculate
                    chan_snapshot = self.chan_snapshot[code]
                    chan_snapshot.trigger_load({self.lv_list[0]: [batch_combined_klu]}, batch_volume_profile) # feed day bar
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

                    Ctime = batch_combined_klu.time
                    top = False; bottom = False
                    if cur_lv_kline[-2].fx == FX_TYPE.BOTTOM: # and last_bsp.is_buy:
                        bottom = True
                        bt_config.plot_para["marker"]["markers"][Ctime] = (f'b{T_sum}', 'down', 'red')
                    elif cur_lv_kline[-2].fx == FX_TYPE.TOP: # and not last_bsp.is_buy:
                        top = True
                        bt_config.plot_para["marker"]["markers"][Ctime] = (f's{T_sum}', 'up', 'green')
                    # note that for fine data period (e.g. 1m_bar), fx(thus bsp) of the same type would occur consecutively

                    if T_sum == 0:
                        break
                    trade = True # trade signal generated
                    break
                
            if trade:
                print(f'cpu:{id}: {code}-{date}-{curTime}: buy:{bottom} sell:{top}')
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
    
    def on_backtest_end(self):
        try: # self.id == 0:
            if self.code_list == []:
                return
            code = self.code_list[0]
            if self.id == 0:
                print(f'Asset for Analyzing: cpu:{self.id:2} code:{code}')
                print('T1:', self.num_bsp_T1, ' T2:', self.num_bsp_T2, ' T3:', self.num_bsp_T3)
                print('Plotting ...')
            self.chan_snapshot[code].plot(save=True, animation=False, update_conf=True, conf=bt_config)
        except Exception as e:
            print(f"{type(e).__name__}")
            print(e)
    
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
        
    def config_price_bin_width(self, price:float) -> float:
        # fixed once configured
        # equity_unit_price, price_bin_width: (for volume profile)
        # <1,    0.01
        # <10,   0.01
        # <100,  0.1
        # >100,  0.1
        if price < 1:
            return 0.01
        elif price < 10:
            return 0.01
        elif price < 100:
            return 0.1
        elif price < 1000:
            return 1
        else:
            return 10
    
    # @jit(nopython=True, parallel=False) # acceleration(static compile before run)
    def process_batch_klu(self, resample_buffer: List[WtNpKline], price_bin_width) -> Tuple[CKLine_Unit, List[int|List[int]]]:
        # batch -> bi -> session -> history
        
        # only session volume profile is important:
        # how to define session? see PA_TreadLine for details
        
        # current session = earliest Zigzag(Chan.bi) of all active trendlines <-> current bar
        
        total_volume:int = 0
        high:float = max([klu.highs[-1] for klu in resample_buffer])
        low:float = min([klu.lows[-1] for klu in resample_buffer])
        first_open:float = resample_buffer[0].opens[-1]
        last_close:float = resample_buffer[-1].closes[-1]
        index_range_low:int = int(low//price_bin_width) # floor
        index_range_high:int = int(high//price_bin_width) # floor
        batch_volume_buyside:List[int] = [0] * (index_range_high-index_range_low+1)
        batch_volume_sellside:List[int] = [0] * (index_range_high-index_range_low+1)
        for klu in resample_buffer:
            open:float = klu.opens[-1]
            close:float = klu.closes[-1]
            volume:int = klu.volumes[-1]
            total_volume += volume
            index:int = int(close//price_bin_width)
            # error correction:
            if not (index_range_low < index < index_range_high):
                index = index_range_low
            if close > open:
                batch_volume_buyside[index-index_range_low] += volume
            else:
                batch_volume_sellside[index-index_range_low] += volume
        batch_time:CTime = parse_time_column(str(resample_buffer[-1].bartimes[-1]))
        batch_combined_klu = CKLine_Unit(
            {
                DATA_FIELD.FIELD_TIME:      batch_time,
                DATA_FIELD.FIELD_OPEN:      first_open,
                DATA_FIELD.FIELD_HIGH:      high,
                DATA_FIELD.FIELD_LOW:       low,
                DATA_FIELD.FIELD_CLOSE:     last_close,
                DATA_FIELD.FIELD_VOLUME:    total_volume,
            },
            autofix=True)
        batch_volume_profile = [
            batch_time,
            index_range_low,
            index_range_high,
            batch_volume_buyside,
            batch_volume_sellside,
            price_bin_width,
        ]
        return batch_combined_klu, batch_volume_profile
