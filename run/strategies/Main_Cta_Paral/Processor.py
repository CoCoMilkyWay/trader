import math
import multiprocessing.synchronize
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

import multiprocessing

def pause():
    import time
    time.sleep(1000)
    return

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
    def __init__(self, lock:multiprocessing.synchronize.Lock):
        if lock:
            self.lock = lock
        self.chan_snapshot: Dict[str, CChan] = {}
        self.code_list: List[str] = []
        self.price_bin_width: Dict[str, float] = {}
        self.lv_list = [KL_TYPE.K_DAY]
        self.resample_buffer: Dict[str, List[WtNpKline]] = {}  # store temp bar to form larger bar
        
        self.num_klu:     Dict[str, int] = {}
        self.num_bi:      Dict[str, int] = {}
        self.num_bsp_T1:  Dict[str, int] = {}
        self.num_bsp_T2:  Dict[str, int] = {}
        self.num_bsp_T3:  Dict[str, int] = {}
        # tune Chan config for day bar
        
    def process_slave_task(self, id:int, tasks:List[MetadataIn]):
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
                
                self.num_klu[code]    = 0
                self.num_bi[code]     = 0
                self.num_bsp_T1[code] = 0
                self.num_bsp_T2[code] = 0
                self.num_bsp_T3[code] = 0
            
            self.num_klu[code] += 1
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
                    # 0: feed & calculate Chan elements ========================
                    chan_snapshot = self.chan_snapshot[code]
                    chan_snapshot.trigger_load({self.lv_list[0]: [batch_combined_klu]}, batch_volume_profile) # feed day bar
                    cur_lv_kline_list = chan_snapshot[0] # __getitem__: return Kline list of level n
                    
                    # choose 1
                    CHECK_BSP = False
                    CHECK_FX = True
                    
                    if CHECK_BSP:
                        bsp_list = chan_snapshot.get_bsp()

                        # 1: initial condition
                        if not bsp_list:
                            break
                        
                        # 2: check if new bsp is formed (thus new bi is also formed)
                        last_bsp = bsp_list[-1]
                        kline_idx_last_bsp = last_bsp.klu.klc.idx
                        kline_idx_cur = cur_lv_kline_list[-2].idx
                        idx = kline_idx_cur
                        if kline_idx_last_bsp != kline_idx_cur: # this is kline(combined kline) index
                            # NOTE: bsp need 1 future bar to establish
                            # however, it is possible to extract FX info (BOT/TOP), note that it may not be sure
                            break
                        
                        # 3: check bsp type
                        last_bsp_type = last_bsp.type
                        T = [0,0,0]
                        if BSP_TYPE.T1 in last_bsp_type  or BSP_TYPE.T1P in last_bsp_type:
                            self.num_bsp_T1[code] += 1; T[0] = 1
                        if BSP_TYPE.T2 in last_bsp_type  or BSP_TYPE.T2S in last_bsp_type:
                            self.num_bsp_T2[code] += 1; T[1] = 1
                        if BSP_TYPE.T3A in last_bsp_type or BSP_TYPE.T3B in last_bsp_type:
                            self.num_bsp_T3[code] += 1; T[2] = 1
                        T_sum = 0
                        for idx, t in enumerate(T):
                            T_sum += t * (idx+1)
                        if T_sum == 0:
                            break # not expected bsp
                        
                        # 4: check FX and bsp type
                        top = False; bottom = False
                        if cur_lv_kline_list[-2].fx == FX_TYPE.BOTTOM and last_bsp.is_buy: # check if bi is down
                            bottom = True
                        elif cur_lv_kline_list[-2].fx == FX_TYPE.TOP and not last_bsp.is_buy: # check if bi is up
                            top = True
                        else:
                            print(
                                'Err: ',
                                last_bsp.klu.klc.idx, # bsp kline idx
                                last_bsp.is_buy, # bi is down
                                last_bsp_type, # bsp_type
                                cur_lv_kline_list[-1][-1].close, # price
                                cur_lv_kline_list[-2].fx, # fx_type
                                )
                            break
                        
                        # 5:　generate trading signals
                        Ctime = batch_combined_klu.time
                        if bottom:
                            bt_config.plot_para["marker"]["markers"][Ctime] = (f'b{T_sum}', 'down', 'red')
                            trade = True
                        elif top:
                            bt_config.plot_para["marker"]["markers"][Ctime] = (f's{T_sum}', 'up', 'green')
                            trade = True
                            
                    elif CHECK_FX:
                        bi_list = cur_lv_kline_list.bi_list
                        
                        # 1: initial condition
                        if len(bi_list)==0:
                            break
                        
                        # 2: check if new bi is formed
                        # combined kline = kline/klc
                        # original kline = kline/klu
                        # this is kline unit(not combined kline) index
                        last_bi = bi_list[-1]
                        kline_unit_idx_last_bi_begin = last_bi.get_begin_klu().idx
                        kline_unit_idx_cur = cur_lv_kline_list[-1][-1].idx
                        idx = kline_unit_idx_cur
                        if self.num_bi[code] == len(bi_list):
                            break
                        self.num_bi[code] = len(bi_list)
                        
                        # 3: check FX type
                        top = False; bottom = False
                        if bi_list[-2].is_down():
                            bottom = True
                        elif bi_list[-2].is_up():
                            top = True
                        else:
                            # print(
                            #     'Err: ',
                            #     cur_lv_kline_list[-1][-1].close, # price
                            #     cur_lv_kline_list[-2].fx, # fx_type
                            #     last_bi.is_sure, # if FX is sure
                            #     )
                            break
                        
                        # 4:　generate trading signals
                        # sure = '!' if bi_list[-2].is_sure else ''
                        Ctime = batch_combined_klu.time
                        if bottom:
                            bt_config.plot_para["marker"]["markers"][Ctime] = (f'v', 'down', 'red')
                            trade = True
                        elif top:
                            bt_config.plot_para["marker"]["markers"][Ctime] = (f'^', 'up', 'green')
                            trade = True
                        
                    # 1: prepare Machine Learning features from Chan.metrics
                    #    Labels, however, need future information (cannot be done here)
                    metrics = cur_lv_kline_list.metric_model_lst
                    
                    # 2: break
                    break
                
            if trade:
                DEBUG = False
                if DEBUG:
                    dir = '^' if top else 'v' if bottom else '?'
                    print(f'cpu:{id:2}: {code:>16}-{date}-{curTime:4}: idx:{idx:4}, FX:{dir} price:{cur_lv_kline_list[-1][-1].close}')
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
        def print_results():
            SAMPLES = self.num_klu[self.code_list[0]]
            T1 = self.num_bsp_T1[self.code_list[0]]; T1_ratio = T1 / SAMPLES *100
            T2 = self.num_bsp_T2[self.code_list[0]]; T2_ratio = T2 / SAMPLES *100
            T3 = self.num_bsp_T3[self.code_list[0]]; T3_ratio = T3 / SAMPLES *100
            print(f'Asset for Analyzing: cpu:{self.id:2} code:{code}')
            print('bsp: ',
                'T1:', f'{T1}({T1_ratio:.1f}%) '
                'T2:', f'{T2}({T2_ratio:.1f}%) '
                'T3:', f'{T3}({T3_ratio:.1f}%) '
                )
            print('Plotting .....................................')
            
        try: # self.id == 0:
            if self.code_list == []:
                return
            code = self.code_list[0]
            
            if self.lock:
                with self.lock:
                    print_results()
            else:
                if self.id == 0: print_results()
                
            if self.id == 0:
                self.chan_snapshot[code].plot(save=True, print=True, animation=False, update_conf=True, conf=bt_config)
            else:
                self.chan_snapshot[code].plot(save=True, print=False, animation=False, update_conf=True, conf=bt_config)
        
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
    def process_batch_klu(self, resample_buffer: List[WtNpKline], price_bin_width:float) -> Tuple[CKLine_Unit, List[int|List[int]]]:
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
            volume:int = int(klu.volumes[-1])
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
        new_bi = False
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
            new_bi, # wait for upper function to fill it
        ]
        return batch_combined_klu, batch_volume_profile
