import os
import sys
import math
import numpy as np
from numpy.polynomial import Polynomial
from typing import List, Dict, Union
import copy

from Chan.Bi.Bi import CBi
from Chan.Bi.BiList import CBiList
from app.PA.PA_types import vertex
from app.PA.PA_Pattern_Chart import conv_type
from app.PA.PA_Liquidity import PA_Liquidity
from app.PA.PA_Volume_Profile import PA_Volume_Profile

# things that PA(Price Action) is lacking:
#   HFT market making,
#   statistical arbitrage,
#   gamma scalping,
#   volatility arbitrage,
#   systematic trading

DEBUG = False

# PA update flow(per batch klu): with callback functions
#   1. add new bi (vertex) (if exist)
#   2. add batch volume profile

# PA algos are afflicted under chan.bi, also updated with it

class PA_Core:
    # PA: Price Action

    # top/bot FX has type: loss/half/strict
    # for half/strict FX, weak FX will be overwrite by stronger FX in the same direction,
    # this is effectively a breakthrough from microstructure

    # Chart Patterns: using "loss" type FX is recommended
    # Trendlines:
    # Volume-Profiles:

    def __init__(self, bi_list: CBiList, shape_keys: List[str], liquidity: bool):
        self.bi_len: int = 0
        self.bi_list: CBiList = bi_list
        self.vertex_list: List[vertex] = []
        self.shape_keys = shape_keys
        self.liquidity = liquidity
        
        self.init_PA_elements()
        self.ts_1st: float = 0
        self.ts_cur: float = 0

    def init_PA_elements(self):
        # init shapes
        self.PA_Shapes_developed: Dict[str, List[conv_type]] = {}
        self.PA_Shapes_developing: Dict[str, List[conv_type]] = {}
        self.PA_Shapes_trading_now: Dict[str, conv_type] = {}
        for key in self.shape_keys:
            self.PA_Shapes_developed[key] = []
            self.PA_Shapes_developing[key] = []
        if self.liquidity:
            self.PA_Liquidity: PA_Liquidity = PA_Liquidity()
            self.PA_Volume_Profile: PA_Volume_Profile = PA_Volume_Profile()

    #                     +-------------------------+
    #                     |    add virtual bi       |
    #                     v    (potential bi-break) |
    # [*] ----> [Virtual BI] ----------------> [Sure BI] -----> [Update End] --> [*]
    #            ^    ^   ^  del virtual bi                       |
    #            |    |   |  add sure bi                          |
    #            +----+   |         add virtual bi                |
    #     del virtual bi  |         (potential bi-break)          |
    #     add virtual bi  +---------------------------------------+

    # LOGIC:
    # span_check: at-least 4 k-lines between points
    # FX: if FX formed
    # peak: for top, check there are no lower lows between last bi and attempted new bi
    #       else update the last bi first(becomes static) then add new bi

    # a sure bi is fixed only after arrival of new virtual bi
    # because end of a sure bi can be updated

    # original: bi_list:         sure, sure, sure, sure, sure, sure, x
    # case 1:   bi_list:         sure, sure, sure, sure, sure, sure, x(updated,sure<->virtual)
    #                           |                            static | dynamic
    # case 2:   bi_list:(right)  sure, sure, sure, sure, sure, sure, sure, x
    #                           |                                  static | dynamic
    # case 3:   bi_list:(left)   sure, sure, sure, sure, sure, sure
    #                           |                            static | dynamic
    # note that in case 3, only virtual bi can potentially be removed(shift-left),
    # and is guaranteed not to trigger left-shift twice, so the last sure bi will remain static

    def parse_dynamic_bi_list(self):
        self.check_update_and_shift()
        # first feed new static bi to the static shapes
        if self.shift_l:  # do nothing
            pass
        else:
            if self.shift_r:  # get 1 more static bi
                new_static_bi = self.bi_list[-2]
                self.feed_bi_to_all_PA_elements(new_static_bi)
            # self.get_dynamic_shapes()

    def check_update_and_shift(self):
        bi_len = len(self.bi_list)
        self.shift_l = False
        self.shift_r = False
        if bi_len > self.bi_len and bi_len > 1:
            self.shift_r = True
        elif bi_len < self.bi_len:
            self.shift_l = True
        self.bi_len = bi_len
        return

    def feed_bi_to_all_PA_elements(self, bi: CBi):
        end_x: int = bi.get_end_klu().idx
        end_y: float = bi.get_end_val()
        end_ts: float = bi.get_end_klu().time.ts
        self.end_open: float = bi.get_end_klu().open
        self.end_close: float = bi.get_end_klu().close
        self.end_volume: int = int(bi.get_end_klu().volume)
        end_bi_vertex = vertex(end_x, end_y, end_ts)
        # this is just an ugly fix ([-2] may not be static)
        if end_bi_vertex.ts == self.ts_cur:
            return
        
        if len(self.vertex_list) == 0:
            begin_x: int = bi.get_begin_klu().idx
            begin_y: float = bi.get_begin_val()
            begin_ts: float = bi.get_begin_klu().time.ts
            self.ts_1st = begin_ts
            start_bi_vertex = vertex(begin_x, begin_y, begin_ts)
            self.vertex_list.append(start_bi_vertex)
            self.feed_vertex_to_all_PA_elements(start_bi_vertex)
        self.vertex_list.append(end_bi_vertex)
        self.feed_vertex_to_all_PA_elements(end_bi_vertex)
        if self.liquidity:
            # get bi price_mapped_volume
            bi_VP = self.PA_Volume_Profile.update_volume_profile(bi)
            self.PA_Liquidity.update_volume_zone(bi_VP)
        
        self.ts_cur = end_bi_vertex.ts

    def feed_vertex_to_all_PA_elements(self, vertex: vertex):
        self.add_vertex_to_shapes(vertex)
        if self.liquidity:
            self.add_vertex_to_liquidity(vertex)
            
    def add_vertex_to_shapes(self, vertex: vertex):
        for shape_name in self.shape_keys:
            # Use a slice to make a copy of the list so can remove item on-fly
            for shape in self.PA_Shapes_developing[shape_name][:]:
                # Update existing shapes
                success = shape.add_vertex(vertex)
                if shape.is_complete():
                    self.PA_Shapes_developed[shape_name].append(shape)
                    self.PA_Shapes_developing[shape_name].remove(shape)
                    continue
                if DEBUG:
                    print(shape.name, shape.vertices, shape.state, success)
                if not success:  # try add vertex and failed shape FSM check
                    self.PA_Shapes_developing[shape_name].remove(shape)
            if DEBUG:
                print('=================================================: ', vertex)
            # Start new potential shapes
            if shape_name == 'conv_type':
                self.PA_Shapes_developing[shape_name].append(conv_type(vertex))

    def add_vertex_to_liquidity(self, vertex: vertex):
        self.PA_Liquidity.add_vertex(vertex)

    # # @jit(nopython=True, parallel=False) # acceleration(static compile before run)
    # def process_batch_klu(self, resample_buffer: List[WtNpKline], price_bin_width:float) -> Tuple[CKLine_Unit, List[int|List[int]]]:
    #     # batch -> bi -> session -> history
    #     
    #     # only session volume profile is important:
    #     # how to define session? see PA_TreadLine for details
    #     
    #     # current session = earliest Zigzag(Chan.bi) of all active trendlines <-> current bar
    #     
    #     total_volume:int = 0
    #     high:float = max([klu.highs[-1] for klu in resample_buffer])
    #     low:float = min([klu.lows[-1] for klu in resample_buffer])
    #     first_open:float = resample_buffer[0].opens[-1]
    #     last_close:float = resample_buffer[-1].closes[-1]
    #     index_range_low:int = int(low//price_bin_width) # floor
    #     index_range_high:int = int(high//price_bin_width) # floor
    #     batch_volume_buyside:List[int] = [0] * (index_range_high-index_range_low+1)
    #     batch_volume_sellside:List[int] = [0] * (index_range_high-index_range_low+1)
    #     for klu in resample_buffer:
    #         open:float = klu.opens[-1]
    #         close:float = klu.closes[-1]
    #         volume:int = int(klu.volumes[-1])
    #         total_volume += volume
    #         index:int = int(((close+open)/2)//price_bin_width)
    #         # error correction:
    #         if not (index_range_low < index < index_range_high):
    #             index = index_range_low
    #         if close > open:
    #             batch_volume_buyside[index-index_range_low] += volume
    #         else:
    #             batch_volume_sellside[index-index_range_low] += volume
    #     batch_time:CTime = parse_time_column(str(resample_buffer[-1].bartimes[-1]))
    #     new_bi = False
    #     batch_combined_klu = CKLine_Unit(
    #         {
    #             DATA_FIELD.FIELD_TIME:      batch_time,
    #             DATA_FIELD.FIELD_OPEN:      first_open,
    #             DATA_FIELD.FIELD_HIGH:      high,
    #             DATA_FIELD.FIELD_LOW:       low,
    #             DATA_FIELD.FIELD_CLOSE:     last_close,
    #             DATA_FIELD.FIELD_VOLUME:    total_volume,
    #         },
    #         autofix=True)
    #     batch_volume_profile = [
    #         batch_time,
    #         index_range_low,
    #         index_range_high,
    #         batch_volume_buyside,
    #         batch_volume_sellside,
    #         price_bin_width,
    #         new_bi, # wait for upper function to fill it
    #     ]
    #     return batch_combined_klu, batch_volume_profile

    def get_dynamic_shapes(self):
        """recalculate dynamic shapes if need for potential open position"""
        if not self.shift_l:
            # dynamic_bi = shift_r?: new_dynamic bi : updated_dynamic_bi
            if self.bi_len > 0:
                dynamic_bi: CBi = self.bi_list[-1]
                end_x: int = dynamic_bi.get_end_klu().idx
                end_y: float = dynamic_bi.get_end_val()
                end_ts: float = dynamic_bi.get_end_klu().time.ts
                # end_open:float = dynamic_bi.get_end_klu().open
                # end_close:float = dynamic_bi.get_end_klu().close
                # end_volume:int = int(dynamic_bi.get_end_klu().volume)
                end_bi_vertex = vertex(end_x, end_y, end_ts)

                for shape_name in self.shape_keys:
                    # only interested in the 1st(longest existing) shape at current lv
                    shapes = self.PA_Shapes_developing[shape_name]
                    if len(shapes) < 2:
                        continue
                    shape_active = copy.deepcopy(shapes[0])
                    success = shape_active.add_vertex(end_bi_vertex)
                    if success and shape_active.is_potential():
                        self.PA_Shapes_trading_now[shape_name] = shape_active
                return self.PA_Shapes_trading_now

    def get_static_shapes(self, complete: bool = False, potential: bool = False, with_idx: bool = False):
        shapes: Dict[str, List[conv_type]] = {}
        exist = False
        for shape_name in self.shape_keys:
            shapes[shape_name] = []
            if complete:
                for shape in self.PA_Shapes_developed[shape_name]:
                    shapes[shape_name].append(shape)
                    exist = True
            if potential:
                Shapes_developing = self.PA_Shapes_developing[shape_name]
                if len(Shapes_developing) != 0:
                    # only return the longest existing one
                    shapes[shape_name].append(Shapes_developing[0])
                    exist = True
                # if with_idx:
                #     shapes.append(shape.state)
        return exist, shapes
