import copy
import os, sys
import math
import numpy as np
from numpy.polynomial import Polynomial
from typing import List, Dict

from Chan.Bi.Bi import CBi
from Chan.Math.PA_types import vertex
from Chan.Math.PA_Pattern_Chart import nexus_type
from Chan.Math.PA_Liquidity import PA_Liquidity
from Chan.Math.PA_Volume_Profile import PA_Volume_Profile

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
    
    def __init__(self):
        self.bi_lst: List[vertex] = []
        self.bi_lst_is_sure: bool = True # if the last bi in bi_lst is sure
        self.shape_keys = ['nexus_type',]
        self.init_PA_elements()
        
    def add_bi(self, bi:CBi, is_sure:bool = False):
        # bi would be updated multiple times until it is sure
        # update bi-based metric whenever new bi comes in, stabilize it when is sure
        self.bi_idx:int = bi.idx
        end_x:int = bi.get_end_klu().idx
        end_y:float = bi.get_end_val()
        self.end_open:float = bi.get_end_klu().open
        self.end_close:float = bi.get_end_klu().close
        self.end_volume:int = int(bi.get_end_klu().volume)
        end_bi_vertex = vertex(end_x, end_y)
        # print('add bi: ', bi.idx, is_sure, end_bi_vertex)
        if len(self.bi_lst) == 0:
            if is_sure: # skip fist few bi that is not sure
                begin_x:int = bi.get_begin_klu().idx
                begin_y:float = bi.get_begin_val()
                start_bi_vertex = vertex(begin_x, begin_y)
                self.bi_lst.append(start_bi_vertex)
                self.feed_vertex_to_all_PA_elements(is_sure)
                self.bi_lst.append(end_bi_vertex)
        elif self.bi_lst_is_sure:
            self.bi_lst.append(end_bi_vertex)
        else:
            self.bi_lst[-1] = end_bi_vertex
        self.feed_vertex_to_all_PA_elements(is_sure)
        self.bi_lst_is_sure = is_sure
        # if DEBUG:
        #     print(self.bi_lst)
        
    def init_PA_elements(self):
        # init shapes
        self.PA_Shapes: Dict[str, List[
            nexus_type
            ]] = {}
        for key in self.shape_keys:
            self.PA_Shapes[key] = []

        self.PA_Liquidity: PA_Liquidity = PA_Liquidity()
        
        self.PA_Volume_Profile: PA_Volume_Profile = PA_Volume_Profile()
        
        self.shapes_deep_copy = self.PA_Shapes
        # self.liquidity_deep_copy = self.PA_Liquidity
    
    def feed_vertex_to_all_PA_elements(self, is_sure:bool):
        vertex = self.bi_lst[-1]
        # only detect bi-level shapes, not seg-level shapes
        if self.bi_lst_is_sure and not is_sure:
            self.shapes_deep_copy = copy.deepcopy(self.PA_Shapes)
        if not self.bi_lst_is_sure or not is_sure:
            self.PA_Shapes = copy.deepcopy(self.shapes_deep_copy)
                            
        self.add_vertex_to_shapes(vertex, is_sure)
        self.add_vertex_to_liquidity(vertex, is_sure, self.end_open, self.end_close, self.end_volume)
        
    def add_vertex_to_shapes(self, vertex:vertex, is_sure:bool):
        for shape_name in self.shape_keys:
            num_of_shape = len(self.PA_Shapes[shape_name])
            if num_of_shape > 0:
                # Update existing shapes
                for shape in self.PA_Shapes[shape_name][:]: # Use a slice to make a copy of the list so can remove item on-fly
                    if shape.is_complete():
                        continue
                    success = shape.add_vertex(vertex)
                    if DEBUG:
                        print(shape.name, shape.vertices, shape.state, success)
                    if not success: # try add vertex and failed shape FSM check
                        self.PA_Shapes[shape_name].remove(shape)
            if is_sure:
                if DEBUG:
                    print('=================================================: ', vertex)
                # Start new potential shapes
                if shape_name == 'nexus_type':
                    if is_sure:
                        self.PA_Shapes[shape_name].append(nexus_type(vertex))
                        
    def add_vertex_to_liquidity(self, vertex:vertex, is_sure:bool, end_open:float, end_close:float, end_volume:int):
        # liquidity zone should be formed at breakthrough, but for ease of computation
        # only update at FX formation
        
        if is_sure: # it is fine to update later
            self.PA_Liquidity.add_vertex(vertex, end_open, end_close, end_volume)
            
    def get_chart_pattern_shapes(self, complete:bool=False, potential:bool=False, with_idx:bool=False):
        shapes:List[
            nexus_type
            ] = []
        for shape_name in self.shape_keys:
            for shape in self.PA_Shapes[shape_name]:
                if complete and shape.is_complete():
                    shapes.append(shape)
                elif potential and not shape.is_potential():
                    shapes.append(shape)

                # if with_idx:
                #     shapes.append(shape.state)
        return shapes
    
    def get_liquidity_class(self) -> PA_Liquidity:
        return self.PA_Liquidity
    
    def get_volume_profile(self) -> PA_Volume_Profile:
        return self.PA_Volume_Profile