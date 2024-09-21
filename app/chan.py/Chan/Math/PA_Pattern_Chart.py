# PA: Price Action
# ref: https://tmipartner.com/lessons/institution/
# ref: https://www.youtube.com/watch?v=i0hPu7eNty0

# continuation pattern:
#   up:
#       .bearish flag
#       .bullish pennant
#       .ascending triangle
#       .bullish channel
#   down:
#       .bullish flag
#       .bearish pennant
#       .descending triangle
#       .bearish channel
#   flat
#       cup with handle
# neutral pattern(dir of breakouts):
#       .symmetrical triangle
#       .rectangle
# Reversal:
#   flat:
#       double top/bot with no liquidation grab
#       triple or more top/bot with liquidation grab and sup/res test
#       Head and shoulders
#       diamond top
#       round with 2 handles
#       bump and run
#       .megaphone
#   up:
#       .falling wedge
#   down:
#       .rising wedge

# 楔形回调入场
# 窄幅区间突破入场
# 双十字星入场
# 二次，三次突破入场
# 失败的失败入场
# 强趋势K线突破入场
# ref: https://www.youtube.com/watch?v=NZKpJ71iZyU
# ref: https://www.youtube.com/watch?v=2wIqCWKnWVk

import copy
import math
import numpy as np
from numpy.polynomial import Polynomial
from typing import List, Dict

from Chan.Bi.Bi import CBi
from Chan.Common.CEnum import BI_DIR, TREND_LINE_SIDE

debug = False

# Chart Pattern is afflicted under chan.bi, also updated with it
class Chart_Patterns:
    def __init__(self):
        self.bi_lst: List[List[int|float]] = []
        self.bi_lst_is_sure: bool = True # if the last bi in bi_lst is sure
        self.keys = ['nexus_type',]
        self.shapes = self.init_shapes()
        self.shapes_deep_copy = self.shapes
        
    def init_shapes(self):
        shapes: Dict[str, List[
            nexus_type
            ]] = {}
        for key in self.keys:
            shapes[key] = []
        return shapes
    
    def get_shapes(self, complete:bool=False, potential:bool=False, with_idx:bool=False):
        shapes:List[
            nexus_type
            ] = []
        for shape_name in self.keys:
            for shape in self.shapes[shape_name]:
                if complete and shape.is_complete():
                    shapes.append(shape)
                elif potential and not shape.is_potential():
                    shapes.append(shape)
                
                # if with_idx:
                #     shapes.append(shape.state)
        return shapes
    
    def add_bi(self, bi:CBi, is_sure:bool = False):
        self.bi_idx:int = bi.idx
        end_x:int = bi.get_end_klu().idx
        end_y:float = bi.get_end_val()
        end_bi_vertex = [end_x, end_y]
        print('add bi: ', bi.idx, is_sure, end_bi_vertex)
        if len(self.bi_lst) == 0:
            if is_sure: # skip fist few bi that is not sure
                begin_x:int = bi.get_begin_klu().idx
                begin_y:float = bi.get_begin_val()
                start_bi_vertex = [begin_x, begin_y]
                self.bi_lst.append(start_bi_vertex)
                self.feed_vertex_to_all_chart_patterns(is_sure)
                self.bi_lst.append(end_bi_vertex)
        elif self.bi_lst_is_sure:
            self.bi_lst.append(end_bi_vertex)
        else:
            self.bi_lst[-1] = end_bi_vertex
        self.feed_vertex_to_all_chart_patterns(is_sure)
        self.bi_lst_is_sure = is_sure
        # if debug:
        #     print(self.bi_lst)
        
    def feed_vertex_to_all_chart_patterns(self, is_sure:bool):
        vertex = self.bi_lst[-1]
        # only detect bi-level shapes, not seg-level shapes
        if self.bi_lst_is_sure and not is_sure:
            self.shapes_deep_copy = copy.deepcopy(self.shapes)
        if not self.bi_lst_is_sure or not is_sure:
            self.shapes = copy.deepcopy(self.shapes_deep_copy)
                            
        self.add_vertex_to_shapes(vertex, is_sure)
        
    def add_vertex_to_shapes(self, vertex:List[int|float], is_sure:bool):
        for shape_name in self.keys:
            num_of_shape = len(self.shapes[shape_name])
            if num_of_shape > 0:
                # Update existing shapes
                for shape in self.shapes[shape_name][:]: # Use a slice to make a copy of the list so can remove item on-fly
                    if shape.is_complete():
                        continue
                    success = shape.add_vertex(vertex)
                    if debug:
                        print(shape.name, shape.vertices, shape.state, success)
                    if not success: # try add vertex and failed shape FSM check
                        self.shapes[shape_name].remove(shape)
            if is_sure:
                if debug:
                    print('=================================================: ', vertex)
                # Start new potential shapes
                if shape_name == 'nexus_type':
                    if is_sure:
                        self.shapes[shape_name].append(nexus_type(vertex))
                        
def util_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def util_angle(p1, p2, p3):
    a = util_distance(p2, p3)
    b = util_distance(p1, p3)
    c = util_distance(p1, p2)
    return math.acos((a**2 + c**2 - b**2) / (2 * a * c))

def fit_linear(vertices:List[List[int|float]]):
    x_coords = np.array([v[0] for v in vertices])
    y_coords = np.array([v[1] for v in vertices])
    
    # Number of points
    n = len(vertices)
    
    # Calculate sums needed for slope and intercept
    sum_x = np.sum(x_coords)
    sum_y = np.sum(y_coords)
    sum_x_squared = np.sum(x_coords**2)
    sum_xy = np.sum(x_coords * y_coords)
    # print(sum_x, sum_y, sum_x_squared, sum_xy)
    
    # Calculate slope (m) and intercept (b)
    m:float = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
    b:float = (sum_y - m * sum_x) / n
    m_percent: float = m / y_coords[0]
    
    # Compute the predicted y values using the fitted line
    y_pred = m * x_coords + b
    
    # Calculate the residuals (difference between actual and predicted y-values)
    residuals = y_coords - y_pred
    total_residual:float = np.sum(np.abs(residuals))  # sum of absolute residuals
    
    x:List[int] = [x_coords[0], x_coords[-1]]
    y:List[float] = [y_pred[0], y_pred[-1]]
    return m_percent, x, y, total_residual
    
pnl = 2 # pnl (filter: harder to get in/out of micro-structure)

# bullish flag
# bearish pennant
# descending triangle
# bearish channel
# NOTE: usually it is better to wait for price break BOTH chart and certain support
class nexus_type: # continuation or breakout or reversal
    START = 0
    ENTRY = 1
    RISING_M = 2 # M: potential multiple entry
    FALLING_M = 3 # M: potential multiple entry
    COMPLETED = 4
    def __init__(self, start_vertex:List[int|float]):
        self.vertices = [start_vertex]
        self.state = self.START
        self.rising_cnt:int = 0
        self.falling_cnt:int = 0
        self.entry_dir:int = 0
        self.abs_d1:float = 0 # entry absolute delta-y
        
        self.max_drawdown:float = 0
        self.up_support:float = 0
        self.down_support:float = 0
        self.up_resistance:float = 0
        self.down_resistance:float = 0
        self.high:float = 0
        self.low:float = start_vertex[1]
        
        self.top_m = 0, 0
        self.top_x = [0, 0]
        self.top_y = [0.0, 0.0]
        self.top_residue = 0
        self.bot_m = 0, 0
        self.bot_x = [0, 0]
        self.bot_y = [0.0, 0.0]
        self.bot_residue = 0
        
        self.potential_trade:bool = False
        self.name:str = ''
        self.color:str = 'purple'
        
    def is_potential(self): # the trading opportunity is ongoing
        return self.state == self.potential_trade
    
    def is_complete(self): # the trading opportunity is over
        return self.state == self.COMPLETED
    
    # def small_pct_diff(self, a:float, b:float, align_pct:float) -> bool:
    #     return abs(a - b) < align_pct * min(abs(a), abs(b))
    
    def check_shape_fit(self, vertices, last_vertex_type:str):
        if last_vertex_type == 'top':
            top_vertices = vertices[-1::-2]  # Take -1, -3, -5, etc. (reversed odd indices)
            bottom_vertices = vertices[-2::-2]  # Take -2, -4, -6, etc. (reversed even indices)
        elif last_vertex_type == 'bot':
            top_vertices = vertices[-2::-2]  # Take -1, -3, -5, etc. (reversed odd indices)
            bottom_vertices = vertices[-1::-2]  # Take -2, -4, -6, etc. (reversed even indices)
        top_m, top_x, top_y, top_residue = fit_linear(top_vertices)
        bot_m, bot_x, bot_y, bot_residue = fit_linear(bottom_vertices)
        
        if max(top_residue, bot_residue) > 0.4:
            return False
        
        self.top_m, self.top_x, self.top_y, self.top_residue = top_m, top_x, top_y, top_residue
        self.bot_m, self.bot_x, self.bot_y, self.bot_residue = bot_m, bot_x, bot_y, bot_residue
        
        UP = 0.001
        DOWN = -0.001
        # define entry_dir and near bar, far bar
        # away: continue
        # revs: reversal
        if self.entry_dir==1: # UP entry
            m_far = self.top_m
            m_near = self.bot_m
            far_away = m_far > UP
            far_revs = m_far < DOWN
            near_away = m_near > UP
            near_revs = m_near < DOWN
        elif self.entry_dir==-1: # DOWN entry
            m_near = self.top_m
            m_far = self.bot_m
            far_away = m_far < DOWN
            far_revs = m_far > UP
            near_away = m_near < DOWN
            near_revs = m_near > UP
            
        far_cons = not (far_away or far_revs)                           # consolidate
        near_cons = not (near_away or near_revs)                        # consolidate
        
        self.name = f'undefined'
        # bellow 0.5% change on High_High / Low_Low per bar 
        # on average is considered "sideways consolidation"
        self.potential_trade = False
        if near_revs and far_revs:
            self.name = f'flag'             # continuation
        elif near_away and far_away:
            self.name = f'channel'          # continuation
        elif near_cons and far_cons:
            self.name = f'rect'             # continuation or breakout

        elif near_revs and far_away:
            self.name = f'meg_sym'          # breakout, low pnl :(
        elif near_revs and far_cons:
            self.name = f'meg_brk_far'      # continuation
        elif near_cons and far_away:
            self.name = f'meg_rev_bak'      # reversal
        elif near_away and far_revs:
            self.name = f'tri_sym'          # breakout
        elif near_away and far_cons:
            self.name = f'tri_brk_far'      # continuation or reversal(wedge)
        elif near_cons and far_revs:
            self.name = f'tri_rev_bak'      # reversal
        
        if self.entry_dir == 1:
            self.name += f'.UP'
        elif self.entry_dir == -1:
            self.name += f'.DN'
            
        # self.name += f'.{self.top_residue:.3f}.{self.bot_residue:.3f}'
        # self.name += f'{m_near:.3f}.{m_far:.3f}.{top_vertices[0]}.{top_vertices[-1]}'
        # print('top: ', top_vertices,    self.top_x, self.top_y, self.top_residue)
        # print('bot: ', bottom_vertices, self.bot_x, self.bot_y, self.bot_residue)
        return True
    
    def add_vertex(self, new_vertex:List[int|float]):
        UP = 1
        DOWN = -1
        if self.state == self.COMPLETED:
            return True
        if self.rising_cnt > 8 or self.falling_cnt > 8: # abnormal shape
            return False
        
        last_vertex = self.vertices[-1]
        delta_y = new_vertex[1] - last_vertex[1]
        delta_x = new_vertex[0] - last_vertex[0]
        if self.state == self.START:
            if delta_y < 0:
                self.entry_dir = DOWN
                self.abs_d1 = -delta_y
                self.max_drawdown = self.abs_d1/pnl
                self.down_resistance = new_vertex[1] + self.max_drawdown
                self.state = self.ENTRY
                self.vertices.append(new_vertex)
                return True
            else:
                self.entry_dir = UP
                self.abs_d1 = delta_y
                self.max_drawdown = self.abs_d1/pnl
                self.up_support = new_vertex[1] - self.max_drawdown
                self.state = self.ENTRY
                self.vertices.append(new_vertex)
                return True
            
        if self.state == self.ENTRY:
            if delta_y > 0:
                if delta_y < self.max_drawdown:
                    if new_vertex[1] < self.down_resistance:
                        self.down_support = new_vertex[1] - self.max_drawdown
                        self.vertices.append(new_vertex)
                        self.state = self.RISING_M
                        self.rising_cnt += 1
                        return True
                return False
            else:
                if -delta_y < self.max_drawdown:
                    if new_vertex[1] > self.up_support:
                        self.up_resistance = new_vertex[1] + self.max_drawdown
                        self.vertices.append(new_vertex)
                        self.state = self.FALLING_M
                        self.falling_cnt += 1
                        return True
                return False
            
        if self.state == self.RISING_M:
            if delta_y < 0:
                if -delta_y < self.max_drawdown:
                    if self.entry_dir == DOWN and new_vertex[1] > self.down_support:
                        self.vertices.append(new_vertex)
                        self.state = self.FALLING_M
                        if self.rising_cnt > 1:
                            if not self.check_shape_fit(self.vertices[1:], 'bot'):
                                # if residue goes out of bound, it is a breakout
                                self.state = self.COMPLETED
                                self.vertices.pop()
                        return True
                    if self.entry_dir == UP and new_vertex[1] > self.up_support:
                        self.up_resistance = new_vertex[1] + self.max_drawdown
                        self.vertices.append(new_vertex)
                        self.state = self.FALLING_M
                        self.falling_cnt += 1
                        if self.falling_cnt > 1:
                            if not self.check_shape_fit(self.vertices[1:], 'bot'):
                                self.state = self.COMPLETED
                                self.vertices.pop()
                        return True
                    
                # break resistance/support, opportunity lost
                if (self.entry_dir == DOWN and self.rising_cnt > 1) or \
                    (self.entry_dir == UP and self.falling_cnt > 1):
                    self.vertices.append(new_vertex)
                    self.potential_trade = False
                    self.state = self.COMPLETED
                    return True
                else:
                    return False
            else:
                return False
        
        if self.state == self.FALLING_M:
            if delta_y > 0:
                if delta_y < self.max_drawdown:
                    if self.entry_dir == DOWN and new_vertex[1] < self.down_resistance:
                        self.down_support = new_vertex[1] - self.max_drawdown
                        self.vertices.append(new_vertex)
                        self.state = self.RISING_M
                        self.rising_cnt += 1
                        if self.rising_cnt > 1:
                            if not self.check_shape_fit(self.vertices[1:], 'top'):
                                # if residue goes out of bound, it is a breakout
                                self.state = self.COMPLETED
                                self.vertices.pop()
                        return True
                    if self.entry_dir == UP and new_vertex[1] < self.up_resistance:
                        self.vertices.append(new_vertex)
                        self.state = self.RISING_M
                        if self.falling_cnt > 1:
                            if not self.check_shape_fit(self.vertices[1:], 'top'):
                                self.state = self.COMPLETED
                                self.vertices.pop()
                        return True
                    
                # break resistance/support, opportunity lost
                if (self.entry_dir == DOWN and self.rising_cnt > 1) or \
                    (self.entry_dir == UP and self.falling_cnt > 1):
                    self.vertices.append(new_vertex)
                    self.potential_trade = False
                    self.state = self.COMPLETED
                    return True
                else:
                    return False
            else:
                return False
        
        print('case not covered: ', self.state)
        return False
    
