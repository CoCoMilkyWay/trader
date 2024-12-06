# PA: Price Action
# ref: https://tmipartner.com/lessons/institution/
# ref: https://www.youtube.com/watch?v=i0hPu7eNty0
# ref: https://www.youtube.com/watch?v=sWTnFS10tdQ

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

# regress to filter out FX @ balanced zone(no profit), and try to fix FX @ premium zone better

import os
import sys
import math
import numpy as np
from typing import List, Dict, Literal
from app.PA.PA_types import vertex
from Util.Filter import AverageFilter

CONV_PNL = 2  # pnl (filter: harder to get in/out of micro-structure)

def util_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def util_angle(p1, p2, p3):
    a = util_distance(p2, p3)
    b = util_distance(p1, p3)
    c = util_distance(p1, p2)
    return math.acos((a**2 + c**2 - b**2) / (2 * a * c))


def fit_linear(vertices: List[vertex]):
    x_coords = np.array([v.idx for v in vertices])
    y_coords = np.array([v.value for v in vertices])

    # Number of points
    n = len(vertices)

    # Calculate sums needed for slope and intercept
    sum_x = np.sum(x_coords)
    sum_y = np.sum(y_coords)
    sum_x_squared = np.sum(x_coords**2)
    sum_xy = np.sum(x_coords * y_coords)
    # print(sum_x, sum_y, sum_x_squared, sum_xy)

    # Calculate slope (m) and intercept (b)
    m: float = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
    b: float = (sum_y - m * sum_x) / n
    m_percent: float = m / y_coords[0]

    # Compute the predicted y values using the fitted line
    y_pred = m * x_coords + b

    # Calculate the residuals (difference between actual and predicted y-values)
    residuals = y_coords - y_pred
    normalized_total_residual: float = np.sum(
        np.abs(residuals))/y_coords[-1]  # sum of absolute residuals

    x: List[int] = [x_coords[0], x_coords[-1]]
    y: List[float] = [y_pred[0], y_pred[-1]]
    ts: List[float] = [vertices[0].ts, vertices[-1].ts]
    return m_percent, x, y, ts, normalized_total_residual

# convergence after rapid price movement


class conv_type:  # continuation or breakout or reversal or nexus
    START = 0
    ENTRY = 1
    RISING_M = 2  # M: potential multiple entry
    FALLING_M = 3  # M: potential multiple entry
    COMPLETED = 4

    def __init__(self, start_vertex: vertex):
        self.vertices = [start_vertex]
        self.state = self.START
        self.rising_cnt: int = 0
        self.falling_cnt: int = 0
        self.entry_dir: int = 0
        self.abs_d1: float = 0  # entry absolute delta-y

        self.far_cons: bool = False
        self.near_cons: bool = False

        self.max_drawdown: float = 0
        self.up_support: float = 0
        self.down_support: float = 0
        self.up_resistance: float = 0
        self.down_resistance: float = 0
        # self.high: float = 0
        # self.low: float = start_vertex.value

        self.top_m = 0.0
        self.top_x = [0, 0]
        self.top_y = [0.0, 0.0]
        self.top_ts = [0.0, 0.0]
        self.top_residue = 0.0
        self.bot_m = 0.0
        self.bot_x = [0, 0]
        self.bot_y = [0.0, 0.0]
        self.bot_ts = [0.0, 0.0]
        self.bot_residue = 0.0

        self.potential_trade: bool = False
        self.name: str = ''
        self.bi_avg_delta_x = AverageFilter(50)
        self.bi_avg_delta_y = AverageFilter(50)
        # self.name_onehot:List[bool] = [True, True, True, True]

    def is_potential(self):  # the trading opportunity is ongoing
        return self.potential_trade

    def is_complete(self):  # the trading opportunity is over
        return self.state == self.COMPLETED

    # def small_pct_diff(self, a:float, b:float, align_pct:float) -> bool:
    #     return abs(a - b) < align_pct * min(abs(a), abs(b))

    def check_shape_fit(self, vertices: List[vertex], last_vertex_type: str):
        if last_vertex_type == 'top':
            # Take -1, -3, -5, etc. (reversed odd indices)
            top_vertices = vertices[-1::-2]
            # Take -2, -4, -6, etc. (reversed even indices)
            bottom_vertices = vertices[-2::-2]
        elif last_vertex_type == 'bot':
            # Take -1, -3, -5, etc. (reversed odd indices)
            top_vertices = vertices[-2::-2]
            # Take -2, -4, -6, etc. (reversed even indices)
            bottom_vertices = vertices[-1::-2]
        # print([v.ts for v in top_vertices])
        # print([v.ts for v in bottom_vertices])
        self.top_m, self.top_x, self.top_y, self.top_ts, self.top_residue = fit_linear(
            top_vertices)
        self.bot_m, self.bot_x, self.bot_y, self.bot_ts, self.bot_residue = fit_linear(
            bottom_vertices)

        bi_avg_delta_x = self.bi_avg_delta_x.get_average()
        bi_avg_delta_y = self.bi_avg_delta_y.get_average()
        THD_RESIDUE = bi_avg_delta_y/vertices[-1].value
        THD_SLOPE = bi_avg_delta_y/vertices[-1].value/bi_avg_delta_x/20
        if max(self.top_residue, self.bot_residue) > THD_RESIDUE:
            return False

        UP = THD_SLOPE
        DOWN = -THD_SLOPE
        # print(self.thd, self.top_m, self.bot_m)
        # define entry_dir and near bar, far bar
        # away: continue
        # revs: reversal
        if self.entry_dir == 1:  # UP entry
            m_far = self.top_m
            m_near = self.bot_m
            far_away = m_far > UP
            far_revs = m_far < DOWN
            near_away = m_near > UP
            near_revs = m_near < DOWN
        elif self.entry_dir == -1:  # DOWN entry
            m_near = self.top_m
            m_far = self.bot_m
            far_away = m_far < DOWN
            far_revs = m_far > UP
            near_away = m_near < DOWN
            near_revs = m_near > UP

        self.far_cons = not (far_away or far_revs)      # consolidate
        self.near_cons = not (near_away or near_revs)   # consolidate

        self.name = f'undefined'
        # self.name_onehot = [True, True, True, True]
        # bellow 0.5% change on High_High / Low_Low per bar
        # on average is considered "sideways consolidation"
        # self.potential_trade = False
        if near_revs and far_revs:
            self.name = f'flag'             # continuation
            # self.name_onehot = [True, False, False, False]
        elif near_away and far_away:
            self.name = f'channel'          # continuation
            # self.name_onehot = [False, False, False, False]
        elif self.near_cons and self.far_cons:
            self.name = f'rect'             # continuation or breakout
            # self.name_onehot = [False, False, False, True]

        elif near_revs and far_away:
            self.name = f'meg_sym'          # breakout, low pnl :(
            # self.name_onehot = [False, False, True, False]
        elif near_revs and self.far_cons:
            self.name = f'meg_brk_far'      # continuation
            # self.name_onehot = [False, False, True, True]
        elif self.near_cons and far_away:
            self.name = f'meg_rev_bak'      # reversal
            # self.name_onehot = [False, True, False, False]
        elif near_away and far_revs:
            self.name = f'tri_sym'          # breakout
            # self.name_onehot = [False, True, False, True]
        elif near_away and self.far_cons:
            self.name = f'tri_brk_far'      # continuation or reversal(wedge)
            # self.name_onehot = [False, True, True, False]
        elif self.near_cons and far_revs:
            self.name = f'tri_rev_bak'      # reversal
            # self.name_onehot = [False, True, True, True]

        if self.entry_dir == 1:
            self.name += f'.UP'
        elif self.entry_dir == -1:
            self.name += f'.DN'

        # print(self.name, THD_SLOPE, self.top_m, self.bot_m)
        # self.name += f'.{self.top_residue:.3f}.{self.bot_residue:.3f}'
        # self.name += f'{m_near:.3f}.{m_far:.3f}.{top_vertices[0]}.{top_vertices[-1]}'
        # print('top: ', top_vertices,    self.top_x, self.top_y, self.top_residue)
        # print('bot: ', bottom_vertices, self.bot_x, self.bot_y, self.bot_residue)
        return True

    def add_vertex(self, new_vertex: vertex):
        UP = 1
        DOWN = -1
        if self.state == self.COMPLETED:
            return True
        # if self.rising_cnt > 8 or self.falling_cnt > 8: # abnormal shape
        #     return False

        last_1_vertex = self.vertices[-1]
        delta_y = new_vertex.value - last_1_vertex.value
        delta_x = new_vertex.idx - last_1_vertex.idx
        # update average x/y of be (determine slope/residue)
        self.bi_avg_delta_x.update(delta_x)
        self.bi_avg_delta_y.update(abs(delta_y))

        if self.state == self.START:
            if delta_y < 0:
                self.entry_dir = DOWN
                self.abs_d1 = -delta_y
                self.max_drawdown = self.abs_d1/CONV_PNL
                self.down_resistance = new_vertex.value + self.max_drawdown
                self.state = self.ENTRY
                self.vertices.append(new_vertex)
                return True
            else:
                self.entry_dir = UP
                self.abs_d1 = delta_y
                self.max_drawdown = self.abs_d1/CONV_PNL
                self.up_support = new_vertex.value - self.max_drawdown
                self.state = self.ENTRY
                self.vertices.append(new_vertex)
                return True

        if self.state == self.ENTRY:
            if delta_y > 0:
                if delta_y < self.max_drawdown:
                    if new_vertex.value < self.down_resistance:
                        self.down_support = new_vertex.value - self.max_drawdown
                        self.vertices.append(new_vertex)
                        self.state = self.RISING_M
                        self.name = 'entry'
                        self.rising_cnt += 1
                        return True
                return False
            else:
                if -delta_y < self.max_drawdown:
                    if new_vertex.value > self.up_support:
                        self.up_resistance = new_vertex.value + self.max_drawdown
                        self.vertices.append(new_vertex)
                        self.state = self.FALLING_M
                        self.name = 'entry'
                        self.falling_cnt += 1
                        return True
                return False

        if self.state == self.RISING_M:
            if delta_y < 0:
                self.falling_cnt += 1
                self.state = self.FALLING_M
                self.vertices.append(new_vertex)
                if -delta_y < self.max_drawdown:
                    if self.entry_dir == DOWN and new_vertex.value > self.down_support:
                        self.down_resistance = new_vertex.value + self.max_drawdown
                        if self.rising_cnt > 1:
                            if self.check_shape_fit(self.vertices[1:], 'bot'):
                                self.potential_trade = True
                            else:
                                self.potential_trade = False
                                self.state = self.COMPLETED
                        return True
                    if self.entry_dir == UP and new_vertex.value > self.up_support:
                        self.up_resistance = new_vertex.value + self.max_drawdown
                        if self.rising_cnt > 1:
                            if self.check_shape_fit(self.vertices[1:], 'bot'):
                                self.potential_trade = True
                            else:
                                self.potential_trade = False
                                self.state = self.COMPLETED
                        return True

                # break resistance/support after convergence formed
                if (self.entry_dir == DOWN and self.rising_cnt > 1) or \
                        (self.entry_dir == UP and self.rising_cnt > 1):
                    self.check_shape_fit(self.vertices[1:-1], 'top')
                    self.potential_trade = False
                    self.state = self.COMPLETED
                    return True
                else:  # breakout before convergence
                    return False
            else:  # illegal
                return False

        if self.state == self.FALLING_M:
            if delta_y > 0:
                self.rising_cnt += 1
                self.state = self.RISING_M
                self.vertices.append(new_vertex)
                if delta_y < self.max_drawdown:
                    if self.entry_dir == DOWN and new_vertex.value < self.down_resistance:
                        self.down_support = new_vertex.value - self.max_drawdown
                        if self.falling_cnt > 1:
                            if self.check_shape_fit(self.vertices[1:], 'top'):
                                self.potential_trade = True
                            else:
                                self.potential_trade = False
                                self.state = self.COMPLETED
                        return True
                    if self.entry_dir == UP and new_vertex.value < self.up_resistance:
                        self.up_support = new_vertex.value - self.max_drawdown
                        if self.falling_cnt > 1:
                            if self.check_shape_fit(self.vertices[1:], 'top'):
                                self.potential_trade = True
                            else:
                                self.potential_trade = False
                                self.state = self.COMPLETED
                        return True

                # break resistance/support after convergence formed
                if (self.entry_dir == DOWN and self.falling_cnt > 1) or \
                        (self.entry_dir == UP and self.falling_cnt > 1):
                    self.check_shape_fit(self.vertices[1:-1], 'bot')
                    self.potential_trade = False
                    self.state = self.COMPLETED
                    return True
                else:  # breakout before convergence
                    return False
            else:  # illegal
                return False

        print('case not covered: ', self.state)
        return False

# for head&shoulder, double/triple top/bot, it is better to model them directly as
# liquidity distributions


class round_type:
    # cup with handle, round shape
    # less frequent
    # TODO
    pass
