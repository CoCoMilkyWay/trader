import numpy as np
from typing import Tuple, List, Dict

from Chan.Bi.BiList import CBiList
from Chan.Common.CEnum import FX_TYPE, BI_DIR

from config.cfg_cpt import cfg_cpt

# 15M Bar for S&R analysis
# 1M Bar for entry

# mini-head-and-shoulder
# mini-3-points-trendline

PULLBACK_RATIO = 2

class Mini_Entry_Pattern:
    def __init__(self, bi_list: CBiList):
        self.bi_list = bi_list
        
        # State variables
        # self.prev_bi_index = 0
        
        # Pattern detection results
        self.current_patterns = []
        
        # Configuration
        self.head_shoulder_points = 6  # Number of points needed for H&S pattern
        self.trendline_points = 6      # Number of points needed for 3-point trendline
        # max([NO.points of shapes])
        # also, because last bi is usually formed prematurely(which is okay),
        # we expand shape searching range by 1~2 bi
        self.num_points = 6
        
    def check_patterns(self, long_short:bool) -> Tuple[bool, List[float], str, float, float]: # return list of shape values
        """
        Update pattern detection when new bi points are available
        Returns list of detected patterns
        """
        # current_bi_index = self.bi_list[-2].idx # [-2] is the static bi
        # 
        # # Check if new bi points are available
        # if current_bi_index > self.prev_bi_index:
        #     # Process new bi points
        #     self.prev_bi_index = current_bi_index
        return self._check_patterns(long_short)
    
    def _check_patterns(self, long_short:bool):
        """Check for both head&shoulder and trendline patterns"""
        if len(self.bi_list) <= self.num_points:
            return (False,[],'',0.0,0.0)
        
        self.vtx:List[float] = [] # vertex value
        self.idx:List[float] = [] # vertex index
        for i in range(-self.num_points, 0):
            self.vtx.append(self.bi_list[i].get_end_val())
            self.idx.append(self.bi_list[i].idx)
        if long_short:
            found, pullback = self._is_head_shoulder_shape(inverse=True)
            if found:
                return (True, self.idx, 'H&Sv', self.vtx[-1], pullback)
            found, pullback = self._is_trendline_shape(inverse=False)
            if found: 
                return (True, self.idx, 'TL3v', self.vtx[-1], pullback)
        else:
            found, pullback = self._is_head_shoulder_shape(inverse=False)
            if found:
                return (True, self.idx, 'H&S^', self.vtx[-1], pullback)
            found, pullback = self._is_trendline_shape(inverse=True)
            if found: 
                return (True, self.idx, 'TL3^', self.vtx[-1], pullback)
                
        return (False,[],'',0.0,0.0)
        
    def _is_head_shoulder_shape(self, inverse: bool = False) -> Tuple[bool, float]:
        """
        Check if values form a head and shoulder pattern
        inverse=True checks for inverse head and shoulders
        """
        
        # breakout people got ass-kicked and we join the winning side :)
        #         -3         #
        #    -5   /\    -1   #
        #    /\  /  \  /\    #
        #   /  \/    \/  -0  #
        #  /   -4    -2      #
        # /-6                #
        
        # For regular H&S: shoulders should be lower than head
        # For inverse H&S: shoulders should be higher than head
        if not inverse:
            if self.bi_list[-1].dir == BI_DIR.UP:
                if self.vtx[-3] > max(self.vtx[-5], self.vtx[-1]):
                    if self.vtx[-2] < self.vtx[-5]:
                        pullback = min(abs(self.vtx[-1]-self.vtx[-2]),abs(self.vtx[-5]-self.vtx[-4]))
                        #if self.vtx[-6] < min(self.vtx[-4], self.vtx[-2])-pullback:
                        return True, pullback/PULLBACK_RATIO
            return False, 0.0
        else:
            if self.bi_list[-1].dir == BI_DIR.DOWN:
                if self.vtx[-3] < min(self.vtx[-5], self.vtx[-1]):
                    if self.vtx[-2] > self.vtx[-5]:
                        pullback = min(abs(self.vtx[-1]-self.vtx[-2]),abs(self.vtx[-5]-self.vtx[-4]))
                        #if self.vtx[-6] > max(self.vtx[-4], self.vtx[-2])+pullback:
                        return True, pullback/PULLBACK_RATIO
            return False, 0.0
    
    def _is_trendline_shape(self, inverse: bool = False) -> Tuple[bool, float]:
        """
        Check if points form a valid trendline pattern
        ascending=True checks for bullish trendline
        """
        # -6\       -2 /     #
        #    \   -4 /\/      #
        #     \  /\/ -1      #
        #      \/ -3         #
        #      -5            #

        if not inverse:
            if self.bi_list[-1].dir == BI_DIR.DOWN:
                if self.vtx[-1] > self.vtx[-3] > self.vtx[-5]:
                    pullback = min(abs(self.vtx[-2]-self.vtx[-1]),abs(self.vtx[-4]-self.vtx[-3]))
                    #if self.vtx[-6] > max(self.vtx[-4], self.vtx[-2])+pullback:
                    return True, pullback/PULLBACK_RATIO
            return False, 0.0
        else:
            if self.bi_list[-1].dir == BI_DIR.UP:
                if self.vtx[-1] < self.vtx[-3] < self.vtx[-5]:
                    pullback = min(abs(self.vtx[-2]-self.vtx[-1]),abs(self.vtx[-4]-self.vtx[-3]))
                    #if self.vtx[-6] < min(self.vtx[-4], self.vtx[-2])-pullback:
                    return True, pullback/PULLBACK_RATIO
            return False, 0.0