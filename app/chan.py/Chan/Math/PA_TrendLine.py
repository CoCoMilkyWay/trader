import copy
from dataclasses import dataclass
from math import sqrt

from Chan.Common.CEnum import BI_DIR, TREND_LINE_SIDE

@dataclass
class Point:
    x: int
    y: float

    def cal_slope(self, p):
        return (self.y-p.y)/(self.x-p.x) if self.x != p.x else float("inf")

@dataclass
class Line:
    p: Point
    slope: float

    def cal_dis(self, p):
        if self.slope == float('inf') or self.slope == -float('inf'):
            return float('nan')
        else:
            return abs(self.slope*p.x - p.y + self.p.y - self.slope*self.p.x) / sqrt(self.slope**2 + 1)

class PA_TrendLine:
    # a trendline, if not refreshed(touched but not broke), will lost effect after
    #   1. breakthrough from one side, role of support/resistance could swap(after touch from the otherside)
    #   2. followed by breakthrough from the other side
    
    # for rising seg:
    #   support is more important
    #   support = begin of all in-trend bi = end of all out-of-trend bi
    # for falling seg:
    #   resistance is more important
    #   resistance = begin of all in-trend bi = end of all out-of-trend bi
    
    # 1. consider all possible convex interpolation of one side of vertexes (all points within 1 side of a trend-line)
    # 2. find trend-line with max amplitude slope, while having the same direction as newest bi
    # the concept is similar to velocity line

    def __init__(self, lst, side=TREND_LINE_SIDE.ALONG):
        self.line = None
        self.side = side
        self.cal(lst)

    def cal(self, lst): # lst = bi_list, called every time updating new bi
        bench = float('inf')
        if self.side == TREND_LINE_SIDE.ALONG:
            # p1: important side (establish trend)
            # p2: less important side
            all_p1 = [Point(bi.get_begin_klu().idx, bi.get_begin_val()) for bi in lst[-1::-2]]
            # all_p2 = [Point(bi.get_end_klu().idx, bi.get_end_val()) for bi in lst[-1::-2]]
        else:
            all_p1 = [Point(bi.get_end_klu().idx, bi.get_end_val()) for bi in lst[-1::-2]]
            # all_p2 = [Point(bi.get_begin_klu().idx, bi.get_begin_val()) for bi in lst[-1::-2]]

        c_p1 = copy.copy(all_p1)
        # c_p2 = copy.copy(all_p2)
        while True:
            line, idx = cal_tl(c_p1, lst[-1].dir, self.side) # c_p1: end to start
            dis = sum(line.cal_dis(p) for p in all_p1)
            if dis < bench:
                bench = dis
                self.line = line
            c_p1 = c_p1[idx:]
            if len(c_p1) == 1:
                break

def init_peak_slope(_dir, side):
    if side == TREND_LINE_SIDE.ALONG:
        return 0
    elif _dir == BI_DIR.UP:
        return float("inf")
    else:
        return -float("inf")

def cal_tl(c_p, _dir, side):
    # consider all point from side of interest within a seg:
    # from end of seg -> first point of seg
    # return trend-line from start of the seg to out-most vertexes
    # for a DOWN seg, the first and last bi will both be down
    p = c_p[0]
    peak_slope = init_peak_slope(_dir, side)
    idx = 1
    for point_idx, p2 in enumerate(c_p[1:]): # end to start
        slope = p.cal_slope(p2)
        if (_dir == BI_DIR.UP and slope < 0) or (_dir == BI_DIR.DOWN and slope > 0):
            # no matter where the segment goes, if newest bi is going up, try to find trend-line with max slope,
            # also ignore any trend-line with negative slope
            continue
        if side == TREND_LINE_SIDE.ALONG:
            if (_dir == BI_DIR.UP and slope > peak_slope) or (_dir == BI_DIR.DOWN and slope < peak_slope):
                peak_slope = slope
                idx = point_idx+1
        else:
            if (_dir == BI_DIR.UP and slope < peak_slope) or (_dir == BI_DIR.DOWN and slope > peak_slope):
                peak_slope = slope
                idx = point_idx+1
    return Line(p, peak_slope), idx
