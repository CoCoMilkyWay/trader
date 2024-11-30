import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import datetime
from typing import Dict, Iterable, List, Optional, Union
from collections import defaultdict, deque
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Chan.ChanConfig import CChanConfig
from Chan.Common.CEnum import KL_TYPE
from Chan.Common.ChanException import CChanException, ErrCode
from Chan.Common.CTime import CTime
from Chan.Common.func_util import check_kltype_order, kltype_lte_day
from Chan.KLine.KLine_List import CKLine_List
from Chan.KLine.KLine_Unit import CKLine_Unit


class CChan:
    """
    A class for managing and analyzing K-line (candlestick) data across multiple time levels.
    Supports hierarchical K-line analysis with parent-child relationships between different timeframes.
    """

    def __init__(
        self,
        code: str,
        begin_time: Optional[Union[str, datetime.date]] = None,
        end_time: Optional[Union[str, datetime.date]] = None,
        lv_list: Optional[List[KL_TYPE]] = None,
        config: Optional[CChanConfig] = None,
    ):
        """
        Initialize CChan with configuration for K-line analysis.

        Args:
            code: Stock/instrument code
            begin_time: Start time for analysis
            end_time: End time for analysis
            lv_list: List of K-line time levels (e.g., daily, hourly)
            config: Configuration object for analysis parameters
        """
        # Set default levels if none provided (Day and 60-minute)
        if lv_list is None:
            lv_list = [KL_TYPE.K_DAY, KL_TYPE.K_60M]
        # Ensure levels are ordered from high to low
        check_kltype_order(lv_list)

        # Initialize basic attributes
        self.code = code
        self.begin_time = str(begin_time) if isinstance(
            begin_time, datetime.date) else begin_time
        self.end_time = str(end_time) if isinstance(
            end_time, datetime.date) else end_time
        self.lv_list: List[KL_TYPE] = lv_list
        self.new_bi_start: bool = False  # Flag for new highest level K-line bi
        self.volume_profile_batch: List[Union[int, List[int]]]

        # Set configuration
        self.conf = config if config is not None else CChanConfig()

        # Initialize tracking variables
        self.kl_misalign_cnt = 0  # Counter for misaligned K-lines
        self.kl_inconsistent_detail = defaultdict(
            list)  # Track inconsistent K-line details
        self.g_kl_iter = defaultdict(list)  # K-line iterators for each level

        # Initialize data structures
        self.do_init()

    def do_init(self):
        """Initialize K-line data structures for each time level."""
        self.kl_datas: Dict[KL_TYPE, CKLine_List] = {
            level: CKLine_List(level, conf=self.conf)
            for level in self.lv_list
        }

    def trigger_load(self, klu_dict: Dict[KL_TYPE, List[CKLine_Unit]]):
        """
        Trigger the loading and processing of K-line data.

        Args:
            klu_dict: Dictionary mapping time levels to their K-line unit lists
        """
        # Initialize caches if not present
        if not hasattr(self, 'klu_cache'):
            self.klu_cache: List[Optional[CKLine_Unit]] = [
                None] * len(self.lv_list)
        if not hasattr(self, 'klu_last_t'):
            self.klu_last_t = [CTime(1980, 1, 1, 0, 0)] * len(self.lv_list)

        # Process each level
        for lv_idx, lv in enumerate(self.lv_list):
            if lv not in klu_dict:
                if lv_idx == 0:
                    raise CChanException(
                        f"Highest level {lv} has no data", ErrCode.NO_DATA)
                continue

            assert isinstance(klu_dict[lv], list)
            self.add_lv_iter(lv, iter(klu_dict[lv]))

        # Load all data through iterator
        for _ in self.load_iterator(lv_idx=0, parent_klu=None, step=False):
            pass
        for lv in self.lv_list:
            self.kl_datas[lv].try_add_virtual_bi()
            
    def add_lv_iter(self, lv_idx: Union[int, KL_TYPE], iter_obj: Iterable[CKLine_Unit]):
        """Add a new iterator for a specific level."""
        if isinstance(lv_idx, int):
            self.g_kl_iter[self.lv_list[lv_idx]].append(iter_obj)
        else:
            self.g_kl_iter[lv_idx].append(iter_obj)

    def load_iterator(self, lv_idx: int, parent_klu: Optional[CKLine_Unit], step: bool):
        """
        Main iterator for loading and processing K-line data.
        Handles hierarchical relationships between different time levels.

        Args:
            lv_idx: Index of current level being processed
            parent_klu: Parent K-line unit (if any)
            step: Whether to yield after each step
        """
        cur_lv = self.lv_list[lv_idx]
        pre_klu = None

        while True:
            # Get next K-line unit (from cache or iterator)
            if self.klu_cache[lv_idx]:
                kline_unit = self.klu_cache[lv_idx]
                assert kline_unit is not None
                self.klu_cache[lv_idx] = None
            else:
                try:
                    kline_unit = self.get_next_lv_klu(lv_idx)
                    self.try_set_klu_idx(lv_idx, kline_unit)

                    # Ensure time monotonicity
                    if not kline_unit.time > self.klu_last_t[lv_idx]:
                        raise CChanException(
                            f"K-line time error, current={
                                kline_unit.time}, last={self.klu_last_t[lv_idx]}",
                            ErrCode.KL_NOT_MONOTONOUS
                        )
                    self.klu_last_t[lv_idx] = kline_unit.time
                except StopIteration:
                    break

            # Check if we've exceeded parent time
            if parent_klu and kline_unit.time > parent_klu.time:
                self.klu_cache[lv_idx] = kline_unit
                break

            # Set relationships and process
            kline_unit.set_pre_klu(pre_klu)
            pre_klu = kline_unit
            self.add_new_kl(cur_lv, kline_unit)

            if parent_klu:
                self.set_klu_parent_relation(
                    parent_klu, kline_unit, cur_lv, lv_idx)

            # Process lower levels recursively
            if lv_idx != len(self.lv_list) - 1:
                for _ in self.load_iterator(lv_idx + 1, kline_unit, step):
                    pass
                self.check_kl_align(kline_unit, lv_idx)

            if lv_idx == 0 and step:
                yield self

    def get_next_lv_klu(self, lv_idx: Union[int, KL_TYPE]) -> CKLine_Unit:
        """Get next K-line unit from the appropriate iterator."""
        if isinstance(lv_idx, int):
            lv_idx = self.lv_list[lv_idx]

        if not self.g_kl_iter[lv_idx]:
            raise StopIteration

        try:
            return next(self.g_kl_iter[lv_idx][0])
        except StopIteration:
            self.g_kl_iter[lv_idx] = self.g_kl_iter[lv_idx][1:]
            if self.g_kl_iter[lv_idx]:
                return self.get_next_lv_klu(lv_idx)
            raise

    def set_klu_parent_relation(
        self,
        parent_klu: CKLine_Unit,
        kline_unit: CKLine_Unit,
        cur_lv: KL_TYPE,
        lv_idx: int
    ):
        """Set parent-child relationships between K-line units and perform consistency checks."""
        if (self.conf.kl_data_check and
            kltype_lte_day(cur_lv) and
                kltype_lte_day(self.lv_list[lv_idx-1])):
            self.check_kl_consitent(parent_klu, kline_unit)

        parent_klu.add_children(kline_unit)
        kline_unit.set_parent(parent_klu)

    def check_kl_align(self, kline_unit: CKLine_Unit, lv_idx: int):
        """Check if K-line units are properly aligned across levels."""
        if self.conf.kl_data_check and not kline_unit.sub_kl_list:
            self.kl_misalign_cnt += 1
            if self.conf.print_warning:
                print(f"[WARNING-{self.code}] No K-line found at {kline_unit.time} "
                      f"for level {self.lv_list[lv_idx+1]}!")
            if self.kl_misalign_cnt > 0:
                raise CChanException(
                    "No K-line found in sublevel!", ErrCode.KL_DATA_NOT_ALIGN)

    def check_kl_consitent(self, parent_klu: CKLine_Unit, sub_klu: CKLine_Unit):
        """Check time consistency between parent and child K-line units."""
        if (parent_klu.time.year != sub_klu.time.year or
            parent_klu.time.month != sub_klu.time.month or
                parent_klu.time.day != sub_klu.time.day):

            self.kl_inconsistent_detail[str(
                parent_klu.time)].append(sub_klu.time)
            if self.conf.print_warning:
                print(f"[WARNING-{self.code}] Parent time is {parent_klu.time}, "
                      f"but child time is {sub_klu.time}")
            if self.kl_inconsistent_detail:
                raise CChanException("Parent & child K-line times inconsistent!",
                                     ErrCode.KL_TIME_INCONSISTENT)

    def add_new_kl(self, cur_lv: KL_TYPE, kline_unit: CKLine_Unit):
        """Add a new K-line unit and check for new bi formation at highest level."""
        try:
            self.kl_datas[cur_lv].add_single_klu(kline_unit)

            # Check for new bi at highest level
            if cur_lv == self.lv_list[0]:
                self.new_bi_start = self.kl_datas[cur_lv].new_bi_start

        except Exception:
            if self.conf.print_err_time:
                print(
                    f"[ERROR-{self.code}] Error processing K-line at {kline_unit.time}!")
            raise

    def try_set_klu_idx(self, lv_idx: int, kline_unit: CKLine_Unit):
        """Set index for K-line unit if not already set."""
        if kline_unit.idx >= 0:
            return

        if not self[lv_idx]:
            kline_unit.set_idx(0)
        else:
            kline_unit.set_idx(self[lv_idx][-1][-1].idx + 1)

    def __getitem__(self, n: Union[KL_TYPE, int]) -> CKLine_List:
        """Get K-line list for a specific level."""
        if isinstance(n, KL_TYPE):
            return self.kl_datas[n]
        elif isinstance(n, int):
            return self.kl_datas[self.lv_list[n]]
        else:
            raise CChanException("Unsupported query type",
                                 ErrCode.COMMON_ERROR)


    def plot(self, save: bool = True):
        from Chan.Plot.PlotDriver import ChanPlotter
        
        fig = ChanPlotter().plot(
            klc_list = self.kl_datas[self.lv_list[0]]
        )
                
        # if save:
        #     plot_driver.save2img(mkdir(f'./outputs_bt/{self.code}.png'))
        # else:
        #     plot_driver.figure.show()

def mkdir(path_str: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    path = os.path.dirname(path_str)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path_str
