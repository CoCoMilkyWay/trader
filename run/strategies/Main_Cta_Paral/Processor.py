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

# Those Processors are largely insulated, try minimize data throughput
class SlaveProcessor:
    def __init__(self):
        self.chan_snapshot: Dict[str, CChan] = {}

    def process_slave_task(self, id, task, meta):
        print(id, task)
        # Use and update self.state as needed
        result = []  # Process the task and produce a result
        return result
    
class MasterProcessor:
    def __init__(self):
        self.state = {}  # Initialize any state variables here

    def process_master_task(self, id, task, meta):
        # Use and update self.state as needed
        pass  # Process the task