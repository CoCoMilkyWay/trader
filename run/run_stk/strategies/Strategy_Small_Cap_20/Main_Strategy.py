import os
import sys
import time
import json
import torch
from tqdm import tqdm
from typing import List, Dict

from wtpy import SelContext
from wtpy import BaseSelStrategy

from config.cfg_stk import cfg_stk
from Util.UtilStk import time_diff_in_min

from strategies.Strategy_Small_Cap_20.TimeSeriesAnalysis_Core import TimeSeriesAnalysis
from strategies.Strategy_Small_Cap_20.CrossSectionAnalysis_Core import CrossSectionAnalysis
from strategies.Strategy_Small_Cap_20.Labels import NUM_LABELS

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


def stdio(str):
    print(str)
    return str


class Main_Strategy(BaseSelStrategy):
    def __init__(self, name: str, codes: List[str], period: str):
        BaseSelStrategy.__init__(self, name)

        self.__period__ = period
        self.__codes__ = codes

        # stats
        self.inited = False
        self.barnum = 0

        # code_info
        self.code_info: Dict[str, Dict] = {}
        for idx, code in enumerate(codes):
            self.code_info[code] = {'idx': idx}

        start = cfg_stk.start
        end = cfg_stk.end
        N_timestamps = int(time_diff_in_min(start, end) * cfg_stk.max_trade_session_ratio)
        N_TS_features = 14
        N_CS_features = 0
        N_labels = 1
        N_columns = N_TS_features + N_CS_features + N_labels
        N_codes = len(codes)
        print(f"Initializing Pytorch Tensor: (timestamp({N_timestamps}), feature({N_TS_features}+{N_CS_features}) + label({N_labels}), codes({N_codes}))")
        print(f"Memory reserving: {(N_timestamps * N_columns * N_codes)*2/(1024**2):.2f} MB")
        self.shared_tensor = torch.zeros((N_timestamps, N_columns, N_codes), dtype=torch.float16).share_memory_()

        # cross-section data
        self.cs_signal = None
        self.cs_value = None
        
        # shared data
        self.new_date = True
        self.date = None
        self.pos = None
        self.equity = cfg_stk.capital
        self.holdings = []

        # TS core (price/volume/fundamentals)
        self.timeseries_analysis: Dict[str, TimeSeriesAnalysis] = {}

        # CS core (ranks)
        self.crosssection_analysis: CrossSectionAnalysis = CrossSectionAnalysis(code_info=self.code_info, shared_tensor=self.shared_tensor)

        for code in self.__codes__:
            self.timeseries_analysis[code] = TimeSeriesAnalysis(code=code, code_idx=self.code_info[code]['idx'], shared_tensor=self.shared_tensor, plot=False)

    def on_init(self, context: SelContext):
        self.context = context  # export environment
        print('Preparing Bars in DDR...')
        self.pbar = tqdm(total=len(self.__codes__))

        for idx, code in enumerate(self.__codes__):
            context.stra_prepare_bars(code, cfg_stk.wt_period_l, 1)
            self.pbar.update(1)
            self.pbar.set_description(f'Init: {code}', True)

        dir = os.path.dirname(os.path.abspath(__file__))
        self.daily_holdings = json.loads(open(f"{dir}/data/daily_holdings.json", encoding='utf-8').read())

        self.start_time = time.time()
        return

    def on_tick(self, context: SelContext, code: str, newTick: dict):
        print(f'on_tick: {code}')
        return

    def on_bar(self, context: SelContext, code: str, period: str, newBar: dict):
        # print(f'on_bar: {code}')

        idx = code.split('.')[2]
        curPos = context.stra_get_position(code)
        curPrice = context.stra_get_price(code)
        # df_bars = context.stra_get_bars(code, 'd1', 1)

        # if self.holdings!={}:
        #     print(code, idx, curPos, curPrice, self.equity, idx in self.holdings)
        #     print(self.holdings)

        if idx in self.holdings:
            target_pos = int(self.equity/20/curPrice)
            context.stra_set_position(code, target_pos, 'enterlong')
            # print(f'做多 {code}: {target_pos} @ {curPrice:.2f}')
            return
        else:
            if curPos != 0:
                target_pos = 0
                context.stra_set_position(code, target_pos, 'clearlong')
                # print(f'平多 {code}: {curPrice:.2f}')
                return

        # # multi-level k bar generation
        # TS = self.timeseries_analysis[code]
        # CS = self.crosssection_analysis
        # TS.analyze(newBar['open'], newBar['high'], newBar['low'],
        #            newBar['close'], newBar['vol'], newBar['time'])
        # CS.prepare()  # prepare TS data for later CS analysis

        # # indicator guard (prepare and align)
        # if not self.inited:
        #     self.barnum += 1
        #     if self.barnum > 1*24*60: # need 1 day(s) of 1M data
        #         self.inited = True
        #     return

        # strategy
        # self.ST_Train(context, code)
        return

    def on_calculate(self, context: SelContext):
        date = context.get_date()
        self.new_date = (self.date != date)
        if self.new_date:
            self.date = date
            date_str = str(date)
            date_index = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            self.holdings = self.daily_holdings.get(date_index, [])
            self.equity = context.stra_get_fund_data(0) + cfg_stk.capital
            self.pos = context.stra_get_all_position()
            print(f"on_calculate: {date}-{self.equity}============================================")
            
            for key in self.pos.keys():
                if self.pos[key] != 0:
                    print(f"持仓: {key} {self.pos[key]} @ {context.stra_get_price(key):.2f}")

    def on_backtest_end(self, context: SelContext):
        self.elapsed_time = time.time() - self.start_time
        print(f'main BT loop time elapsed: {self.elapsed_time:2f}s')

        # # feature distribution analysis
        # if cfg_stk.stat and self.__id__ == 0:
        #     from Util.CheckDist import CheckDist
        #     import pandas as pd
        #     df = pd.DataFrame(
        #         self.shared_tensor[:, :, 0].to(torch.float32).numpy())
        #     df.columns = pd.Index(self.feature_names + self.label_names)
        #     CheckDist(df, [self.feature_types])
        return
