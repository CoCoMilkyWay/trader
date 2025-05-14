# coding:utf-8
import pytz
import numpy as np
import pandas as pd

from tqdm import tqdm
from pprint import pprint
from datetime import date, datetime, timedelta
from tvDatafeed import TvDatafeed, TvDatafeedLive, Interval

from Arbitrage.Util import *

class TV:
    """ Tradingview API"""

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.tv = TvDatafeed(self.username, self.password)
        # tvl = TvDatafeedLive()

    def get_cme_bars(self, symbol:str):
        
        # 'MNQM2025'
        df = self.tv.get_hist(
            symbol=f'CME_MINI:{symbol}',
            exchange='CME_MINI',
            interval=Interval.in_1_minute,
            n_bars=1000,
            extended_session=True
        )
        
        return df