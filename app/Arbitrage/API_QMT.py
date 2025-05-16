# coding:utf-8
import sys
import time
import pytz
import numpy as np
import pandas as pd

from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant
from tqdm import tqdm
from typing import List, Dict, Any
from datetime import date, datetime, timedelta

from pprint import pprint

from Arbitrage.Util import *


class QMT:
    def __init__(self, trader_path: str, account_id: str):
        self.trader_path = trader_path
        self.account_id = account_id
        self.session_id = int(time.time())  # per-strategy basis
        self.xt = xtdata
        self.xt.enable_hello = False

        self.trader = XtQuantTrader(self.trader_path, self.session_id)
        # exclusive query thread
        self.trader.set_relaxed_response_order_enabled(False)

        TYPE = 'STOCK'
        print(f'[API]: {BLUE}QMT{RESET}: Registering {RED}{TYPE}{RESET} account {RED}{account_id}{RESET}')
        # 股票STOCK 信用CREDIT 期货FUTURE
        self.account = StockAccount(account_id, TYPE)

        print(f'[API]: {BLUE}QMT{RESET}: Registering trading callback functions')
        self.callback = QMT_trader_callback()  # trader callback
        self.trader.register_callback(self.callback)

        self.connect()

    def connect(self):
        """
        券商QMT 需要打开miniqmt界面，不能用API指定服务器/登录账户
        权限: 普通版
        https://xuntou.net/#/productvip?id=
        """
        self.trader.connect()
        print(f'[API]: {BLUE}QMT{RESET}: Server Connected')
            
        # if 0:
        #     while (r := self.trader.connect()) != 0:
        #         print(f'[API]: Trader Connecting... {r}')
        #         print(self.trader.connect())
        #         time.sleep(1)
        #         break
        #     while (r := self.trader.subscribe(self.account)) != 0:
        #         print(f'[API]: Subscription Channel Connecting... {r}')
        #         time.sleep(1)
        #         break
        

    def disconnect(self):
        self.xt.disconnect()
        print(f'[API]: {BLUE}QMT{RESET}: Client Disconnected')


    def run(self):
        """Run the trading system"""
        if self.trader:
            self.trader.run_forever()

    def get_sectors(self) -> List[str]:
        """
        +------------------------+-----------------------------------------------+
        | TYPE                   | DESCRIPTION                                   |
        +------------------------+-----------------------------------------------+
        | EXCHANGE TYPES         | Major exchanges, market boards, cross-border  |
        |                        | mechanisms (上期所, 创业板, 沪港通)             |
        +------------------------+-----------------------------------------------+
        | INDEX CATEGORIES       | Market indices, weighted indices, sector      |
        |                        | indices (沪深指数, 板块加权指数)                |
        +------------------------+-----------------------------------------------+
        | PRODUCT TYPES          | Equities, fixed income, derivatives, funds    |
        |                        | (股票, 债券, 期权, ETF)                        |
        +------------------------+-----------------------------------------------+
        | INDUSTRY CLASSES       | Primary, secondary, tertiary industry sectors |
        |                        | (一级行业, 二级行业, 三级行业)                  |
        +------------------------+-----------------------------------------------+
        | THEMATIC GROUPS        | Tech themes, policy initiatives, emerging     |
        |                        | industries (AI, 国企改革, 新能源)              |
        +------------------------+-----------------------------------------------+
        | GEOGRAPHIC REGIONS     | Provincial markets, economic zones, regional  |
        |                        | integration (上海, 自贸区, 长三角)             |
        +------------------------+-----------------------------------------------+
        | INVESTMENT STYLES      | Value, growth, dividend, market cap based     |
        |                        | (价值, 成长, 红利, 大中小盘)                    |
        +------------------------+-----------------------------------------------+
        """
        return xtdata.get_sector_list()

    def get_bars(self, asset: str, days:int, period: str, exg_timezone: str = 'Asia/Shanghai'):
        """
        data need to be handled/saved in exchange timezone for better organization
        """
        print(f'[API]: {BLUE}QMT{RESET}: Pulling {days} days {asset} ETF data...')

        end_time = datetime.now(pytz.timezone(exg_timezone)).replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(days=days)
        s = start_time.strftime('%Y%m%d')
        e = end_time.strftime('%Y%m%d')
        
        self.xt.download_history_data(asset, period, start_time=s, incrementally=True)

        all_df:pd.DataFrame = xtdata.get_market_data_ex([],[asset],period=period,start_time=s)[asset]
        # IBKR and other American brokers use open time (we use this)
        # XT and other Chinese brokers use close time
        all_df.index = (pd.to_datetime(all_df.index).tz_localize(exg_timezone)-timedelta(minutes=1)).strftime('%Y%m%d%H%M').astype('int64')
        
        full_index = pd.date_range(start=start_time, end=end_time, freq='1min')
        full_index = full_index.strftime('%Y%m%d%H%M').astype('int64')
        columns = ['open', 'high', 'low', 'close', 'volume']
        full_df = pd.DataFrame(index=full_index, columns=columns, dtype=np.float16)
        full_df[columns] = all_df[columns].reindex(full_index)
        full_df['close'] = full_df['close'].ffill().bfill() # note: backward fill uses future data, but is mostly okay
        full_df['volume'] = full_df['volume'].fillna(0)
        missing_mask = full_df['open'].isna()
        full_df.loc[missing_mask, ['open', 'high', 'low']] = full_df.loc[missing_mask, 'close'].values[:, None].repeat(3, axis=1)
        full_df = full_df.loc[full_df.index >= int(start_time.strftime('%Y%m%d%H%M'))]
        return True, full_df

    def subscribe(self, assets: List[str]):
        def on_subscribed_data(data):
            print(data)

        for asset in assets:
            xtdata.subscribe_quote(asset, period='tick', start_time='',
                                   end_time='', count=1, callback=on_subscribed_data)
        return
# def subscribe_QMT_etf(self, filename='full-tick.txt'):
#     # Define callback for subscription data
#     def subscribed_data_callback(data):
#         now = datetime.now()
#         print(now, ': ', sys.getsizeof(data))
#         self.print_to_file(data, filename)
#
#         # Uncomment to enable trading logic
#         # for stock in data:
#         #     if stock not in self.hsa:
#         #         continue
#         #     cuurent_price = data[stock][0]['lastPrice']
#         #     pre_price = data[stock][0]['lastClose']
#         #     ratio = cuurent_price / pre_price - 1 if pre_price > 0 else 0
#         #     if ratio > 0.09 and stock not in self.bought_list:
#         #         print(f"{now} Latest price Buy {stock} 200 shares")
#         #         # async_seq = self.trader.order_stock_async(self.account, stock,
#         #         #                                          xtconstant.STOCK_BUY, 1,
#         #         #                                          xtconstant.LATEST_PRICE, -1,
#         #         #                                          'strategy_name', stock)
#         #         self.bought_list.append(stock)
#
#     # Subscribe to whole market data
#     xtdata.subscribe_whole_quote(
#         ["SH", "SZ"], callback=subscribed_data_callback)
#     print('[API]: Data Subscribed, DataFeed Start')

class QMT_trader_callback(XtQuantTraderCallback):
    """Trading callback implementation handling various trading events"""

    def on_disconnected(self):
        print(datetime.now(), 'Connection disconnected callback')

    def on_stock_order(self, order):
        print(datetime.now(), 'Order callback', order.order_remark)

    def on_stock_trade(self, trade):
        print(datetime.now(),
              'Trade execution callback', trade.order_remark)

    def on_order_error(self, order_error):
        print(
            f"Order error callback {order_error.order_remark} {order_error.error_msg}")

    def on_cancel_error(self, cancel_error):
        print(datetime.now(), sys._getframe().f_code.co_name)

    def on_order_stock_async_response(self, response):
        print(f"Async order callback {response.order_remark}")

    def on_cancel_order_stock_async_response(self, response):
        print(datetime.now(), sys._getframe().f_code.co_name)

    def on_account_status(self, status):
        print(datetime.now(), sys._getframe().f_code.co_name)
