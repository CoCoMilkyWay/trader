# coding:utf-8
import sys
import time
import numpy as np
import pandas as pd

from datetime import date, datetime, timedelta
from typing import List, Dict, Any
from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant

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
        self.trader.start()
        if 0:
            while (r := self.trader.connect()) != 0:
                print(f'[API]: Trader Connecting... {r}')
                print(self.trader.connect())
                time.sleep(1)
                break
            while (r := self.trader.subscribe(self.account)) != 0:
                print(f'[API]: Subscription Channel Connecting... {r}')
                time.sleep(1)
                break
        print(f'[API]: {BLUE}QMT{RESET}: Server Connected')

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
    
    def get_bars(self, contract: Contract, days: int, bar_size: str = '1 min', exg_timezone: str = 'Asia/Shanghai'):
        """
        data need to be handled/saved in exchange timezone for better organization
        doesn't include end(today's) data, as it is incomplete
        doesn't include start data, as it is incomplete
        """
        print(f'[API]: {BLUE}IBKR{RESET}: Pulling {days} days {contract.localSymbol} futures data...')
        all_dfs = []
        end_time = datetime.now(pytz.timezone(exg_timezone)).replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(days=days)
        increment = 10
        incr_time = end_time
        for i in tqdm(range(((days//increment)+1)), desc=contract.localSymbol):
            df = util.df(self.ib.reqHistoricalData(
                contract,
                endDateTime=incr_time,
                durationStr=f'{min(increment,days)+1} D',  # IBKR data on 1st day is not complete (from trading session start, no pre-session data)
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1,
                timeout=0,  # wait forever
            ))
            assert df is not None, f"Failed to get bars for {contract} at end date {incr_time}"
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                df = df.set_index(df['datetime'].dt.strftime('%Y%m%d%H%M').astype(int))
                df = df.rename(columns={'': ''})[['open', 'high', 'low', 'close', 'volume']]
                all_dfs.append(df)
            incr_time -= timedelta(days=increment)
        all_df = pd.concat(all_dfs).drop_duplicates().sort_index()
        full_index = pd.date_range(start=start_time, end=end_time, freq='1min')
        full_index = full_index.strftime('%Y%m%d%H%M').astype('int64')
        columns = ['open', 'high', 'low', 'close', 'volume']
        full_df = pd.DataFrame(index=full_index, columns=columns, dtype=np.float16)
        full_df[columns] = all_df[columns].reindex(full_index)
        full_df['close'] = full_df['close'].ffill().bfill() # note: backward fill uses future data, but is mostly okay
        full_df['volume'] = full_df['volume'].fillna(0)
        missing_mask = full_df['open'].isna()
        full_df.loc[missing_mask, ['open', 'high', 'low']] = full_df.loc[missing_mask, 'close'].values[:, None].repeat(3, axis=1)
        full_df = full_df[full_df.index >= int(start_time.strftime('%Y%m%d%H%M'))]
        return full_df
    
    def subscribe(self, assets: List[str]):
        def on_subscribed_data(data):
            print(data)

        for asset in assets:
            xtdata.subscribe_quote(asset, period='tick', start_time='',
                                   end_time='', count=1, callback=on_subscribed_data)
        return


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
