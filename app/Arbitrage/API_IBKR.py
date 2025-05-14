# coding:utf-8
import pytz
import numpy as np
import pandas as pd

from tqdm import tqdm
from pprint import pprint
from datetime import date, datetime, timedelta
from ib_async import IB, util, Stock, Future, ContFuture, Option, Contract

from Arbitrage.Util import *

class IBKR:
    """
    Unfortunately IBKR has shitty servers:
    not only they lag
    they miss data occasionally
    """
    
    def __init__(self, gateway_ip: str, gateway_port: int, account_id: str = ''):
        util.patchAsyncio()

        self.gateway_ip = gateway_ip
        self.gateway_port = gateway_port
        self.account_id = account_id
        self.ib = IB()
        self.connect()
        self.loop = util.getLoop()
        # self.positions = self.ib.positions(self.account_id)
        # self.loop.run_forever()

    def connect(self):
        """Connect to IBKR Gateway"""
        self.ib.connect(self.gateway_ip, self.gateway_port, 1)  # int(time.time()))
        print(f'[API]: {BLUE}IBKR{RESET}: Client Gateway Connected')

    def disconnect(self):
        """Disconnect from IBKR Gateway"""
        self.ib.disconnect()
        print(f'[API]: {BLUE}IBKR{RESET}: Client Gateway Disconnected')

    def get_futures(self, symbol: str, type: str):
        """
        use continuous futures for simplicity
        """
        if type == 'Futures':
            fut = Future(symbol)
        elif type == 'ContFutures':
            fut = ContFuture(symbol)
            fut.includeExpired = True

        cds = self.ib.reqContractDetails(fut)
        assert cds, f"Failed to get contract details for {type}: {fut}"
        contracts = [cd.contract for cd in cds if cd.contract is not None]

        def parse_last_trade_date(contract):
            return datetime.strptime(contract.lastTradeDateOrContractMonth, "%Y%m%d")

        contracts.sort(key=parse_last_trade_date)
        return contracts

    def get_bars(self, contract: Contract, days: int, bar_size: str = '1 min', exg_timezone: str = 'Asia/Shanghai'):
        """
        data need to be handled/saved in exchange timezone for better organization
        doesn't include end(today's) data, as it is incomplete
        doesn't include start data, as it is incomplete
        """
        print(f'[Main]: Pulling {days} days {contract.localSymbol} futures data from IBKR...')
        all_dfs = []
        end_date = datetime.now(pytz.timezone(exg_timezone)).replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days)
        increment = 10
        incr_date = end_date
        for i in tqdm(range(((days//increment)+1)), desc=contract.localSymbol):
            df = util.df(self.ib.reqHistoricalData(
                contract,
                endDateTime=incr_date,
                durationStr=f'{min(increment,days)+1} D',  # IBKR data on 1st day is not complete (from trading session start, no pre-session data)
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1,
                timeout=0,  # wait forever
            ))
            assert df is not None, f"Failed to get bars for {contract} at end date {incr_date}"
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                df = df.set_index(df['datetime'].dt.strftime('%Y%m%d%H%M').astype(int))
                df = df.rename(columns={'': ''})[['open', 'high', 'low', 'close', 'volume']]
                all_dfs.append(df)

            incr_date -= timedelta(days=increment)

        full_df = pd.concat(all_dfs).drop_duplicates().sort_index()
        full_df = full_df[full_df.index >= int(start_date.strftime('%Y%m%d%H%M'))]
        return full_df

    # def get_bars_whole(self, contract: Contract, days: int, bar_size: str = '1 min', exg_timezone: str = 'Asia/Shanghai'):
    #     """
    #     data need to be handled/saved in exchange timezone for better organization
    #     doesn't include end(today's) data, as it is incomplete
    #     doesn't include start data, as it is incomplete
    #     """
    #     print(f'[Main]: Pulling {days} days {contract.symbol}({contract.secType}) data from IBKR...')
    #     end_date = datetime.now(pytz.timezone(exg_timezone)).replace(hour=0, minute=0, second=0, microsecond=0)
    #     start_date = end_date - timedelta(days=days)
    #     df = util.df(self.ib.reqHistoricalData(
    #         contract,
    #         endDateTime=None,
    #         durationStr=f'{days+1} D',  # IBKR data on 1st day is not complete (from trading session start, no pre-session data)
    #         barSizeSetting=bar_size,
    #         whatToShow='TRADES',
    #         useRTH=False,
    #         formatDate=1,
    #         timeout=0,  # wait forever
    #     ))
    #     assert df is not None, f"Failed to get bars for {contract} at end date {end_date}"
    #     if not df.empty:
    #         df['datetime'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    #         df = df.set_index(df['datetime'].dt.strftime('%Y%m%d%H%M%S').astype(int))
    #         df = df.rename(columns={'': ''})[['open', 'high', 'low', 'close', 'volume']]
    #     df = df.sort_index()
    #     df = df[int(end_date.strftime('%Y%m%d%H%M%S')) > df.index >= int(start_date.strftime('%Y%m%d%H%M%S'))]
    #     print(df)
    #     print(start_date)
    #     print(end_date)
    #     print(min(df.index))
    #     return df

    # # Add other methods following the same pattern
    # def get_stock_info(self, symbol):
    #     """Example of another sync wrapper method"""
    #     async def _get_stock_info_async():
    #         contract = Stock(symbol, 'SMART', 'USD')
    #         details = await self.ib.reqContractDetailsAsync(contract)
    #         # Process data...
    #         return details
    #
    #     return asyncio.run(_get_stock_info_async())

# # Define AAPL contract
# contract = Stock('AAPL', 'SMART', 'USD')
# # Request real-time market data
# ticker = ib.reqMktData(contract)
# # Give it a moment to stream data
# ib.sleep(2)
# # Print real-time price info
# print('Last price:', ticker.last)
# print('Bid:', ticker.bid)
# print('Ask:', ticker.ask)
# ib.disconnect()
