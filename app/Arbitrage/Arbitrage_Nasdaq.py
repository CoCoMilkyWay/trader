# coding:utf-8
import re
import os
import sys
import json
import pytz
import aiohttp
import asyncio
import numpy as np
import pandas as pd
import exchange_calendars as ecals

from tqdm import tqdm
from typing import List, Dict, Any
from datetime import date, datetime, timedelta

from pprint import pprint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Arbitrage.Util import *
from Arbitrage.API_QMT import QMT
from Arbitrage.API_IBKR import IBKR
from Arbitrage.API_TV import TV

pd.set_option('display.max_rows', None)

'''
每日收盘：
1. 确认NAV(per share), net asset value: （基金真实仓位 + 现金头寸 + 应收款 - 应付款）/基金总份额
2. 确认基础参数:
Creation Unit（U）：固定常数（如100万份），由基金合同规定。
标的指数成分及权重：从指数提供商获取最新成分股权重
管理费率/托管费率：用于预提费用计算
3. 确认第二天PCF（Portfolio Composition File）篮子
申赎单位目标市值 V = NAVperCU = NAV(收盘后) * U ~= IOPV(收盘时) * U
根据权重得到个股份额并向下取整
确认估算现金差额（EstimateCashComponent）
= 初始现金缺口(V - 新权重的收盘总价)（和真实仓位成比例预扣现金）
+ 当日除息股票的股息（分红）- 每日管理费（自然反映到估值中，不需要申购者预扣，这样所有人都会承担管理费）
+ 其他费用（交易费用、税费，外汇成本，企业行为调整（如配股、拆股）等）（申购者承担，数额较小）
4. 第二天交易时段实时计算IOPV（indicative optimized portfolio value）
=（当日pcf篮子成分股实时价值 + 当日pcf估算现金差额）/ 创赎单位份额

一级市场交易商（基金公司），可以在盘中/盘后NAV预估前按需调仓

众所周知, 跨境ETF的IOPV净值因为不能反映实时的个股夜盘/期货的价格, 在场内交易时段可能不会更新
为了获得真正的实时IOPV, 我们需要自己实时计算

纽约时间(UTC-4): 对于纳斯达克股票现货, A股ETF的NAV计算于常规白盘结束时(4:00pm)
    注意4:00pm~4:15pm 为盘后撮合，大量机构在盘尾提交的MOC/LOC/IO类型订单必须在4:00pm后才可以开始撮合
    交易所以“最大成交量 + 最小不平衡量”作为撮合策略，实现Closing Cross
    这时的主板实时动态挂单与预估撮合价可以被专业数据订阅者看到，从而实现错位成交套利
    撮合时间一般来说很快，几秒钟就可以完成
    不需要等到4:15pm，实际上在4:00pm:0xs时，价格已经脱钩完成，定价权由正常交易的期货/暗池暂时接管

芝加哥时间(UTC-5): 对于CME指数期货
上海时间(UTC+8): 对于A股股票/ETF现货
'''

PCF_UPDATE_DATE = int(datetime.today().strftime('%Y%m%d')) - 0  # '20250101'

HK_ETF = []  # 70+
US_ETF = []  # 20+
JP_ETF = []  # 5
OTHER_ETF = []  # German, France, Korea, SASEN, Saudi Arabia

# codeID, exchangeID, pcfID
nasdaq100 = [
    ['159941', 'SZ', '   '],  # 纳指ETF
    ['513100', 'SH', '044'],  # 纳指ETF
    ['159632', 'SZ', '   '],  # 纳斯达克ETF
    ['159509', 'SZ', '   '],  # 纳指科技ETF
    ['159501', 'SZ', '   '],  # 纳斯达克指数ETF
    ['159513', 'SZ', '   '],  # 纳斯达克100指数ETF
    ['513300', 'SH', '286'],  # 纳斯达克ETF
    ['159659', 'SZ', '   '],  # 纳斯达克100ETF
    ['159696', 'SZ', '   '],  # 纳指ETF易方达
    ['159660', 'SZ', '   '],  # 纳指100ETF
    ['513110', 'SH', '551'],  # 纳斯达克100ETF
    ['513390', 'SH', '581'],  # 纳指100ETF
    ['513870', 'SH', '635'],  # 纳指ETF富国
]
hangsengtech = [
    ['513180', 'SH', '423'],  # 恒生科技指数ETF
    ['513130', 'SH', '424'],  # 恒生科技ETF
    ['513010', 'SH', '426'],  # 恒生科技ETF易方达
    ['159740', 'SZ', '   '],  # 恒生科技ETF
    ['513380', 'SH', '520'],  # 恒生科技ETF龙头
    ['513580', 'SH', '425'],  # 恒生科技ETF指数基金
    ['159742', 'SZ', '   '],  # 恒生科技指数ETF
    ['513260', 'SH', '512'],  # 恒生科技ETF基金
    ['159741', 'SZ', '   '],  # 恒生科技ETF基金
    ['513890', 'SH', '490'],  # 恒生科技HKETF
]

# Proxy
proxy_ip = '127.0.0.1'
proxy_port = 20000

# QMT
qmt_trader_path = r'D:\QMT\userdata_mini'
qmt_account_id = '8881848660'

# IBKR
ibkr_gateway_ip = '127.0.0.1'
ibkr_gateway_port = 7497
ibkr_account_id = 'DU12939891'

# TV
tv_username = 'Coco_MilkyWay'
tv_password = 'Wang475869123'

class Main:
    """Main trading application class"""

    def __init__(self):
        """
        Make sure QMT is running
        """
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.qmt = QMT(qmt_trader_path, qmt_account_id)
        self.ibkr = IBKR(ibkr_gateway_ip, ibkr_gateway_port)
        self.tv = TV(tv_username, tv_password)

        # self.sector_names = self.qmt.get_sectors()

        # self.download()

        self.daily_routine_nq100()

    def daily_routine_nq100(self):
        # Prepare Assets info =====================================================================
        self.qdii_etf_info = self.get_etf_info(['ETF跨境型'])  # 'ETF行业指数')
        self.nq_pcf_info = {}
        print(f'[Main]: Filtering Nasdaq100 from QDII ETFs...')
        # NASDAQ100
        SSE_NQ_ETFs = []
        SZSE_NQ_ETFs = []
        for etf in nasdaq100:
            # print(f"{self.qdii_etf_info[int(etf[0])]['InstrumentName']}")
            if etf[1] == 'SH':
                SSE_NQ_ETFs.append(etf)
            elif etf[1] == 'SZ':
                SZSE_NQ_ETFs.append(etf)
        print(f'[Main]:\n',
              f'       SSE  ETFs: {[self.qdii_etf_info[int(etf[0])]['InstrumentName'] for etf in SSE_NQ_ETFs]}\n',
              f'       SZSE ETFs: {[self.qdii_etf_info[int(etf[0])]['InstrumentName'] for etf in SZSE_NQ_ETFs]}\n',
              )
        self.nq_pcf_info = self.get_etf_pcf_info(SSE_NQ_ETFs, 'SH', self.nq_pcf_info)
        self.nq_pcf_info = self.get_etf_pcf_info(SZSE_NQ_ETFs, 'SZ', self.nq_pcf_info)
        self.save_to_json_compact('info_etf_qdii.json', self.qdii_etf_info)
        self.save_to_json_compact('info_pcf_nq.json', self.nq_pcf_info)
        self.nq_companies = self.get_etf_companies(self.nq_pcf_info)

        symbol = 'MNQ'
        self.mnq_futures = self.ibkr.get_futures(symbol, 'Futures')  # for simplicity
        for fut in self.mnq_futures:
            assert fut.symbol == symbol, f"non-{symbol} futures found"
        pprint(self.mnq_futures)

        # self.cont_fut: Contract = self.mnq_futures[0]
        self.pri_fut = self.mnq_futures[0]
        self.sec_fut = self.mnq_futures[1]
        print(f'[Main]: Primary   MNQ contract({self.pri_fut.exchange}:{self.pri_fut.currency}): {RED}{self.pri_fut.symbol}: {self.pri_fut.localSymbol}{RESET}: {self.pri_fut.lastTradeDateOrContractMonth}')
        print(f'[Main]: Secondary MNQ contract({self.sec_fut.exchange}:{self.sec_fut.currency}): {RED}{self.sec_fut.symbol}: {self.sec_fut.localSymbol}{RESET}: {self.sec_fut.lastTradeDateOrContractMonth}')

        # prepare recent history ===================================================================
        duration = 6  # months
        fut_tz = self.ibkr.ib.reqContractDetails(self.pri_fut)[0].timeZoneId
        end_date = datetime.now(pytz.timezone(fut_tz)).date()
        start_date = end_date - timedelta(days=30*duration)
        trade_days_cme = self.get_tradedays("CMES", start_date, end_date, type='futures')
        trade_days_sse = self.get_tradedays("XSHG", start_date, end_date, type='spot')
        print(f'[Main]: {YELLOW}CME     {RESET} trade days in last {duration} month: {len(trade_days_cme)}, sessions(US/Central):    17:00 - 15:59')
        print(f'[Main]: {YELLOW}SSE/SZSE{RESET} trade days in last {duration} month: {len(trade_days_sse)}, sessions(Asia/Shanghai): 09:30 - 14:57')

        for fut in self.mnq_futures:
            print(fut)
            try: 
                # 1. failed to get data 
                # 2. failed to get complete data
                fut_details = self.ibkr.ib.reqContractDetails(fut)
                assert len(fut_details) == 1, f"Failed to qualify contract {fut}"
                fut_tz = fut_details[0].timeZoneId
                # print(fut_details)

                missing_days_fut = self.check_bars(fut.localSymbol, trade_days_cme, fut_tz)
                print(missing_days_fut)
                if missing_days_fut > 0:
                    bars = self.ibkr.get_bars(fut, days=missing_days_fut, bar_size='1 min', exg_timezone=fut_tz)
                    self.store_bars(bars, fut.localSymbol, trade_days_cme)
            except Exception as e:
                print(e)
        # pprint(bars, compact=True)
        # self.qmt.subscribe_QMT_etf('full-tick.txt')

        # self.qmt.subscribe([])

        # cur_prices = self.get_robinhood_price(self.nasdaq100_assets)
        # print(cur_prices)

        # Run the system
        print('[API]: Running...')
        # xtdata.disconnect()
        self.ibkr.disconnect()
        # self.run()

    def get_etf_info(self, sector_list: List[str]) -> Dict[int, Dict[str, Any]]:
        etf_info: Dict[int, Dict[str, Any]] = {}
        print(f'[Main]: Pulling {sector_list} info from xtdata...')
        for sector in sector_list:
            codes: List[str] = self.qmt.xt.get_stock_list_in_sector(sector)
            for code in codes:
                info: Dict = self.qmt.xt.get_instrument_detail(code)  # type: ignore
                if info:
                    ID = int(info.get('InstrumentID'))  # type: ignore
                    etf_info[ID] = info
                    FloatVolume = float(
                        info.get('FloatVolume'))  # type: ignore
                    SettlementPrice = float(
                        info.get('SettlementPrice'))  # type: ignore
                    etf_info[ID]['MarketCap'] = int(
                        FloatVolume * SettlementPrice / 10**6)  # in Million
                else:
                    print(f"skipping {code}")
        return etf_info

        # for code in self.etf_info.keys():
        #     info = self.etf_info[code]
        #     if '纳' in info['InstrumentName']:
        #         nasdaq100.append((code, info['InstrumentName'], info['MarketCap']))
        #     if '恒生科' in info['InstrumentName']:
        #         hangsengtech.append((code, info['InstrumentName'], info['MarketCap']))
        #
        # self.nasdaq100 = sorted(nasdaq100, key=lambda x: x[2], reverse=True)
        # self.hangseng = sorted(hangsengtech, key=lambda x: x[2], reverse=True)
        # print(self.nasdaq100)
        # print(self.hangseng)

    def get_etf_pcf_info(self, ETFs: List[str], exchangeID: str, pcf_info: Dict):
        from Paser_SSE import Paser_SSE
        from Paser_SZSE import Paser_SZSE

        async def get_sse_pcf(session, ETF):
            codeID = ETF[0]
            exchangeID = ETF[1]
            pcfID = ETF[2]

            url = f"https://query.sse.com.cn/etfDownload/downloadETF2Bulletin.do?etfType={pcfID}"
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text(encoding='gbk')
                    result = Paser_SSE(text)
                else:
                    assert False, f"Failed to fetch data for {pcfID}: {response.status}"

                check_integrity(result, codeID)

                pcf_info[codeID] = result
                return

        async def get_szse_pcf(session, ETF):
            codeID = ETF[0]
            exchangeID = ETF[1]
            pcfID = ETF[2]

            url = f"https://reportdocs.static.szse.cn/files/text/etf/ETF{codeID}{PCF_UPDATE_DATE}.txt"
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text(encoding='gbk')
                    result = Paser_SZSE(text)
                else:
                    assert False, f"Failed to fetch data for {pcfID}: {response.status}"

                check_integrity(result, codeID)

                pcf_info[codeID] = result
                return

        async def fetch_all_async(ETFs, exchangeID):
            async with aiohttp.ClientSession() as session:
                if exchangeID == 'SH':
                    print('[Main]: pulling ETF PCF from SSE websites...')
                    tasks = [get_sse_pcf(session, ETF) for ETF in ETFs]
                elif exchangeID == 'SZ':
                    print('[Main]: pulling ETF PCF from SZSE websites...')
                    tasks = [get_szse_pcf(session, ETF) for ETF in ETFs]
                return await asyncio.gather(*tasks)  # type: ignore

        def check_integrity(result, codeID):
            assert int(codeID) == result['FundID'], f"ETF code mismatch: {codeID} != {result['FundID']}"
            assert int(PCF_UPDATE_DATE) == result['TradingDay'], f"Date mismatch: {datetime.today().strftime('%Y%m%d')} != {result['TradingDay']}"
            assert len(result['pcf']) == int(result['StockNum']), f"Expected {result['StockNum']} stocks, but got {len(result['pcf'])}."

        asyncio.run(fetch_all_async(ETFs, exchangeID))
        return pcf_info

    def get_etf_companies(self, pcf_info: Dict) -> List[str]:
        assets = []
        for pcf in pcf_info.values():
            for asset in pcf['pcf']:
                if asset[0] not in assets:
                    assets.append(asset[0])
        return assets

    def get_tradedays(self, exg: str, start_date: date, end_date: date, type: str) -> List[str]:
        """
        Note that a session may start on the day prior to the session label or
        end on the day following the session label. Such behaviour is common
        for calendars that represent futures exchange
        """
        def fill_gaps_with_previous_day(dates):
            # Convert strings to datetime objects
            date_objs = [datetime.strptime(d, '%Y%m%d') for d in dates]
            result = [date_objs[0]]

            for i in range(1, len(date_objs)):
                prev = date_objs[i - 1]
                curr = date_objs[i]
                if (curr - prev).days > 1:
                    # Insert the day before the current date
                    result.append(curr - timedelta(days=1))
                result.append(curr)

            # Convert back to string format
            return [d.strftime('%Y%m%d') for d in result]

        trade_days = ecals.get_calendar(exg).sessions_in_range(
            pd.Timestamp(start_date),
            pd.Timestamp(end_date)
        ).strftime('%Y%m%d').tolist()

        if type == 'spot':
            return trade_days
        elif type == 'futures':
            return fill_gaps_with_previous_day(trade_days)
        else:
            assert False, f"Invalid type: {type}"

    def check_bars(self, symbol: str, tradedays: list[str], tz:str):
        missing = tradedays.copy()
        dir_path = os.path.join(self.dir, f"history/{symbol}")
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            for file in files:
                if file.startswith("VALID_"):
                    _, valid_start, valid_end = file.split('_')
                    assert missing[0] < valid_end, f"more bars needed for continuous history data"
                    missing = [m for m in missing if m > valid_end]
                    break
            for file in files:
                valid = False
                if file.endswith('.parquet'):
                    date = file.split('_')[0]
                    if date <= valid_end:
                        valid = True
                    elif date in missing:
                        missing.remove(date)
                        valid = True
                elif file.startswith("VALID_"):
                    valid = True
                if not valid:
                    filepath = os.path.join(dir_path, file)
                    os.remove(filepath)
        else:
            os.makedirs(dir_path, exist_ok=True)

        missing = [int(m) for m in missing]
        start = datetime.strptime(str(min(missing)), '%Y%m%d').date()
        print(start, missing)
        today = datetime.now(pytz.timezone(tz)).date()
        delta_days = (today - start).days
        return delta_days

    def store_bars(self, df: pd.DataFrame, symbol: str, tradedays: list[str]):
        """
        user make sure the first/last date has complete data
        """
        updated_dates = []
        df["date"] = df.index.astype(str).str[:8]  # e.g., "20250505"
        dir_path = os.path.join(self.dir, f"history/{symbol}")
        dates = []
        for file in os.listdir(dir_path):
            if file.endswith('.parquet'):
                date = file.split('_')[0]
                dates.append(date)
                
        for date, group in df.groupby("date"):
            if date not in dates:
                # data integrity checks (data may not be reliable)
                day_session, num_minutes = get_cme_day_session(tradedays, tradedays.index(str(date)))
                l = len(group.index)
                s = int(day_session[0].strftime('%Y%m%d%H%M'))
                e = int(day_session[-1].strftime('%Y%m%d%H%M'))
                if l != num_minutes:
                    missing = check_missing_minutes(list(group.index))
                    print(group)
                    print(f"{RED}missing: {missing}{RESET}")
                    assert False, (f"{RED}Err(data fetch):{symbol}:{date} {l} out of {num_minutes} fetched{RESET}")
                elif s != group.index[0]:
                    assert False, (f"{RED}Err(data fetch):{symbol}:{date} start: {s}/{group.index[0]}{RESET}")
                elif e != group.index[-1]:
                    assert False, (f"{RED}Err(data fetch):{symbol}:{date} end: {e}/{group.index[-1]}{RESET}")
                
                daily_volume = int(group['volume'].sum())
                filepath = os.path.join(dir_path, f"{date}_{daily_volume}.parquet")
                group = group.sort_index()
                group.drop(columns='date', inplace=True)
                # disable compression for faster read
                group.to_parquet(filepath)  # compression=None)
                updated_dates.append(date)

        if len(updated_dates) != 0:
            for file in os.listdir(dir_path):
                if file.startswith("VALID_"):
                    _, valid_start, valid_end = file.split('_')
                    old_valid_file = os.path.join(dir_path, file)
                    new_valid_file = os.path.join(dir_path, f"VALID_{valid_start}_{max(updated_dates)}")
                    os.rename(old_valid_file, new_valid_file)
                    break
            print(f'[Main]: {symbol} updated: {updated_dates}')

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

    def print_to_file(self, data, filename):
        """Print data to a file"""
        with open(filename, 'a') as file:
            print(data, file=file)

    def save_to_json_compact(self, name: str, data: Dict):
        json_path = os.path.join(self.dir, name)
        # 1) Pretty-print as usual
        json_str = json.dumps(data, indent=4, ensure_ascii=False)

        # 2) Remove newlines and spaces after commas inside the innermost [...] blocks
        json_str = re.sub(
            r'\[[^\[\]]*?\]',
            lambda m: re.sub(
                r',\s*', ',', m.group(0).replace('\n', '').replace(' ', '')),
            json_str,
            flags=re.DOTALL
        )

        # 3) Write it back
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f'[Main]: Saved {BLUE}{name}{RESET} to {GREEN}{json_path}{RESET}')


if __name__ == '__main__':
    app = Main()
