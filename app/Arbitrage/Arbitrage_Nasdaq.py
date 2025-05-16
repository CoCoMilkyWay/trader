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
import akshare as ak
import exchange_calendars as ecals

from tqdm import tqdm
from typing import List, Dict, Any
from datetime import date, time, datetime, timedelta

from pprint import pprint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Arbitrage.API_TV import TV
from Arbitrage.API_IBKR import IBKR
from Arbitrage.API_QMT import QMT
from Arbitrage.Backtest import Backtest
from Arbitrage.Util import *

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
today = datetime.now().date()
trade_days = ecals.get_calendar('XSHG').sessions_in_range(pd.Timestamp(today - timedelta(days=14)), pd.Timestamp(today)).strftime('%Y%m%d').tolist()

if today.strftime('%Y%m%d') in trade_days and (time(8, 0) < datetime.now().time() < time(16, 0)):
    # this check if now is active trading session (pcf can be sure to be aligned, otherwise, maybe for different days)
    # fund manager would update NAV at different time (all ready before 8am in trading days)
    # '20250101'
    PCF_UPDATE_DATE = trade_days[-1] # we are certain all pcf files are ready and are for yesterday
    active = 1
else:
    PCF_UPDATE_DATE = trade_days[-2] # worst case (today's morning(trading), last day not updated, so the 2nd last day pcf
    active = 0

duration = 12*6  # months
UPDATE = False
DUMP = False
BACKTEST = True

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
# C:\Users\chuyin.wang\Desktop\share\fin\国金证券QMT交易端\userdata_mini
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
        # self.tv = TV(tv_username, tv_password)
        
        self.daily_routine_nq100()

    def daily_routine_nq100(self):
        # Prepare Assets info =====================================================================
        # =========================================================================================
        self.qdii_etf_info = self.get_etf_info(['ETF跨境型'])  # 'ETF行业指数'
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
        # year = str(PCF_UPDATE_DATE)[3]
        self.mnq_futures = self.ibkr.get_futures(symbol, 'Futures')  # for simplicity
        for fut in self.mnq_futures:
            assert fut.symbol == symbol, f"non-{symbol} futures found"
        # for non-main contracts, ibkr only records(return) bars with trade
        # for >3rd front month contracts, it is normal that no trades at all for a few days
        # expect returned data to be invalid
        pprint(self.mnq_futures)

        # self.cont_fut: Contract = self.mnq_futures[0]
        self.pri_fut = self.mnq_futures[0]
        self.sec_fut = self.mnq_futures[1]
        print(f'[Main]: Primary   MNQ contract({self.pri_fut.exchange}:{self.pri_fut.currency}): {RED}{self.pri_fut.symbol}: {self.pri_fut.localSymbol}{RESET}: {self.pri_fut.lastTradeDateOrContractMonth}')
        print(f'[Main]: Secondary MNQ contract({self.sec_fut.exchange}:{self.sec_fut.currency}): {RED}{self.sec_fut.symbol}: {self.sec_fut.localSymbol}{RESET}: {self.sec_fut.lastTradeDateOrContractMonth}')

        # prepare recent history
        fut_tz = "America/Chicago" # CME (Central time)
        stk_tz = "America/New_York" # NYSE/Nasdaq (Eastern time)
        etf_tz = "Asia/Shanghai" # SSE/SZSE/
        # | Region         | Time Zone   | UTC Offset (Standard) | UTC Offset (DST) | Shanghai Difference (Standard) | Shanghai Difference (DST) |
        # | -------------- | ----------- | --------------------- | ---------------- | ------------------------------ | ------------------------- |
        # | **Shanghai**   | CST (China) | UTC+8                 | N/A              | —                              | —                         |
        # | **US Central** | CST / CDT   | UTC−6                 | UTC−5            | **+14 hours**                  | **+13 hours**             |
        # | **US Eastern** | EST / EDT   | UTC−5                 | UTC−4            | **+13 hours**                  | **+12 hours**             |
        
        fut_end_date = datetime.now(pytz.timezone(fut_tz)).date()
        fut_start_date = fut_end_date - timedelta(days=30*duration)
        etf_end_date = datetime.now(pytz.timezone(etf_tz)).date()
        etf_start_date = fut_end_date - timedelta(days=30*duration)
        trade_days_cme = self.get_tradedays("CMES", fut_start_date, fut_end_date, type='futures')
        trade_days_sse = self.get_tradedays("XSHG", etf_start_date, etf_end_date, type='spot')
        print(f'[Main]: {YELLOW}CME     {RESET} trade days in last {duration} month: {len(trade_days_cme)}, sessions(America/Central): 17:00 - 15:59')
        print(f'[Main]: {YELLOW}SSE/SZSE{RESET} trade days in last {duration} month: {len(trade_days_sse)}, sessions(Asia/Shanghai):   09:30 - 14:57')
        
        if BACKTEST:
            history = pd.read_parquet(os.path.join(self.dir, "history.parquet"))
            print(history[-1000:])
            self.bt = Backtest()
            self.bt.backtest(history, [sym[0] for sym in nasdaq100])
            return
        
        if active and UPDATE:
            # Prepare Futures history data ============================================================
            # =========================================================================================
            for fut in self.mnq_futures:
                fut_details = self.ibkr.ib.reqContractDetails(fut)
                assert len(fut_details) == 1, f"Failed to qualify contract {fut}"
                missing_days_fut = self.check_bars(fut.localSymbol, trade_days_cme, fut_tz)
                if missing_days_fut > 0:
                    success, bars = self.ibkr.get_bars(fut, days=missing_days_fut, bar_size='1 min', exg_timezone=fut_tz)
                    if success:
                        self.store_bars(bars, fut.localSymbol, trade_days_cme, 'futures')

            # Prepare ETFs history data ===============================================================
            # =========================================================================================
            for etf in nasdaq100:
                missing_days_etf = self.check_bars(etf[0], trade_days_sse, etf_tz)
                if missing_days_etf > 0:
                    success, bars = self.qmt.get_bars(f"{etf[0]}.{etf[1]}", days=missing_days_etf, period='1m', exg_timezone=etf_tz)
                    if success:
                        self.store_bars(bars, etf[0], trade_days_sse, 'spot')
        
        # Synthesize Main Fut Contract ============================================================
        # =========================================================================================
        history_dir = os.path.join(self.dir, "history")
        start_date = int(fut_start_date.strftime('%Y%m%d'))

        fut_symbols, etf_symbols = [], []
        futures_data, etfs_data = {}, {}
        futures_records, etfs_records = [], []

        # Collect and organize data
        for symbol in tqdm(os.listdir(history_dir)):
            symbol_dir = os.path.join(history_dir, symbol)
            is_fut = 'MNQ' in symbol

            if is_fut:
                fut_symbols.append(symbol)
                futures_data.setdefault(symbol, [])
            else:
                etf_symbols.append(symbol)
                etfs_data.setdefault(symbol, [])

            for file in os.listdir(symbol_dir):
                if 'VALID_' in file:
                    continue
                
                date_str, value_str = file.removesuffix('.parquet').split('_')
                date = int(date_str)
                if date < start_date:
                    continue
                
                filepath = os.path.join(symbol_dir, file)
                df = pd.read_parquet(filepath)
                # df['symbol'] = symbol
                # df['date'] = date
                # df['value'] = int(value_str) if is_fut else float(value_str)

                if not df.empty:
                    if is_fut:
                        volume = int(value_str)
                        futures_records.append((date, volume, symbol))
                        futures_data[symbol].append(df)
                    else:
                        nav = float(value_str)
                        # nav calc time = 4pm US/East time (in/out DST) = 12 ~ 13 hrs after Shanghai
                        # open bar time = not the after hour auction price (should it be? maybe)
                        nyse_nasdaq_close_time = int((pd.to_datetime(str(date))).strftime('%Y%m%d%H%M')) + 1559
                        nav_sse_szse_time = pd.to_datetime(str(nyse_nasdaq_close_time)).tz_localize(stk_tz).tz_convert(etf_tz).strftime('%Y%m%d%H%M')
                        
                        etfs_records.append((date, nav, symbol))
                        df['nav'] = float("nan")
                        df.at[int(nav_sse_szse_time), 'nav'] = nav
                        etfs_data[symbol].append(df)
                else:
                    print(f'[Main]: {RED} Empty df from {symbol}: {file}{RESET}')

        futures_records.sort()

        # Main contract roll logic
        retired = set()
        main_contract = None
        main_volume = 0
        fut_roll = [] # H->M->U->Z (20240916: MNQZ4)

        for date, volume, contract in futures_records:
            if volume < 1000 or contract in retired:
                continue
            if main_contract is None:
                main_contract = contract
                main_volume = volume
                fut_roll.append((date, contract))
            elif contract != main_contract and volume > main_volume:
                # Only shift if contract is in its expiration month
                current_month = (date % 10000) // 100
                contract_exp_month = month_codes.get(main_contract[-2], 0)
                if current_month == contract_exp_month:
                    retired.add(main_contract)
                    main_contract = contract
                    main_volume = volume
                    fut_roll.append((date, contract))
            else:
                main_volume = volume

        # Concatenate all DataFrames per symbol (because of filter, some dfs may be empty)
        futures_concat = {sym: pd.concat(dfs).sort_index() for sym, dfs in futures_data.items() if dfs}
        etfs_concat = {sym: pd.concat(dfs).sort_index() for sym, dfs in etfs_data.items() if dfs}

        # Generate main_future_concat using fut_roll
        main_future_concat = []
        for i, (start_date, contract) in enumerate(fut_roll):
            end_date = fut_roll[i + 1][0] if i + 1 < len(fut_roll) else None
            start_date = start_date*10000
            end_date = end_date*10000 if end_date else None
            df = futures_concat.get(contract)
            if df is None:
                print(f"[Main]: {RED} No data for main contract {contract}{RESET}")
                continue
            if end_date:
                df = df[(df.index >= start_date) & (df.index < end_date)]
            else:
                df = df[df.index >= start_date]
            df = df.copy()
            df["main_contract"] = contract
            main_future_concat.append(df)
        
        # 20230528: MNQU3
        # 20230911: MNQZ3
        # 20231211: MNQH4
        # 20240311: MNQM4
        # 20240605: MNQU4
        # 20240916: MNQZ4
        # 20241216: MNQH5
        # 20250317: MNQM5
        
        # Final concatenated main future DataFrame
        main_future_concat = pd.concat(main_future_concat).sort_index()

        for date, contract in fut_roll:
            print(f"{date}: {contract}")
        # print(main_future_concat[:10000])
        
        # Merge History Data ======================================================================
        # =========================================================================================
        main_future_concat.index = pd.to_datetime(main_future_concat.index.astype(str)).tz_localize(fut_tz).tz_convert(etf_tz).strftime('%Y%m%d%H%M').astype('int64')
        for sym, df in etfs_concat.items():
            columns = ["close", "nav"] # ["open", "high", "low", "close", "volume", "nav"]
            etfs_concat[sym] = df.rename(columns={col: f"{sym}_{col}" for col in columns})[[f"{sym}_{col}" for col in columns]]
        
        etf_closes = [f"{sym}_close" for sym in etf_symbols]
        etf_navs = [f"{sym}_nav" for sym in etf_symbols]
        
        history = pd.concat([main_future_concat] + list(etfs_concat.values()), axis=1).sort_index()
        
        # FUT: for US holidays like Good Friday, CME is closed while SSE/SZSE is open
        history['close'] = history['close'].ffill()
        history['main_contract'] = history['main_contract'].ffill()
        history['volume'] = history['volume'].fillna(0)
        missing_mask = history['open'].isna()
        history.loc[missing_mask, ['open', 'high', 'low']] = history.loc[missing_mask, 'close'].values[:, None].repeat(3, axis=1) 
        
        # ETFs
        history[etf_closes] = history[etf_closes].ffill().round(4) # note: future info, but is okay
        
        for sym in etf_symbols:
            # NAV calculation
            history[f'{sym}_nav_pointer'] = np.nan
            history[f'{sym}_close_pointer'] = np.nan
            nav_updating_mask = ~ history[f'{sym}_nav'].isna()
            history.loc[nav_updating_mask, [f'{sym}_nav_pointer']] = history.loc[nav_updating_mask, f'{sym}_nav'].values[:, None].repeat(1, axis=1)
            history.loc[nav_updating_mask, [f'{sym}_close_pointer']] = history.loc[nav_updating_mask, 'close'].values[:, None].repeat(1, axis=1)
            
            history[f'{sym}_nav_pointer'] = history[f'{sym}_nav_pointer'].ffill()
            history[f'{sym}_close_pointer'] = history[f'{sym}_close_pointer'].ffill()
            
            # if active:
            #     # add last day's NAV value from data on website
            #     US_time = int((datetime.now().date() - timedelta(days=1)).strftime('%Y%m%d%H%M')) + 1559
            #     China_time = pd.to_datetime(str(US_time)).tz_localize(stk_tz).tz_convert(etf_tz).strftime('%Y%m%d%H%M')
            #     nav = self.nq_pcf_info[sym]['NAV']
            #     print(US_time, China_time, nav)
            #     history.at[int(China_time), f'{sym}_nav'] = nav
            
            history[f'{sym}_nav'] = (history[f'{sym}_nav_pointer'] * (history['close'] / history[f'{sym}_close_pointer'])).round(4)
            history = history.drop(columns=[f'{sym}_nav_pointer', f'{sym}_close_pointer'])
            
            history[f"{sym}_premium"] = ((history[f"{sym}_close"] - history[f"{sym}_nav"])/history[f"{sym}_nav"]*100).round(4)

        # Filter tradable session =================================================================
        # =========================================================================================
        hour_min = history.index % 10000  # Get HHMM part
        # Filter for Shanghai A-share session (09:30–11:29 and 13:00–14:59)
        mask = (
            ((hour_min >= 930) & (hour_min <= 1129)) |
            ((hour_min >= 1300) & (hour_min <= 1459))
        )
        filtered = history[mask] # only during this time arbitrage is possible
        plot_index = pd.to_datetime(filtered.index.astype(str), format='%Y%m%d%H%M').strftime('%H%M-%Y%m%d')
        # print(history[['close', '159509_close', '159509_nav', '159509_premium']]) # [-10000:])
        if DUMP:
            history.to_parquet(os.path.join(self.dir, "history.parquet"))
        
        # self.qmt.subscribe_QMT_etf([f"{syn[0]}.{syn[1]}" for syn in nasdaq100])
        # self.qmt.run()
        
        
        # interactive web GUI =====================================================================
        # =========================================================================================
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2])
        
        for sym in etf_symbols:
            if DUMP:
                px.line(x=plot_index, y=filtered[f"{sym}_premium"], width=1800, height=900).write_image(os.path.join(self.dir, f"fig/premium_{sym}.png"))
            fig.add_trace(go.Scatter(x=plot_index, y=filtered[f"{sym}_premium"], mode='lines', name=f"{sym}_premium"), row=1, col=1)
            
        fig.add_trace(go.Scatter(x=plot_index, y=filtered[f"close"], mode='lines', name=f"Main_Contract: {filtered.iloc[-1]['main_contract']}"), row=2, col=1)
        # fig.add_trace(go.Scatter(x=[date*10000 for date, contract in fut_roll], y=[0] * len(fut_roll), mode='markers', name='Future Rolls'))
        
        fig.update_layout(
            title='QDII NASDAQ100 price premium to NAV',
            xaxis_title='timestamp(min)',
            yaxis_title='premium(percent)',
            template='plotly_white',
            showlegend=True,
            height=700,  # This affects the absolute pixel height of the whole figure
        )
        fig.update_xaxes(
            showgrid=True,
            tickmode='linear',     # Force evenly spaced ticks
            dtick=60*4,               # Set tick interval (e.g., every 1 unit)
            gridcolor='lightgray',
            gridwidth=0.5
        )
        fig.show()
        
        # Run the system
        print('[API]: Running...')
        self.qmt.disconnect()
        self.ibkr.disconnect()
        # self.run()

        # =========================================================================================
        # =========================================================================================
        # =========================================================================================
        # =========================================================================================
        # =========================================================================================
        # =========================================================================================
        # =========================================================================================
        # =========================================================================================
        # =========================================================================================
        # =========================================================================================
        # =========================================================================================
        # =========================================================================================
        # =========================================================================================
        # =========================================================================================
        # =========================================================================================
        # =========================================================================================

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
            if active:
                assert int(PCF_UPDATE_DATE) == result['TradingDay'], f"Date mismatch: {PCF_UPDATE_DATE} != {result['TradingDay']}"
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

    def check_bars(self, symbol: str, tradedays: list[str], tz: str):
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
        today = datetime.now(pytz.timezone(tz)).date()
        delta_days = (today - start).days
        return delta_days

    def store_bars(self, df: pd.DataFrame, symbol: str, tradedays: list[str], type: str):
        """
        user make sure the first/last date has complete data
        """
        updated_dates = []
        df["date"] = df.index.astype(str).str[:8]  # e.g., "20250505"
        dir_path = os.path.join(self.dir, f"history/{symbol}")
        existing_dates = []
        for file in os.listdir(dir_path):
            if file.endswith('.parquet'):
                date = file.split('_')[0]
                existing_dates.append(date)

        start_date = min(df["date"])
        cur_date = max(df["date"])
        if type == 'spot':
            try:
                nav_table = ak.fund_etf_fund_info_em(fund=symbol, start_date=start_date)
                nav_table['净值日期'] = pd.to_datetime(nav_table['净值日期'])
            except:
                return
            
        for date, group in df.groupby("date"):
            if date not in existing_dates:
                # data integrity checks (data may not be reliable)
                try:
                    i = tradedays.index(str(date))
                    if not (0 < i < len(tradedays) - 1):
                        continue
                except:
                    continue
                
                if type == 'spot':
                    day_session, num_minutes = get_A_stock_day_session(str(date))
                elif type == 'futures':
                    day_session, num_minutes = get_cme_day_session(tradedays, i)
                else:
                    assert False, f"Invalid type: {type}"

                if type == 'spot':
                    target_date = pd.to_datetime(str(date))
                    if target_date in nav_table['净值日期'].values:
                        row = nav_table[nav_table['净值日期'] == target_date]
                        nav = row['单位净值'].values[0]
                        filepath = os.path.join(dir_path, f"{date}_{nav}.parquet")
                        
                        day_session = day_session.strftime('%Y%m%d%H%M').astype('int64')
                        group = group.reindex(day_session)
                        group.drop(columns='date', inplace=True)
                        # disable compression for faster read
                        group.to_parquet(filepath)  # compression=None)
                        updated_dates.append(date)
                    
                elif type == 'futures':
                    daily_volume = int(group['volume'].sum())
                    if daily_volume > 0:
                        filepath = os.path.join(dir_path, f"{date}_{daily_volume}.parquet")
                        day_session = day_session.strftime('%Y%m%d%H%M').astype('int64')
                        group = group.reindex(day_session)
                        group.drop(columns='date', inplace=True)
                        # disable compression for faster read
                        group.to_parquet(filepath)  # compression=None)
                    updated_dates.append(date)
                else:
                    assert False, f"Invalid type: {type}"

        if len(updated_dates) != 0:
            for file in os.listdir(dir_path):
                if file.startswith("VALID_"):
                    _, valid_start, valid_end = file.split('_')
                    old_valid_file = os.path.join(dir_path, file)
                    new_valid_file = os.path.join(dir_path, f"VALID_{valid_start}_{max(updated_dates)}")
                    os.rename(old_valid_file, new_valid_file)
                    break
            print(f'[Main]: {RED}{symbol}{RESET} updated: {GREEN}{updated_dates}{RESET}')

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
