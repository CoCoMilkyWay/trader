from config.cfg_stk import cfg_stk
import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Tuple

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
PURPLE = '\033[95m'
CYAN = '\033[96m'
DEFAULT = '\033[0m'

SEC_IN_HALF_YEAR = int(3600*24*365*0.5)


def load_json(file_path):
    import json
    # Note the 'utf-8-sig' which handles BOM if present
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return json.load(file)


def dump_json(file_path, df):
    import json
    with open(file_path, 'w', encoding='utf-8', errors='ignore') as file:
        # df.to_json(file, orient='records', force_ascii=False, indent=4)
        json.dump(df, file, indent=4, ensure_ascii=False)


def mkdir(path_str):
    path = os.path.dirname(path_str)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path_str

# def log(path_str, str, stdout=1, type='a'):
#     # 'w+': write and read
#     # 'w': overwrite
#     # 'a': append
#     with open(path_str,type) as file:
#         # Writing data to the file
#         file.write(f'{str}\n')
#         # # Moving the cursor to the beginning of the file to read from the beginning
#         # file.seek(0)
#         # data = file.read()
#     if stdout:
#         print(str)
#     return str
#
# from wtpy import WtBtEngine, EngineType, WtDtServo
# from wtpy.monitor import WtBtSnooper
# from wtpy.wrapper import WtDataHelper
# from wtpy.apps import WtBtAnalyst
# from wtpy.WtCoreDefs import WTSBarStruct
# from wtpy.SessionMgr import SessionMgr
#
# def testBtSnooper():
#     pass


# ================================================
# CSV to DATABASE file (DSB indexed my month/day)
# ================================================

# dtHelper = WtDataHelper()
#
# def generate_database_files(wt_assets:list, force_sync:bool=False):
#     '''
#     force-sync: would check DATABASE file for each csv \n
#     non-force-sync(default): only check symbol name exist in DATABASE
#     '''
#     print('Analyzing/Generating L1(CSV)/L2(DSB) datebase files...')
#     print(f"SRC_CSV:          {GREEN}{cfg_cpt.CRYPTO_CSV_DIR} /<symbol>/1m/{DEFAULT}")
#     print(f"DB_DSB:           {GREEN}{cfg_cpt.CRYPTO_DB_DIR} /<symbol>/1m/{DEFAULT}")
#
#     assets = wt_assets
#     unprocessed_asset_list = []
#     processed_asset_list = os.listdir(mkdir(f"{cfg_cpt.CRYPTO_DB_DIR}/"))
#
#     # print(f'Num of processed L2(DSB) assets: {len(processed_asset_list)}')
#
#     for asset in assets:
#         if asset not in processed_asset_list or force_sync:
#             unprocessed_asset_list.append(asset)
#         else:
#             pass
#             # print(f'asset {asset} already processed')
#     num_assets = len(unprocessed_asset_list)
#     # print(f'DB(1m): num of assets to be processed: {num_assets}')
#
#     # Replace your for loop with:
#     if len(unprocessed_asset_list)!=0:
#         process_all_assets(unprocessed_asset_list)
#
# from multiprocessing import Pool
# import multiprocessing as mp
#
# def process_all_assets(unprocessed_asset_list):
#     # Use number of CPU cores minus 1 to avoid overloading
#     num_processes = max(1, mp.cpu_count() - 1)
#
#     with Pool(num_processes) as pool:
#         # Use tqdm to show progress
#         results = list(tqdm(
#             pool.imap(process_single_asset, unprocessed_asset_list),
#             total=len(unprocessed_asset_list)
#         ))
#     return results
#
# def process_single_asset(asset):
#     src_folder = f"{cfg_cpt.CRYPTO_CSV_DIR}/{asset}/1m/"
#     src_csvs = os.listdir(src_folder)
#     db_folder = mkdir(f"{cfg_cpt.CRYPTO_DB_DIR}/{asset}/1m/")
#     db_dsbs = os.listdir(db_folder)
#
#     for src_csv in src_csvs:
#         if not str(src_csv).endswith('csv'):
#             continue
#         year = src_csv.split("-")[2]
#         month = src_csv.split("-")[3].split(".")[0]
#         db_dsb = f'{year}.{month}.dsb'
#         if db_dsb not in db_dsbs:
#             src_csv_path = os.path.join(src_folder, src_csv)
#             db_dsb_path = os.path.join(db_folder, db_dsb)
#             process_dataframe(src_csv_path, db_dsb_path)
#     return asset
#
# def process_dataframe(csv_file_path, dsb_file_path):
#     def process_Binance():
#         # Binance csv starts from month xx: 8:01am
#         # print(csv_file_path)
#         df = pd.read_csv(csv_file_path, header=None, skiprows=1, encoding='utf-8', on_bad_lines='warn')
#         # Open_time Open High Low Close Volume Close_time Quote_asset_volume Number_of_trades Taker_buy_base_asset_volume Taker_buy_quote_asset_volume Ignore
#         df = df.iloc[:, :6]
#         df.columns = ['time', 'open', 'high', 'low', 'close', 'vol']
#         # datetime = pd.to_datetime(df['trade_time'], format='%Y-%m-%d %H:%M:%S', errors='raise') # '%Y-%m-%d %H:%M:%S.%f'
#         # df['date'] = datetime
#         # df['year'] = datetime.dt.year
#         # df['month'] = datetime.dt.month
#         # df['day'] = datetime.dt.day
#         # df['time'] = datetime.dt.time
#         return df
#     store_bars(process_Binance(), dsb_file_path)
#     # print(dtHelper.read_dsb_bars(dsb_file_path).to_df())
#
# def store_bars(df, file_path): # 'date' 'time' 'open' 'high' 'low' 'close' 'vol'
#     if os.path.exists(file_path):
#         pass
#     else:
#         # '%Y-%m-%d %H:%M:%S.%f'
#         datetime = pd.to_datetime(df['time'], unit='ms') # Unix Epoch time to datetime
#         df['date'] = datetime.dt.strftime('%Y%m%d').astype('int64')
#         df['time'] = datetime.dt.strftime('%H%M').astype('int')
#
#         # wt-specific times
#         df['time'] = (df['date'] - 19900000)*10000 + df['time']
#
#         df = df[['date', 'time', 'open', 'high', 'low', 'close', 'vol']].reset_index(drop=True)
#         BUFFER = WTSBarStruct*len(df)
#         buffer = BUFFER()
#         def assign(procession, buffer):
#             tuple(map(lambda x: setattr(buffer[x[0]], procession.name, x[1]), enumerate(procession)))
#         df.apply(assign, buffer=buffer)
#         store_path = mkdir(file_path)
#         dtHelper.store_bars(barFile=store_path, firstBar=buffer, count=len(df), period="m1")
#
# # ================================================
# # DATABASE file to MERGED DATABASE file
# # ================================================
# def generate_merged_database_files(symbols:list, resample_n:int=1, begin_date=datetime(1990,1,1), end_date=datetime(2050,1,1), total=True):
#     '''
#     generate temporary merged DB files for back-testing
#     '''
#     print('Analyzing/Generating L3(DSB)/L4(DSB) datebase files...')
#     print(f"MERGED_DB_DSB:    {GREEN}{cfg_cpt.WT_STORAGE_DIR}/his/min1/{cfg_cpt.market}/ <symbol>.dsb{DEFAULT}")
#     print(f"RESAMPLED_DB_DSB: {GREEN}{cfg_cpt.WT_STORAGE_DIR}/his/{'min'+str(resample_n)}/{cfg_cpt.market}/ <symbol>.dsb{DEFAULT}")
#
#     for asset in tqdm(symbols, desc=f'Merging and Resampling(x{resample_n})...'):
#         database_db_folder  = f"{cfg_cpt.CRYPTO_DB_DIR}/{asset}/1m/"
#         merged_db_path      = f'{cfg_cpt.WT_STORAGE_DIR}/his/min1/{cfg_cpt.market}/{asset}.dsb'
#         resampled_db_path   = f'{cfg_cpt.WT_STORAGE_DIR}/his/{'min'+str(resample_n)}/{cfg_cpt.market}/{asset}.dsb'
#         try:
#             combine_dsb_1m(asset, database_db_folder, merged_db_path)
#             # print(dtHelper.read_dsb_bars(merged_db_path).to_df())
#
#         except Exception as e:
#             print(f'Err processing: {asset}')
#             print(e)
#             continue
#         if resample_n != 1:
#             resample(merged_db_path, resample_n, resampled_db_path)
#
# def combine_dsb_1m(asset, database_db_folder, merged_db_path, begin_date=datetime(1990,1,1), end_date=datetime(2050,1,1), total=True):
#     df = []
#     if not os.path.exists(merged_db_path):
#         if total: # read ALL dsb file and return DF
#             sorted_file_list = sort_files_by_date(database_db_folder)
#         else: # only read SOME data and return DF(do not combine all dsb)
#             sorted_file_list = sort_files_by_date(database_db_folder, begin_date, end_date)
#         for file in tqdm(sorted_file_list, desc=f'{asset}'):
#             file_path = os.path.join(database_db_folder, file)
#             df.append(dtHelper.read_dsb_bars(file_path).to_df())
#         df = pd.concat(df, ignore_index=False)
#         print(df)
#         wt_df_2_dsb(df, mkdir(merged_db_path))
#
# def sort_files_by_date(folder_path, start_date=datetime(1900,1,1), end_date=datetime(2050,1,1)):
#     files = [f for f in os.listdir(folder_path) if f.endswith('.dsb')]
#     files_with_dates = []
#     for file in files:
#         file_date = get_file_date(file)
#         if start_date < file_date < end_date:
#             files_with_dates.append((file, file_date))
#     # Sort the files by their date
#     sorted_files = sorted(files_with_dates, key=lambda x: x[1])
#     return [file for file, _ in sorted_files]
#
# def get_file_date(filename):
#     from datetime import datetime
#     base_name = filename[:-4]  # Remove last 4 characters (".dsb")
#     parts = base_name.split('.')
#     if len(parts) == 3:
#         year, month, day = parts
#         return datetime(int(year), int(month), int(day))
#     elif len(parts) == 2:
#         year, month = parts
#         return datetime(int(year), int(month), 1)  # Treat as the first day of the month
#     else:
#         return datetime(0000, 00 ,00)  # Invalid format
#
# def resample(src_path, times, store_path):
#     if not os.path.exists(store_path):
#         script_dir = os.path.dirname(os.path.realpath(__file__))
#         sessMgr = SessionMgr()
#         sessMgr.load(f"{script_dir}/../../config/cpt_sessions.json")
#         sInfo = sessMgr.getSession("ALLDAY")
#         df = dtHelper.resample_bars(
#             barFile=src_path,
#             period='m1',
#             times=times,
#             fromTime=200001010931,
#             endTime=205001010931,
#             sessInfo=sInfo,
#             alignSection=False).to_df()
#         # time: day: yyyymmdd，min: yyyymmddHHMMSS
#         wt_df_2_dsb(df, mkdir(store_path))
#         # print(dtHelper.read_dsb_bars(src_path).to_df())
#         # print(df)
#
# def wt_df_2_dsb(df, store_path):
#     # index: open high low close settle turnover volume open_interest diff bartime
#     # 'date' 'time' 'open' 'high' 'low' 'close' 'vol'
#     df = df.rename(columns={'volume': 'vol'})
#     df['date'] = df['bartime'].astype(str).str[:8].astype(int)
#     df['time'] = df['bartime']-199000000000
#     df = df[['date', 'time', 'open', 'high', 'low', 'close', 'vol']].reset_index(drop=True)
#     BUFFER = WTSBarStruct*len(df)
#     buffer = BUFFER()
#     def assign(procession, buffer):
#         tuple(map(lambda x: setattr(buffer[x[0]], procession.name, x[1]), enumerate(procession)))
#     df.apply(assign, buffer=buffer)
#     dtHelper.store_bars(barFile=store_path, firstBar=buffer, count=len(df), period="m1")

# ================================================
# generate asset list
# ================================================


def generate_asset_list(num=None):
    from DataProvider_API.Mairui.MairuiAPI import MairuiAPI
    from DataProvider_API.Lixingren.LixingrenAPI import LixingrenAPI
    API = LixingrenAPI()
    # r = API.query("basic_all")
    file_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'combined_data.json')
    # dump_json(file_path,r)

    import time
    timestamp_current_s = int(time.time())

    asset_list_updated = os.path.exists(cfg_stk.ASSET_FILE)

    if asset_list_updated:
        try:
            with open(cfg_stk.ASSET_FILE, 'r', encoding='gbk', errors='ignore') as file:
                timestamp_last_update_ms = os.path.getmtime(cfg_stk.ASSET_FILE)
                timestamp_last_update_s = timestamp_last_update_ms / 1000
                dt = datetime.fromtimestamp(timestamp_last_update_s)

                updated_within_1_day = abs(
                    # 24hrs
                    timestamp_current_s - timestamp_last_update_s) <= (3600*24)

                if updated_within_1_day:
                    print(
                        f'A-stock ExchangeInfo Already Updated: {dt.year}-{dt.month}-{dt.day}')
                else:
                    asset_list_updated = False
                    print(
                        f'Old A-stock ExchangeInfo: {dt.year}-{dt.month}-{dt.day}')
        except:
            asset_list_updated = False
            print(f'Error reading A-stock ExchangeInfo')

    Status = {
        "normally_listed": {
            "name": "正常上市",
            "description": "指公司目前在证券交易所上市，并正常交易，没有任何限制。"
        },
        "delisted": {
            "name": "已退市",
            "description": "指公司的股票已从交易所撤销，不能再进行交易。这可能由于多种原因，包括财务不稳定或未能满足上市要求。"
        },
        "listing_suspended": {
            "name": "暂停上市",
            "description": "意味着公司的股票交易已被暂时暂停，可能由于正在进行的调查、财务困难或其他需要解决的问题。"
        },
        "special_treatment": {
            "name": "ST板块",
            "description": "该类别的股票因财务困难或其他重大风险而受到特别处理，通常会受到更严格的监管和监控。"
        },
        "delisting_risk_warning": {
            "name": "*ST",
            "description": "该标识表示公司因财务状况或其他严重问题而面临退市风险，提醒投资者可能会被退市的风险。"
        },
        "issued_but_not_listed": {
            "name": "已发行未上市",
            "description": "指已发行但目前未在任何证券交易所上市的证券，因此无法进行公开交易。"
        },
        "pre_disclosure": {
            "name": "预披露",
            "description": "该状态表示公司计划上市，并已就其意图进行初步披露，通常是在首次公开募股（IPO）之前。"
        },
        "unauthorized": {
            "name": "未过会",
            "description": "表示公司尚未获得监管机构的必要批准以进行上市或公开发行其股票。"
        },
        "issue_failure": {
            "name": "发行失败",
            "description": "该术语指公司未能成功发行证券，通常意味着没有吸引到足够的投资者兴趣。"
        },
        "delisting_transitional_period": {
            "name": "进入退市整理期",
            "description": "该状态表示公司在正式退市之前的一个阶段，此期间可能还会继续交易，但会受到密切监控。"
        },
        "ipo_suspension": {
            "name": "暂缓发行",
            "description": "意味着公司的首次公开募股（IPO）计划已被暂时暂停，可能由于监管问题或市场状况。"
        },
        "ipo_listing_suspension": {
            "name": "暂缓上市",
            "description": "类似于上面，表示某项证券的上市已被推迟。"
        },
        "transfer_suspended": {
            "name": "停止转让",
            "description": "表示股票的所有权转让已被暂停，这可能是由于监管问题或其他复杂情况。"
        },
        "normally_transferred": {
            "name": "正常转让",
            "description": "指股票在没有任何限制或特殊情况的情况下正常进行转让。"
        },
        "investor_suitability_management_implemented": {
            "name": "投资者适当性管理标识",
            "description": "该标识表示公司正在实施投资者适当性管理措施，确保其投资产品适合目标投资者群体。"
        },
        "non_listed": {
            "name": "非上市",
            "description": "该术语描述的是未在任何证券交易所上市的证券，因此不进行公开交易。"
        },
        "transfer_as_specific_bond": {
            "name": "特定债券转让",
            "description": "指在特定条款和条件下转让某些债券，通常不在常规交易框架内进行。"
        },
        "transfer_under_agreement": {
            "name": "协议转让",
            "description": "表示所有权的转让是基于双方之间的协议进行的，而不是通过公共交易过程。"
        },
        "others": {
            "name": "其它",
            "description": "这是一个涵盖未在其他定义中列出的任何上市状态的通用类别。"
        }
    }

    tradable = {
        'normally_listed',
        'special_treatment',
        # 'delisting_risk_warning',
        # 'delisting_transitional_period',
        # 'normally_transferred',
        # 'investor_suitability_management_implemented'
    }

    if not asset_list_updated:
        from DataProvider_API.Lixingren.LixingrenAPI import LixingrenAPI
        print('HTTP Querying A-stock ExchangeInfo...')
        API = LixingrenAPI()
        # info = API.query("basic_all")
        info = load_json(file_path)

        output = {
            "SSE": {},
            "SZSE": {},
            "BJSE": {},
        }

        exg = {'bj': 'BJSE', 'sh': 'SSE', 'sz': 'SZSE'}

        # Populate the data for each symbol
        for symbol in info:
            name = symbol.get('name')
            market = symbol.get('market')
            exchange = symbol.get('exchange')
            areaCode = symbol.get('areaCode')
            stockCode = symbol.get('stockCode')
            fsTableType = symbol.get('fsTableType')
            ipoDate = symbol.get('ipoDate')
            listingStatus = symbol.get('listingStatus')
            mutualMarkets = symbol.get('mutualMarkets')

            if not (name and market and exchange and areaCode and stockCode and fsTableType and ipoDate and listingStatus):
                print('Skipping for incomplete info: ', name)
                continue
            
            if listingStatus not in tradable:
                print('Skipping for listingStatus:', name, Status[listingStatus]['description'])
                continue
            
            if areaCode not in ['cn']:
                print('Skipping for areaCode:', name, areaCode)
                continue
            
            if market not in ['a']:
                print('Skipping for market:', name, market)
                continue
            
            if exchange not in ['sh', 'sz', 'bj']:
                print('Skipping for exchange:', name, exchange)
                continue
            
            # if fsTableType not in ['non_financial']:
            #     print('Skipping for fsTableType:', name, fsTableType)
            #     continue
            
            ipo_date_s = datetime.fromisoformat(ipoDate).timestamp()
            if abs(timestamp_current_s - ipo_date_s) <= SEC_IN_HALF_YEAR:
                print('Skipping for recency(half year):', name)
                continue
                        # 
            # "Binance": {
            #     "BTCUSDT": {
            #         "code": "BTCUSDT",
            #         "name": "BTCUSDT_UM_PERP",
            #         "exchg": "Binance",
            #         "extras": {
            #             "instType": "PERPETUAL",
            #             "baseCcy": "BTC",
            #             "quoteCcy": "USDT",
            #             "category": 22,
            #             "serverTime": 1741593610171,
            #             "onboardDate": 1569398400000,
            #             "deliveryDate": 4133404800000,
            #             "underlyingSubType": [
            #                 "PoW"
            #             ],
            #             "liquidationFee": "0.012500",
            #             "triggerProtect": "0.0500",
            #             "marketTakeBound": "0.05"
            #         },
            #         "rules": {
            #             "session": "ALLDAY",
            #             "holiday": "NO_HOLIDAYS",
            #             "covermode": 3,
            #             "pricemode": 0,
            #             "trademode": 0,
            #             "precision": 2,
            #             "pricetick": 0.1,
            #             "lotstick": 0.001,
            #             "minlots": 0.001,
            #             "volscale": 100
            #         }
            #     },
            # 
            output[exg[exchange]][name] = {
                "code": name,
                "name": f"{name}_UM_PERP",
                "exchg": "Binance",
                "extras": {
                    "instType": "PERPETUAL",
                    "baseCcy": symbol['baseAsset'],
                    "quoteCcy": symbol['quoteAsset'],
                    "category": 22,     # 分类，参考CTP
                                        # 0=股票 1=期货 2=期货期权 3=组合 4=即期
                                        # 5=期转现 6=现货期权(股指期权) 7=个股期权(ETF期权)
                                        # 20=数币现货 21=数币永续 22=数币期货 23=数币杠杆 24=数币期权
                    # "ctVal": "",
                    # "ctValCcy": "",
                    # "lever": "10",
                    # "ctType": ""

                    "serverTime": info['serverTime'],
                    "onboardDate": symbol['onboardDate'],
                    "deliveryDate": symbol['deliveryDate'],
                    "underlyingSubType": symbol['underlyingSubType'],
                    "liquidationFee": symbol['liquidationFee'],
                    "triggerProtect": symbol['triggerProtect'],
                    # threshold for algo order with "priceProtect"
                    "marketTakeBound": symbol['marketTakeBound'],
                    # the max price difference rate( from mark price) a market order can make
                },
                "rules": {
                    "session": "ALLDAY",
                    "holiday": "NO_HOLIDAYS",
                    "covermode": 3,     # 0=开平，1=开平昨平今，2=平未了结的，3=不区分开平
                    "pricemode": 0,     # 价格模式 0=市价限价 1=仅限价 2=仅市价
                    "trademode": 0,     # 交易模式，0=多空都支持 1=只支持做多 2=只支持做多且T+1
                    "precision": int(symbol['pricePrecision']),
                    # 价格小数点位数
                    "pricetick": float(symbol['filters'][0]['tickSize']),
                    # 最小价格变动单位
                    "lotstick": float(symbol['filters'][2]['stepSize']),
                    # 最小交易手数
                    "minlots": float(symbol['filters'][2]['minQty']),
                    # 最小交易手数
                    "volscale": 100,    # 合约倍数
                }

            }
    #     # Write to JSON file
    #     with open(cfg_stk.ASSET_FILE, 'w') as f:
    #         json.dump(output, f, indent=4)
    #
    # with open(cfg_stk.ASSET_FILE, 'r', encoding='gbk', errors='ignore') as file:
    #     asstes_info = json.load(file)

    wt_assets = []
    symbols = []
    # all_underlying_SubTypes = []
    # cnt = 0
    # exchange = 'Binance'
    # product = 'UM'
    # for symbol_key, symbol_value in asstes_info[exchange].items():
    #     wt_assets.append(f'{exchange}.{product}.{symbol_key}')
    #     symbols.append(symbol_key)
    #     for subtype in symbol_value['extras']['underlyingSubType']:
    #         if subtype not in all_underlying_SubTypes:
    #             all_underlying_SubTypes.append(subtype)
    #     cnt += 1
    #     if num and cnt >= num:
    #         break
    #
    # print('All underlying SubTypes:', all_underlying_SubTypes)
    # print('Number of assets:', len(wt_assets))
    #
    return wt_assets, symbols

# ================================================
# Others
# ================================================


def enable_logging():
    import logging
    '''
    This needs to be set up before init(import) of packages(with logging)
    '''
    print(f'Logging saved to logs/wtcpp.log')
    log_file = mkdir('logs/wtcpp.log')
    with open(log_file, 'w', encoding='utf-8') as f:
        # f.write('@charset "gbk";\n')  # CSS-style encoding declaration
        f.write('/* @encoding=gbk */\n')  # Comment-style declaration
    logging.basicConfig(
        filename=log_file,      # Specify the output file
        filemode='a',           # 'w' to overwrite, 'a' to append
        level=logging.NOTSET,   # Capture all levels
        # Only output the message without timestamps etc
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

# def get_bao_stocks(pool: str = 'hs300') -> Tuple[List[str], bool]:
#     import baostock as bs
#     import pandas as pd
#     lg = bs.login()
#     print('login respond error_code:'+lg.error_code)
#     print('login respond  error_msg:'+lg.error_msg)
#     bao_valid:bool = False
#     if pool == 'hs300':
#         rs = bs.query_hs300_stocks()
#         bao_valid = True
#     elif pool == 'zz500':
#         rs = bs.query_zz500_stocks()
#         bao_valid = True
#     print('query error_code:'+rs.error_code)
#     print('query  error_msg:'+rs.error_msg)
#     bao_stocks = []
#     while (rs.error_code == '0') & rs.next():
#         bao_stocks.append(rs.get_row_data())
#     bs.logout()
#     bao_df = pd.DataFrame(bao_stocks, columns=rs.fields)
#     bao_ls = list(bao_df['code'])
#     return bao_ls, bao_valid

# def time_diff_in_min(start: int, end: int) -> int:
#     from datetime import datetime
#     def parse_time(time: int) -> datetime:
#         time_str = str(time)
#         # Extract time components from the last 10 characters of the string
#         year   = int(time_str[-12:-8])
#         month  = int(time_str[-8:-6])
#         day    = int(time_str[-6:-4])
#         hour   = int(time_str[-4:-2])
#         minute = int(time_str[-2:])
#         return datetime(year, month, day, hour, minute)
#     # Parse both start and end strings into datetime objects
#     start_time = parse_time(start)
#     end_time   = parse_time(end)
#     # Calculate the difference in time
#     delta = end_time - start_time
#     # Convert the time difference to minutes and return it as an integer
#     min_diff = int(delta.total_seconds() // 60)
#     return min_diff
