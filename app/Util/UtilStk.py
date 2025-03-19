from wtpy.monitor import WtBtSnooper
from wtpy import WtDtServo
from wtpy.SessionMgr import SessionMgr
from wtpy.WtCoreDefs import WTSBarStruct
from multiprocessing import Pool
import multiprocessing as mp
from wtpy.wrapper import WtDataHelper
from config.cfg_stk import cfg_stk
import os
import json
import time
import copy
import tempfile
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple

from DataProvider_API.Lixingren.LixingrenAPI import LixingrenAPI
from DataProvider_API.Baostock.BaostockAPI import BaostockAPI


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


def dump_json(file_path, df, desc=None):
    if desc:
        print(f'{desc+':':20}{GREEN}{file_path}{DEFAULT} ... ', end="")
    # Create a temporary file in the same directory as the target file.
    dir_name = os.path.dirname(file_path)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_name, encoding='utf-8', errors='ignore') as temp_file:
        # Write the JSON data to the temporary file.
        json.dump(df, temp_file, indent=4, ensure_ascii=False)
        # Flush the file and force the OS to write to disk.
        temp_file.flush()
        os.fsync(temp_file.fileno())

    # Atomically replace the target file with the temporary file.
    os.replace(temp_file.name, file_path)
    if desc:
        print(f'dumped')


def store_sparse_df(df: pd.DataFrame, path: str):
    # df has a time index and many columns with leading NaNs
    df.astype(pd.SparseDtype("float64", fill_value=float("nan"))
              ).to_parquet(path, index=True)


def load_sparse_df(path: str):
    return pd.read_parquet(path).sparse.to_dense()


def store_compressed_array(arr, path):
    np.savez_compressed(mkdir(path), data=arr)


def load_compressed_array(path):
    with np.load(path) as data:
        return data['data']


def mkdir(path_str):
    path = os.path.dirname(path_str)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path_str


def testBtSnooper():
    dtServo = WtDtServo()
    dtServo.setBasefiles(
        folder="cfg/",
        commfile="assets_cfg/stk_comms.json",
        contractfile="assets_list/stocks.json",
        holidayfile="misc/holidays.json",
        sessionfile="sessions/sessions.json",
        hotfile="assets_list/hots.json"
    )
    dtServo.setStorage(
        path='storage',
        adjfactor='cfg/misc/adjfactors.json'
    )
    snooper = WtBtSnooper(dtServo)
    snooper.run_as_server(port=8081, host="0.0.0.0")

# ================================================
# prepare all files
# ================================================


LXR_API = LixingrenAPI()
BS_API = BaostockAPI()


def prepare_all_files():
    """
    wt: wondertrader/wtpy
    lxr: lixingren
    """

    wt_asset = _wt_asset_file(
        cfg_stk.wt_asset_file)
    lxr_profile, wt_asset = _lxr_profile_file(
        cfg_stk.lxr_profile_file, wt_asset)
    lxr_industry, wt_asset = _lxr_industry_file(
        cfg_stk.lxr_industry_file, wt_asset)
    wt_adj_factor, wt_asset = _wt_adj_factor_file(
        cfg_stk.wt_adj_factor_file, wt_asset)
    wt_tradedays, wt_holidays = _wt_tradedays_holidays_file(
        cfg_stk.wt_tradedays_file,
        cfg_stk.wt_holidays_file)

    dump_json(cfg_stk.wt_asset_file, wt_asset, "wt_asset")
    dump_json(cfg_stk.lxr_profile_file, lxr_profile, "lxr_profile")
    dump_json(cfg_stk.lxr_industry_file, lxr_industry, "lxr_industry")
    dump_json(cfg_stk.wt_adj_factor_file, wt_adj_factor, "wt_adj_factor")
    dump_json(cfg_stk.wt_tradedays_file, wt_tradedays, "wt_tradedays")
    dump_json(cfg_stk.wt_holidays_file, wt_holidays, "wt_holidays")

    lxr_meta = _lxr_fundamental_file(
        cfg_stk.STOCK_DB_FUND_DIR, wt_asset, wt_tradedays)
    dump_json(f"{cfg_stk.STOCK_DB_FUND_DIR}/meta.json",
              lxr_meta, 'lxr_index_meta')

    process_bar_data(wt_asset, force_sync=False)

    return wt_asset


def _wt_asset_file(path: str) -> Dict:

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

    def get_sub_exchange(code):
        if code.startswith('60'):
            return 'SSE.A'
        elif code.startswith('900'):
            return 'SSE.B'
        elif code.startswith('68'):
            return 'SSE.STAR'
        elif code.startswith('000') or code.startswith('001'):
            return 'SZSE.A'
        elif code.startswith('200'):
            return 'SZSE.B'
        elif code.startswith('300') or code.startswith('301'):
            return 'SZSE.SB'
        elif code.startswith('002') or code.startswith('003'):
            return 'SZSE.A'
        elif code.startswith('440') or code.startswith('430') or code.startswith('83') or code.startswith('87'):
            return 'NQ'
        else:
            print('Unknown sub-exchange: ', code)
            return 'Unknown'

    state = _check_state(path, "WT-AssetInfo", 1)

    if state != 0:
        old = load_json(path)
    else:  # non-exist
        old = {}

    print('HTTP Querying A-stock ExchangeInfo...')
    new = LXR_API.query("basic_all")

    output = {"SSE": {}, "SZSE": {}, "BJSE": {}, }
    simple = copy.deepcopy(output)
    map = {'bj': 'BJSE', 'sh': 'SSE', 'sz': 'SZSE'}

    # Populate the data for each symbol
    for symbol in new:
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
            print('Skipping for listingStatus:', name,
                  Status[listingStatus]['description'])
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
        if abs(int(time.time()) - ipo_date_s) <= SEC_IN_HALF_YEAR:
            print('Skipping for recency(half year):', name)
            continue

        exg = map[exchange]
        output[exg][stockCode] = {
            "code": stockCode,
            "exchg": exg,
            "name": name,
            "product": "STK",
            "extras": {
                "ipoDate": ipoDate,
                "subexchg": get_sub_exchange(stockCode),
                "industry_names": None,
                "industry_codes": None,
                "companyName": None,
                "website": None,
                "city": None,
                "province": None,
                "businessScope": None,
                # 'ha': 港股通
                "mutualMarkets": mutualMarkets if mutualMarkets else [],
                "fsTableType": fsTableType,

                "update_time_profile": None,
                "update_time_adjfactor": _get_nested_value(old, [exg, stockCode, 'extras', 'update_time_adjfactor']),
                "update_time_fundamental": _get_nested_value(old, [exg, stockCode, 'extras', 'update_time_fundamental']),
            },
        }

        simple[exg][stockCode] = {
            "code": stockCode,
            "exchg": exg,
            "name": name,
            "product": "STK",
        }
    dump_json(f"{cfg_stk.script_dir+'/stk_assets_simple.json'}",
              simple, "wt_asset")
    return output


def _lxr_profile_file(path: str, wt_asset: Dict):
    state = _check_state(path, "LXR-Profile", 1)
    exgs = cfg_stk.exchg
    API_LIMITS = 100

    if state != 0:  # exist
        old_lxr = load_json(path)
    else:
        old_lxr = {}
        for exg in exgs:
            old_lxr[exg] = {}

    time = datetime.now(timezone(timedelta(hours=8))).isoformat()  # East Asia

    lxr = {}
    for exg in exgs:
        lxr[exg] = {}
        pending_assets = []
        for key in wt_asset[exg]:
            if key in old_lxr[exg]:
                lxr[exg][key] = old_lxr[exg][key]
            else:
                lxr[exg][key] = None
                pending_assets.append(key)

        pending_assets_lists = _split_list(pending_assets, API_LIMITS)

        if len(pending_assets) != 0:
            print(f"Updating {len(pending_assets)} profiles for: {exg}")
            for pending_assets_list in tqdm(pending_assets_lists):
                assets = LXR_API.query("profile", pending_assets_list)
                assert len(assets) == len(pending_assets_list)
                for asset in assets:
                    code = asset['stockCode']
                    assert code in pending_assets_list
                    lxr[exg][code] = asset
                    lxr[exg][code]['name'] = wt_asset[exg][code]['name']
                    lxr[exg][code]['update_time_profile'] = time

        for key in wt_asset[exg]:
            wt_ = wt_asset[exg][key]['extras']
            lxr_ = lxr[exg][key]
            wt_['update_time_profile'] = lxr_.get('update_time_profile')
            wt_['companyName'] = lxr_.get('companyName')
            wt_['website'] = lxr_.get('website')
            wt_['city'] = lxr_.get('city')
            wt_['province'] = lxr_.get('province')
            wt_['businessScope'] = lxr_.get('businessScope')

    return lxr, wt_asset


def _lxr_industry_file(path: str, wt_asset: Dict):
    state = _check_state(path, "LXR-Industry", 1)
    exgs = cfg_stk.exchg

    if state != 0:  # exist
        old_lxr = load_json(path)
    else:
        old_lxr = {}
        for exg in exgs:
            old_lxr[exg] = {}

    sw21 = _parse_industry()

    lxr = {}
    for exg in exgs:
        lxr[exg] = {}
        pending_assets = []
        for key in wt_asset[exg]:
            # if len(lxr[exg].keys()) > 10: break
            if key in old_lxr[exg]:
                lxr[exg][key] = old_lxr[exg][key]
            else:
                lxr[exg][key] = None
                pending_assets.append(key)

        if len(pending_assets) != 0:
            print(f"Updating {len(pending_assets)} industry for: {exg}")
            for pending_asset in tqdm(pending_assets):
                assets = LXR_API.query("industries", pending_asset)
                lxr[exg][pending_asset] = assets

    lvl = {'one': 0, 'two': 1, 'three': 2}

    for exg in exgs:
        for key in wt_asset[exg]:
            codes = ["", "", ""]
            names = ["", "", ""]
            for item in lxr[exg][key]:
                if item["source"] == "sw_2021":
                    code = item["stockCode"]
                    level, name = sw21[code]
                    level = lvl[level]
                    codes[level] = code
                    names[level] = name
            if "" in codes or "" in names:
                print(
                    f"Err updating industries for {key}:{wt_asset[exg][key]["name"]}")
            wt_asset[exg][key]['extras']['industry_codes'] = codes
            wt_asset[exg][key]['extras']['industry_names'] = names

    return lxr, wt_asset


def _lxr_dividend_file(path: str, wt_asset: Dict):
    """ 分红
    - board_director_plan (董事会预案): Proposed by the board of directors.                                    
    - shareholders_meeting_plan (股东大会预案): Submitted for approval at the shareholders' meeting.           
    - company_plan (公司预案): Internally proposed dividend plan by the company.                               
    - delay_implementation (延迟实施): Dividend payout implementation is delayed.                              
    - cancelled (取消分红): Dividend distribution has been cancelled.                                          
    - implemented (已执行): Dividend plan has been fully executed.                                             
    - terminated (终止): Dividend process has been terminated.                                                 
    - plan (预案): General status indicating that a dividend plan is in place, though it may not be finalized. 

    NOTE: shareholders_meeting(high importance) supervise over the board
          board_director_plan(medium importance) has to be submitted to and reviewed by shareholders_meeting
          company_plan(low importance) may or may not be submitted to shareholders_meeting
          dividends are usually board_director_plan, has to be approved by shareholders_meeting
          some major decisions are done by shareholders_meeting directly (major investment, profit allocations etc.)

    NOTE:
    - adj_factor = adj_dividend * adj_allotment
    - adj_dividend = (1-Cash_Dividend_per_Share/Previous_Close_Price_per_Share)
    - adj_allotment = (1/(1+Bonus_Shares_per_Share))
    """
    return None, None


def _lxr_allotment_file(path: str, wt_asset: Dict):
    """ 配股
    - board_directors_approved (董事会通过): Dividend plan has been approved by the board of directors.
    - shareholders_meeting_approved (股东大会通过): Dividend plan has been approved by the shareholders' meeting.
    - approved (已批准): Dividend plan has received approval.
    - implemented (已执行): Dividend plan has been fully executed.
    - postphoned (已延期): Dividend plan implementation has been postponed.
    - terminated (终止): Dividend process has been terminated.
    - unapproval (未获准): Dividend plan has not been approved.

    NOTE:
    - adj_factor = adj_dividend * adj_allotment
    - adj_dividend = (1-Cash_Dividend_per_Share/Previous_Close_Price_per_Share)
    - adj_allotment = (1/(1+Bonus_Shares_per_Share))
    """
    return None, None


def _wt_adj_factor_file(path: str, wt_asset: Dict):
    state = _check_state(path, "WT-AdjFactor", 1)
    exgs = cfg_stk.exchg
    map = {'SSE': 'sh', 'SZSE': 'sz', 'BJSE': 'bj', }

    if state != 0:  # exist
        old_wt = load_json(path)
    else:
        old_wt = {}
        for exg in exgs:
            old_wt[exg] = {}

    time = datetime.now(timezone(timedelta(hours=8))).isoformat()  # East Asia

    wt = {}
    for exg in exgs:
        wt[exg] = {}
        pending_assets = []
        for key in wt_asset[exg]:
            if key in old_wt[exg] and wt_asset[exg][key]['extras']['update_time_adjfactor'] is not None:
                wt[exg][key] = old_wt[exg][key]
            else:
                wt[exg][key] = None
                pending_assets.append(key)
                wt_asset[exg][key]['extras']['update_time_adjfactor'] = time

        if len(pending_assets) != 0:
            print(f"Updating {len(pending_assets)} adj_factors for: {exg}")
            codes = [f"{map[exg]}.{asset}" for asset in pending_assets]
            adj_factors = BS_API.query_adjust_factor(
                codes, '1990-01-01', '2050-01-01')
            for pending_asset in pending_assets:
                wt[exg][pending_asset] = adj_factors[pending_asset]
    return wt, wt_asset


def _wt_tradedays_holidays_file(tradedays_path: str, holidays_path: str):
    state_h = _check_state(holidays_path, "WT-Holidays", 1)
    state_t = _check_state(tradedays_path, "WT-Tradedays", 1)
    if state_t == 2 and state_h == 2:
        return load_json(tradedays_path), load_json(holidays_path)

    import akshare as ak
    tradedays_df = ak.tool_trade_date_hist_sina()
    # Convert trade_date column to datetime
    tradedays_df['trade_date'] = pd.to_datetime(tradedays_df['trade_date'])
    # Generate the complete range of weekdays
    start_date = tradedays_df['trade_date'].min()
    end_date = tradedays_df['trade_date'].max()
    all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
    # Convert the trade dates to a set for faster operations
    trade_dates_set = set(tradedays_df['trade_date'])
    # Filter out the trade dates to find holidays
    tradedays = sorted([date for date in trade_dates_set])
    holidays = sorted(
        [date for date in all_weekdays if date not in trade_dates_set])
    # Convert holidays list to a DataFrame
    tradedays_df = pd.DataFrame(tradedays, columns=['CHINA'])
    tradedays_df['CHINA'] = tradedays_df['CHINA'].dt.strftime('%Y%m%d')
    holidays_df = pd.DataFrame(holidays, columns=['CHINA'])
    holidays_df['CHINA'] = holidays_df['CHINA'].dt.strftime('%Y%m%d')
    # Create a JSON object with "CHINA" as the key and the formatted dates as a list
    tradedays_json = {"CHINA": tradedays_df['CHINA'].tolist()}
    holidays_json = {"CHINA": holidays_df['CHINA'].tolist()}
    return tradedays_json, holidays_json


def _get_nested_value(d, keys, default=None):
    for key in keys:
        d = d.get(key, default)
        if d == default:
            return default
    return d


def _parse_industry() -> Dict:
    sw21_file = cfg_stk.script_dir + '/info/shenwan2021.json'
    sw21_list = load_json(sw21_file)

    sw21 = {}
    for item in sw21_list:
        # if "delistedDate" not in item.keys():
        code = item["stockCode"]
        level = item["level"]
        name = item["name"]
        sw21[code] = (level, name)
    return sw21


def _split_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def _generate_dt_ranges(start_year: int, end_year: int, interval: int = 10):
    ranges = []
    for year in range(start_year, end_year, interval):
        start_date = f"{year}-01-01"
        end_date = f"{year + interval-1}-12-31"
        ranges.append((start_date, end_date))
    return ranges


def _check_state(file_path: str, file_name: str, days: int = 1) -> int:
    """
    not exist: 0,
    old: 1,
    new: 2,
    """

    if not os.path.exists(file_path):
        print(f'{file_name} not exist')
        return 0

    timestamp_last_update_s = os.path.getmtime(file_path)
    dt = datetime.fromtimestamp(timestamp_last_update_s)
    updated_within_x_day = \
        abs(time.time() - timestamp_last_update_s) <= (3600*24*days)
    if updated_within_x_day:
        print(
            f'{file_name} Already Updated: {dt.year}-{dt.month}-{dt.day}')
        return 2
    else:
        print(
            f'Old {file_name}: {dt.year}-{dt.month}-{dt.day}')
        return 1


# ================================================
# Fundamental DataBase
# ================================================


def _lxr_fundamental_file(path: str, wt_asset: Dict, wt_tradedays: Dict):
    # print('Analyzing/Generating Fundamental database files...')
    print(f"Metric_NpArray(zip):{GREEN}{path}/<symbol>/{DEFAULT}")

    API_LIMITS = 10
    exgs = cfg_stk.exchg

    # generate new meta
    new_codes: List[str] = []
    for exg in exgs:
        for key in wt_asset[exg]:
            new_codes.append(key)
    dates = sorted([int(date) for date in wt_tradedays["CHINA"]])
    dates = [str(date) for date in dates]
    new_dates: Dict[str, List[str]] = {}
    cur_year = None
    # cur_month = None
    for date in dates:
        year = date[:4]
        month = date[4:6]
        day = date[6:8]
        if year != cur_year:
            new_dates[year] = []
            cur_year = year
        # if month != cur_month:
        #     new_dates[year][month] = []
        #     cur_month = month
        new_dates[year].append(month+day)

    new_metrics: List[str] = list(LXR_API.all_metrics)
    # ['pe_ttm', 'd_pe_ttm', 'pb', 'pb_wo_gw', 'ps_ttm', 'pcf_ttm', 'dyr', 'sp', 'spc', 'spa', 'tv', 'ta', 'to_r', 'shn', 'mc', 'mc_om', 'cmc', 'ecmc', 'ecmc_psh', 'fpa', 'fra', 'fb', 'ssa', 'sra', 'sb', 'ha_sh', 'ha_shm', 'mm_nba', 'ev_ebit_r', 'ev_ebitda_r', 'ey', 'pev']
    # commonly missing: ['fpa', 'fra', 'fb', 'ssa', 'sra', 'sb', 'ha_sh', 'ha_shm', 'mm_nba', 'ev_ebit_r', 'ev_ebitda_r', 'pev']

    # dates = pd.to_datetime(wt_tradedays["CHINA"], format='%Y%m%d').sort_values()
    # all_dates: List[datetime] = [
    #     date for date in dates if date < datetime.today()]

    meta = {
        'codes': new_codes,
        'dates': new_dates,
        'metrics': new_metrics,
    }

    meta_path = f"{path}/meta.json"
    if os.path.exists(meta_path):
        # check meta data
        # NOTE: metric, dates information will only be recorded as meta, check they are consistent
        old_meta = load_json(meta_path)
        old_codes: List[str] = old_meta['codes']
        old_dates: Dict[str, List[str]] = old_meta['dates']
        old_years = len(old_dates.keys())
        for idx, (year, month_and_days) in enumerate(old_dates.items()):
            # allow last year to have different dates (will be removed anyway)
            if idx != old_years-1:
                assert month_and_days == meta['dates'][year]
        old_metrics: List[str] = old_meta['metrics']
        assert old_metrics == meta['metrics']

    # numpy is also row major
    template = np.full((365, len(new_metrics)), np.nan, dtype=np.float32)

    # now we are sure that all the meta are the same, fell free to update data
    processed_assets = os.listdir(mkdir(f"{path}/"))

    for exg in exgs:
        pending_assets = []
        for key in wt_asset[exg]:
            if key not in processed_assets:
                pending_assets.append(key)
        if len(pending_assets) != 0:
            print(f"Updating {len(pending_assets)} fundamentals for: {exg}")
            for pending_asset in tqdm(pending_assets):
                fsTableType = wt_asset[exg][pending_asset]['extras']['fsTableType']
                ipoDate = wt_asset[exg][pending_asset]['extras']['ipoDate']
                ipoTime = datetime.fromisoformat(ipoDate)
                ranges = _generate_dt_ranges(
                    ipoTime.year, datetime.today().year, API_LIMITS)
                cur_year = None
                map = {}
                data = np.array([])
                for range in ranges:
                    items = LXR_API.query_fundamental(
                        fsTableType, range[0], range[1], [pending_asset])
                    for item in reversed(items):
                        date_str = item['date']
                        year = str(date_str[:4])
                        month = str(date_str[5:7])
                        day = str(date_str[8:10])
                        if year != cur_year:
                            # save yearly data as npy file
                            if data.size != 0:
                                store_compressed_array(
                                    data, f"{path}/{pending_asset}/{year}.npy")
                            # build new year map
                            map = {s: i for i, s in enumerate(new_dates[year])}
                            n = len(new_dates[year])
                            data = copy.deepcopy(template[:n, :])
                            cur_year = year
                        idx_d = map.get(month+day)
                        for idx_m, metric in enumerate(new_metrics):
                            data[idx_d, idx_m] = item.get(metric)

    return meta


def process_bar_data(wt_assets: Dict, force_sync: bool = False) -> None:
    processor = BarProcessor()
    processor.process_wt_data(wt_assets, force_sync)


class BarProcessor:
    """
    Main class for handling stock data processing operations including:
    - L1 to L2: CSV to DSB conversion
    - L3: DSB file merging
    - L4: DSB resampling
    """

    def __init__(self):
        # Don't initialize WtDataHelper in __init__ since it can't be pickled
        # It will be created in each process as needed
        self.exchange_mapping = {'SSE': 'SH', 'SZSE': 'SZ', 'BJSE': 'BJ'}
        self.reverse_exchange_mapping = {
            v: k for k, v in self.exchange_mapping.items()}
        self.num_processes = max(1, mp.cpu_count() - 1)

    def process_wt_data(self, wt_assets: Dict, force_sync: bool = False) -> None:
        """
        Main entry point for processing WT data files

        Args:
            wt_assets: Dictionary of assets by exchange
            force_sync: Whether to force processing of already processed assets
        """
        # Generate full asset list from configured exchanges
        all_assets = self._generate_asset_list(wt_assets)

        # Step 1: Process CSV to DSB (L1 to L2)
        self._process_l1_to_l2(all_assets, force_sync)

        # Step 2: Process DSB merging and resampling (L3 and L4)
        self._process_l3_and_l4(all_assets)

    def _generate_asset_list(self, wt_assets: Dict) -> List[str]:
        """Generate complete asset list in the format 'symbol.exchange'"""
        assets = []
        for exchange in cfg_stk.exchg:
            for key in wt_assets[exchange]:
                assets.append(f"{key}.{self.exchange_mapping[exchange]}")
        return assets

    def _process_l1_to_l2(self, all_assets: List[str], force_sync: bool) -> None:
        """
        Process CSV files to DSB files (L1 to L2 conversion)

        Args:
            assets: List of assets to process
            force_sync: Whether to force processing even if already processed
        """
        # print('Analyzing/Generating L1(CSV)/L2(DSB) database files...')
        print(
            f"SRC_CSV:            {GREEN}{cfg_stk.STOCK_CSV_DIR}/<year>/<symbol>/{DEFAULT}")
        print(
            f"DB_DSB:             {GREEN}{cfg_stk.STOCK_DB_BAR_DIR}/<symbol>/1m/{DEFAULT}")

        # Determine which assets need processing
        unprocessed_assets = self._get_unprocessed_assets(
            all_assets, force_sync)

        # Process unprocessed assets with multiprocessing
        if unprocessed_assets:
            self._process_assets_parallel(unprocessed_assets)

    def _get_unprocessed_assets(self, assets: List[str], force_sync: bool) -> List[str]:
        """
        Determine which assets need processing

        Args:
            assets: Full list of assets
            force_sync: Whether to force processing

        Returns:
            List of assets that need processing
        """
        src_folder = f"{cfg_stk.STOCK_CSV_DIR}/"
        processed_output_list = \
            os.listdir(mkdir(f"{cfg_stk.STOCK_DB_BAR_DIR}/"))

        valid_input_list = set()
        if os.path.exists(src_folder):
            for src_year in os.listdir(src_folder):
                src_csvs = os.listdir(src_folder + src_year)  # 600000.SH.csv
                for src_csvs in src_csvs:
                    asset = src_csvs[:-4]  # remove .csv
                    valid_input_list.add(asset)

        unprocessed_asset_list = []
        for asset in assets:
            # output lacking and input ready
            if ((asset not in processed_output_list) and (asset in valid_input_list)) or force_sync:
                unprocessed_asset_list.append(asset)

        return unprocessed_asset_list

    def _process_assets_parallel(self, assets: List[str]) -> None:
        """Process multiple assets in parallel using multiprocessing"""
        with Pool(self.num_processes) as pool:
            results = list(tqdm(
                pool.imap(self._process_single_asset, assets),
                total=len(assets),
                desc="Processing CSV to DSB"
            ))

    def _process_single_asset(self, asset: str) -> None:
        """
        Process single asset from CSV to DSB

        Args:
            asset: Asset identifier in format 'symbol.exchange'
        """
        # Create WtDataHelper in the process
        dt_helper = WtDataHelper()

        src_folder = f"{cfg_stk.STOCK_CSV_DIR}/"
        src_years = os.listdir(src_folder)
        db_folder = mkdir(f"{cfg_stk.STOCK_DB_BAR_DIR}/{asset}/1m/")
        # Files in format '{year}.{month}.dsb'

        asset_name = f"{asset}.csv"
        for src_year in src_years:
            # Find matching CSV file for this asset
            src_csvs = os.listdir(src_folder + src_year)
            if asset_name not in src_csvs:
                continue

            src_csv_path = os.path.join(src_folder, src_year, asset_name)
            self._csv_to_dsb(src_csv_path, src_year,
                             db_folder, dt_helper)

    def _csv_to_dsb(self, src_csv_path: str, src_year: str, db_folder: str, dt_helper) -> None:
        """
        Convert CSV data to DSB format

        Args:
            src_csv_path: Path to source CSV file
            src_year: Year of the data
            db_folder: Destination folder for DSB files
            dt_helper: Instance of WtDataHelper
        """
        # Read and format CSV data
        df = pd.read_csv(src_csv_path, header=None, skiprows=1,
                         encoding='utf-8', on_bad_lines='warn')
        df = df.iloc[:, :6]
        df.columns = ['time', 'open', 'high', 'low', 'close', 'vol']

        # Convert time to WT-specific format
        datetime_col = pd.to_datetime(
            df['time'], format='%Y-%m-%d %H:%M:%S', errors='raise')
        df['date'] = datetime_col.dt.strftime('%Y%m%d').astype('int64')
        df['time'] = datetime_col.dt.strftime('%H%M').astype('int')
        df['time'] = (df['date'] - 19900000) * 10000 + \
            df['time']  # WT-specific times
        df = df[['date', 'time', 'open', 'high', 'low',
                 'close', 'vol']].reset_index(drop=True)

        # Group data by month and create DSB files
        dfs_by_month = {month: sub_df for month,
                        sub_df in df.groupby(datetime_col.dt.month)}

        for month, sub_df in dfs_by_month.items():
            BUFFER = WTSBarStruct * len(sub_df)
            buffer = BUFFER()

            db_dsb = f'{db_folder}/{src_year}.{month}.dsb'

            # Convert DataFrame to WTSBarStruct buffer
            self._df_to_buffer(sub_df, buffer)

            # Store buffer to DSB file
            store_path = mkdir(db_dsb)
            dt_helper.store_bars(barFile=store_path,
                                 firstBar=buffer, count=len(sub_df), period="m1")

    def _process_l3_and_l4(self, all_assets: List[str]) -> None:
        """
        Process L3 (merged DSB) and L4 (resampled DSB) database files

        Args:
            assets: List of assets to process
        """
        # print('Analyzing/Generating L3(DSB)/L4(DSB) database files...')
        print(
            f"MERGED_DB_DSB:      {GREEN}{cfg_stk.WT_STORAGE_DIR}/his/min1/<exchange>/<symbol>.dsb{DEFAULT}")
        print(
            f"RESAMPLED_DB_DSB:   {GREEN}{cfg_stk.WT_STORAGE_DIR}/his/{'min'+str(cfg_stk.n)}/<exchange>/<symbol>.dsb{DEFAULT}")

        # Prepare parameters for parallel processing
        unprocessed_asset_list = []
        for asset in all_assets:
            name, exchange = asset.split('.')
            database_db_folder = f"{cfg_stk.STOCK_DB_BAR_DIR}/{asset}/1m/"
            merged_db_path = f'{cfg_stk.WT_STORAGE_DIR}/his/min1/{self.reverse_exchange_mapping[exchange]}/{name}.dsb'
            resampled_db_path = f'{cfg_stk.WT_STORAGE_DIR}/his/{"min"+str(cfg_stk.n)}/{self.reverse_exchange_mapping[exchange]}/{name}.dsb'

            input_valid = os.path.exists(database_db_folder) and len(
                os.listdir(database_db_folder)) != 0
            L3_unprocessed = (not os.path.exists(merged_db_path))
            L4_unprocessed = (not os.path.exists(
                resampled_db_path)) or L3_unprocessed

            if input_valid and (L3_unprocessed or L4_unprocessed):
                unprocessed_asset_list.append(
                    (asset, database_db_folder, merged_db_path, resampled_db_path, L3_unprocessed, L4_unprocessed))

        # Process with multiprocessing
        with Pool(self.num_processes) as pool:
            results = list(tqdm(
                pool.imap(self._process_single_merge_resample,
                          unprocessed_asset_list),
                total=len(unprocessed_asset_list),
                desc=f'Merging and Resampling(x{cfg_stk.n})'
            ))
        

    def _process_single_merge_resample(self, params: Tuple) -> None:
        """
        Process single asset for merging and resampling

        Args:
            params: Tuple containing (asset, database_folder, merged_path, resampled_path)
        """
        # Create WtDataHelper in the process
        dt_helper = WtDataHelper()

        asset, database_db_folder, merged_db_path, resampled_db_path, L3_unprocessed, L4_unprocessed = params

        try:
            if L3_unprocessed:
                # Merge DSB files first
                self._combine_dsb_1m(database_db_folder,
                                     merged_db_path, dt_helper=dt_helper)

            # Resample if necessary
            if L4_unprocessed:
                self._resample(merged_db_path, cfg_stk.n,
                               resampled_db_path, dt_helper=dt_helper)

        except Exception as e:
            print(f'Error processing: {asset}')
            print(e)

    def _combine_dsb_1m(self, database_db_folder: str, merged_db_path: str,
                        begin_date: datetime = datetime(1990, 1, 1),
                        end_date: datetime = datetime(2050, 1, 1),
                        total: bool = True,
                        dt_helper=None) -> None:
        """
        Combine multiple DSB files into a single merged file

        Args:
            database_db_folder: Source folder with DSB files
            merged_db_path: Target path for merged DSB file
            begin_date: Start date for filtering
            end_date: End date for filtering
            total: Whether to read all DSB files or just some
            dt_helper: Instance of WtDataHelper (created if None)
        """
        # Create WtDataHelper if not provided
        if dt_helper is None:
            dt_helper = WtDataHelper()

        dataframes = []

        # Get sorted list of DSB files
        if total:
            sorted_file_list = self._sort_files_by_date(database_db_folder)
        else:
            sorted_file_list = self._sort_files_by_date(
                database_db_folder, begin_date, end_date)

        # Read and combine all DSB files
        for file in sorted_file_list:
            file_path = os.path.join(database_db_folder, file)
            dataframes.append(dt_helper.read_dsb_bars(file_path).to_df())

        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=False)
            self._wt_df_to_dsb(combined_df, mkdir(merged_db_path), dt_helper)

    def _resample(self, src_path: str, times: int, store_path: str, dt_helper=None) -> None:
        """
        Resample DSB data to a different time interval

        Args:
            src_path: Source DSB file path
            times: Resampling factor
            store_path: Target DSB file path
            dt_helper: Instance of WtDataHelper (created if None)
        """
        # Create WtDataHelper if not provided
        if dt_helper is None:
            dt_helper = WtDataHelper()

        # Initialize session manager for resampling
        sess_mgr = SessionMgr()
        sess_mgr.load(f"{cfg_stk.script_dir}/stk_sessions.json")
        session_info = sess_mgr.getSession("SD0930")
        # Perform resampling
        df = dt_helper.resample_bars(
            barFile=src_path,
            period='m1',
            times=times,
            fromTime=199001010000,
            endTime= 205001010000,
            sessInfo=session_info,
            alignSection=False
        ).to_df()

        # Save resampled data
        self._wt_df_to_dsb(df, mkdir(store_path), dt_helper)

    def _sort_files_by_date(self, folder_path: str,
                            start_date: datetime = datetime(1900, 1, 1),
                            end_date: datetime = datetime(2050, 1, 1)) -> List[str]:
        """
        Sort DSB files by date

        Args:
            folder_path: Path to folder containing DSB files
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            Sorted list of filenames
        """
        files = [f for f in os.listdir(folder_path) if f.endswith('.dsb')]
        files_with_dates = []

        for file in files:
            file_date = self._get_file_date(file)
            if start_date < file_date < end_date:
                files_with_dates.append((file, file_date))

        # Sort files by date
        sorted_files = sorted(files_with_dates, key=lambda x: x[1])
        return [file for file, _ in sorted_files]

    def _get_file_date(self, filename: str) -> datetime:
        """
        Extract date from filename

        Args:
            filename: Filename to parse

        Returns:
            Datetime object representing the file date
        """
        base_name = filename[:-4]  # Remove last 4 characters (".dsb")
        # base_name = base_name.split('_')[1]
        # parts = base_name.split('-')
        parts = base_name.split('.')

        if len(parts) == 3:
            year, month, day = parts
            return datetime(int(year), int(month), int(day))
        elif len(parts) == 2:
            year, month = parts
            return datetime(int(year), int(month), 1)  # First day of month
        else:
            return datetime(1900, 1, 1)  # Invalid format

    def _df_to_buffer(self, df: pd.DataFrame, buffer) -> None:
        """
        Convert DataFrame to WTSBarStruct buffer

        Args:
            df: Source DataFrame
            buffer: Target buffer for WTSBarStruct data
        """
        def assign(procession, buffer):
            tuple(map(lambda x: setattr(
                buffer[x[0]], procession.name, x[1]), enumerate(procession)))

        df.apply(assign, buffer=buffer)  # type: ignore

    def _wt_df_to_dsb(self, df: pd.DataFrame, store_path: str, dt_helper=None) -> None:
        """
        Convert DataFrame to DSB format and save

        Args:
            df: Source DataFrame
            store_path: Target path for DSB file
            dt_helper: Instance of WtDataHelper (created if None)
        """
        # Create WtDataHelper if not provided
        if dt_helper is None:
            dt_helper = WtDataHelper()

        # Normalize column names
        df = df.rename(columns={'volume': 'vol'})

        # Format date and time columns
        df['date'] = df['bartime'].astype(str).str[:8].astype(int)
        df['time'] = df['bartime'] - 199000000000

        # Select and order columns
        df = df[['date', 'time', 'open', 'high', 'low',
                 'close', 'vol']].reset_index(drop=True)

        # Create buffer and convert
        BUFFER = WTSBarStruct * len(df)
        buffer = BUFFER()
        self._df_to_buffer(df, buffer)

        # Store to DSB
        dt_helper.store_bars(
            barFile=store_path,
            firstBar=buffer,
            count=len(df),
            period="m1")

# ================================================
# Others
# ================================================
# from wtpy.WtDataDefs import WtNpKline
# from wtpy.wrapper import WtDataHelper
# 
# dtHelper = WtDataHelper()
# def compare_read_dsb_bars():
#     
#     ret:WtNpKline = dtHelper.read_dsb_bars(f"/home/chuyin/work/trader/database/stock/bars/000032.SZ/1m/2021.11.dsb")
#     # ret:WtNpKline = dtHelper.read_dsb_bars(f"{cfg_stk.WT_STORAGE_DIR}/his/min1/SSE/600000.dsb")
#     num_bars = len(ret)
#     print(f"read_dsb_bars {num_bars} bars")
#     print(ret.ndarray[-500:])

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

def time_diff_in_min(start: int, end: int) -> int:
    from datetime import datetime

    def parse_time(time: int) -> datetime:
        time_str = str(time)
        # Extract time components from the last 10 characters of the string
        year = int(time_str[-12:-8])
        month = int(time_str[-8:-6])
        day = int(time_str[-6:-4])
        hour = int(time_str[-4:-2])
        minute = int(time_str[-2:])
        return datetime(year, month, day, hour, minute)
    # Parse both start and end strings into datetime objects
    start_time = parse_time(start)
    end_time = parse_time(end)
    # Calculate the difference in time
    delta = end_time - start_time
    # Convert the time difference to minutes and return it as an integer
    min_diff = int(delta.total_seconds() // 60)
    return min_diff
