import os, sys

from Common.CEnum import DATA_FIELD, KL_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from Common.func_util import str2float
from KLine.KLine_Unit import CKLine_Unit

from .CommonStockAPI import CCommonStockApi

def create_item_dict(data, column_name):
    for i in range(len(data)):
        if i == 0:
            data[0] = parse_time_column(str(int(data[0])))
        # data[i] = str2float(data[i])
    return dict(zip(column_name, data))

def parse_time_column(inp):
    # 2020_1102_0931  
    if len(inp) == 12:
        year = int(inp[:4])
        month = int(inp[4:6])
        day = int(inp[6:8])
        hour = int(inp[8:10])
        minute = int(inp[10:12])
    else:
        raise Exception(f"unknown time column from csv:{inp}")
    return CTime(year, month, day, hour, minute)

class WT_API(CCommonStockApi):
    def __init__(self, code, k_type=KL_TYPE.K_1M, begin_date=None, end_date=None, autype=None):
        self.headers_exist = True  # 第一行是否是标题，如果是数据，设置为False
        self.columns = [
            DATA_FIELD.FIELD_TIME,
            DATA_FIELD.FIELD_OPEN,
            DATA_FIELD.FIELD_HIGH,
            DATA_FIELD.FIELD_LOW,
            DATA_FIELD.FIELD_CLOSE,
            DATA_FIELD.FIELD_VOLUME,
            # DATA_FIELD.FIELD_TURNOVER,
            # DATA_FIELD.FIELD_TURNRATE,
        ]  # 每一列字段
        self.time_column_idx = self.columns.index(DATA_FIELD.FIELD_TIME)
        super(WT_API, self).__init__(code, k_type, begin_date, end_date, autype)

    def get_kl_data(self):
        sys.path.append("../../..")
        from run.db.run_db_maintain import cfg
        from run.db.util import combine_dsb_1m, resample
        
        # from wtpy import WtBtEngine, EngineType, WtDtServo
        # from wtpy.monitor import WtBtSnooper
        from wtpy.wrapper import WtDataHelper
        # from wtpy.apps import WtBtAnalyst
        # from wtpy.WtCoreDefs import WTSBarStruct
        # from wtpy.SessionMgr import SessionMgr
        dtHelper = WtDataHelper()
        
        print('Preparing dsb data ...')
        asset = self.code # 'sh.000001'
        asset_dict = {'sh':'SSE', 'sz':'SZSE'}
        parts = asset.split(sep='.')
        exchange = asset_dict[parts[0]]
        code = parts[1]
        # with open(cfg.STOCKS_FILE, 'r', encoding='gbk', errors='ignore') as file:
        #     stocks = json.load(file)
        # try:
        #     type = stocks[exchange][code]['product']
        # except:
        #     type = 'STK'
        # wt_asset = f'{asset_dict[parts[0]]}.{type}.{parts[1]}'
        read_path = f"{cfg.BAR_DIR}/m1/{asset}"
        store_path_1m = f"{cfg.WT_STORAGE_DIR}/his/min1/{exchange}/{code}.dsb"
        # store_path_5m = f"{cfg.WT_STORAGE_DIR}/his/min5/{exchange}/{code}.dsb"
        # store_path_60m = f"{cfg.WT_STORAGE_DIR}/his/min60/{exchange}/{code}.dsb"
        # store_path_240m = f"{cfg.WT_STORAGE_DIR}/his/min240/{exchange}/{code}.dsb"
        
        print('Resampling ...')
        from datetime import datetime
        begin_parts = self.begin_date.split('-')
        end_parts = self.end_date.split('-')
        begin_date = datetime(int(begin_parts[0]), int(begin_parts[1]), int(begin_parts[2]))
        end_date = datetime(int(end_parts[0]), int(end_parts[1]), int(end_parts[2]))
        df = combine_dsb_1m(dtHelper, read_path, store_path_1m, begin_date, end_date, store=False)
        df = df.rename(columns={'bartime': 'time_key'})
        df = df[["time_key", "open", "high", "low", "close", "volume", "turnover"]] # not include "turnover_rate"
        
        # resample(dtHelper, store_path_1m, 5, store_path_5m)
        # resample(dtHelper, store_path_1m, 60, store_path_60m)
        # resample(dtHelper, store_path_1m, 240, store_path_240m)

        # if not os.path.exists(file_path):
        #     raise CChanException(f"file not exist: {file_path}", ErrCode.SRC_DATA_NOT_FOUND)
        # if len(data) != len(self.columns):
        #     raise CChanException(f"file format error: {file_path}", ErrCode.SRC_DATA_FORMAT_ERROR)
        
        for row in df.itertuples(index=False, name=None):
            row_list = list(row)
            yield CKLine_Unit(create_item_dict(row_list, self.columns))

    def SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass

