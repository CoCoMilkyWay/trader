import os, sys

from Chan.Common.CEnum import DATA_FIELD, KL_TYPE
from Chan.Common.ChanException import CChanException, ErrCode
from Chan.Common.CTime import CTime
from Chan.Common.func_util import str2float
from Chan.KLine.KLine_Unit import CKLine_Unit

from .CommonStockAPI import CCommonStockApi
        
# from wtpy import WtBtEngine, EngineType, WtDtServo
# from wtpy.monitor import WtBtSnooper
from wtpy.wrapper import WtDataHelper
# from wtpy.apps import WtBtAnalyst
# from wtpy.WtCoreDefs import WTSBarStruct
# from wtpy.SessionMgr import SessionMgr

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
        self.dtHelper = WtDataHelper()

    def get_kl_data(self):
        sys.path.append("../../..")
        from run.db.db_cfg import cfg_stk
        from run.db.util_stk import combine_dsb_1m, resample, mkdir
        
        dtHelper = self.dtHelper
        print('Preparing dsb data ...')
        asset = self.code # 'sh.000001'
        asset_dict = {'sh':'SSE', 'sz':'SZSE'}
        parts = asset.split(sep='.')
        exchange = asset_dict[parts[0]]
        code = parts[1]
        # with open(cfg_stk.STOCKS_FILE, 'r', encoding='gbk', errors='ignore') as file:
        #     stocks = json.load(file)
        # try:
        #     type = stocks[exchange][code]['product']
        # except:
        #     type = 'STK'
        # wt_asset = f'{asset_dict[parts[0]]}.{type}.{parts[1]}'
        read_path = f"{cfg_stk.BAR_DIR}/m1/{asset}"
        imcomplete_1m_store_path = mkdir(f"{cfg_stk.WT_STORAGE_DIR}/his/temp_1m.dsb") # temp path
        imcomplete_resample_store_path = mkdir(f"{cfg_stk.WT_STORAGE_DIR}/his/temp_resample.dsb") # temp path
        
        print('Resampling ...')
        from datetime import datetime
        begin_parts = self.begin_date.split('-')
        end_parts = self.end_date.split('-')
        begin_date = datetime(int(begin_parts[0]), int(begin_parts[1]), int(begin_parts[2]))
        end_date = datetime(int(end_parts[0]), int(end_parts[1]), int(end_parts[2]))
        combine_dsb_1m(dtHelper, read_path, imcomplete_1m_store_path, begin_date, end_date, total=False)
        
        if self.k_type == KL_TYPE.K_1M  :
            # resample(dtHelper, imcomplete_1m_store_path, 1,    imcomplete_resample_store_path)
            imcomplete_resample_store_path = imcomplete_1m_store_path
        elif self.k_type == KL_TYPE.K_3M  :
            resample(dtHelper, imcomplete_1m_store_path, 3,    imcomplete_resample_store_path)
        elif self.k_type == KL_TYPE.K_5M  :
            resample(dtHelper, imcomplete_1m_store_path, 5,    imcomplete_resample_store_path)
        elif self.k_type == KL_TYPE.K_15M :
            resample(dtHelper, imcomplete_1m_store_path, 15,   imcomplete_resample_store_path)
        elif self.k_type == KL_TYPE.K_30M :
            resample(dtHelper, imcomplete_1m_store_path, 30,   imcomplete_resample_store_path)
        elif self.k_type == KL_TYPE.K_60M :
            resample(dtHelper, imcomplete_1m_store_path, 60,   imcomplete_resample_store_path)
        elif self.k_type == KL_TYPE.K_DAY :
            resample(dtHelper, imcomplete_1m_store_path, 240,  imcomplete_resample_store_path)
            
        df = dtHelper.read_dsb_bars(imcomplete_resample_store_path).to_df()
        df = df.rename(columns={'bartime': 'time_key'}) # type: ignore
        df = df[["time_key", "open", "high", "low", "close", "volume", "turnover"]] # not include "turnover_rate"
        
        try:
            os.remove(imcomplete_1m_store_path)
            os.remove(imcomplete_resample_store_path)
        except:
            pass
        
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

