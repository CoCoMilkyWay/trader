import json
import pyarrow.parquet as pq


class cfg: # Constants Configs
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = script_dir + '/../../../data_stock'
    DB_FILE = script_dir + '/stock_data.parquet'
    INTEGRITY_FILE = script_dir + '/../../database/DB1_0/integrity_table.json'
    METADATA_FILE = script_dir + '/../../database/DB1_0/metadata_table.json'
    HOLIDAYS_FILE = script_dir + '/../cfg/misc/holidays.json'
    STOCKS_FILE = script_dir + '/../cfg/assets_list/stocks.json'
    ADJFACTORS_FILE = script_dir + '/../cfg/misc/adjfactors.json'
    TRADING_HOURS = [(930, 1130), (1300, 1457)] # Augest 2018: modify rules of after-hour-call-auction： 14:57 to 15:00
    # Method to update database path, demonstrating how methods can be included
    @classmethod
    def update_config(cls, new_cfg):
        cls.old_cfg = new_cfg

D1_pre_hour_callback = 0
mannual_bulk_import = 1
def daily_update_metadata():
    # from ctypes import POINTER
    # import datetime
    # import os
    # from wtpy.WtCoreDefs import WTSBarStruct
    import function_datahelper as dhpr
    if D1_pre_hour_callback:
        mischelper = dhpr.mischelper()
        mischelper.update_trade_holiday()
        mischelper.update_assetlist()
        mischelper.update_adjfactors()

    # 下载K线数据
    # hlper.dmpBarsToFile(folder='./', codes=["SZSE.399005","SZSE.399006","SZSE.399303"], period='day')

    # 初始化数据库
    # dbHelper = MysqlHelper("127.0.0.1","root","","test", 5306)
    # dbHelper.initDB()

    # 将数据下载到数据库
    # hlper.dmpBarsToDB(dbHelper, codes=["CFFEX.IF.2103"], period="day")
    # hlper.dmpAdjFactorsToDB(dbHelper, codes=["SSE.600000",'SSE.600001'])

    if mannual_bulk_import:
        # Step 1: Import CSV data
        database_helper = dhpr.database_helper()
        database_helper.import_csv_data()
            # 
        # Step 2: Perform integrity checks
        df = pq.read_table(cfg.DB_FILE).to_pandas()
        integrity_table = dhpr.check_integrity(df)
        with open(cfg.INTEGRITY_FILE, 'w') as f:
            json.dump(integrity_table, f)
            # 
        # Step 3: Generate metadata
        metadata_table = dhpr.get_metadata(df)
        with open(cfg.METADATA_FILE, 'w') as f:
            json.dump(metadata_table, f)

if __name__ == "__main__":
    daily_update_metadata()
    pass