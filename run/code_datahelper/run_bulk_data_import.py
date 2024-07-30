import json

class cfg: # Constants Configs
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = script_dir + '/../../../data_stock'
    INTEGRITY_FILE = script_dir + '/../../database/DB1_0/integrity_table.parquet'
    METADATA_FILE = script_dir + '/../../database/DB1_0/metadata_table.parquet'
    HOLIDAYS_FILE = script_dir + '/../cfg/misc/holidays.json'
    TRADEDAYS_FILE = script_dir + '/../cfg/misc/tradedays.json'
    STOCKS_FILE = script_dir + '/../cfg/assets_list/stocks.json'
    ADJFACTORS_FILE = script_dir + '/../cfg/misc/adjfactors.json'
    TRADING_HOURS = [(930, 1130), (1300, 1457)] # Augest 2018: modify rules of after-hour-call-auction： 14:57 to 15:00
    # Method to update database path, demonstrating how methods can be included
    @classmethod
    def update_config(cls, new_cfg):
        cls.old_cfg = new_cfg

perDAY_1hour_before_trade_start = 0
Once_bulk_import = 1
def daily_update_metadata():
    # from ctypes import POINTER
    # import datetime
    # import os
    # from wtpy.WtCoreDefs import WTSBarStruct

    import function_datahelper as dhpr
    if perDAY_1hour_before_trade_start:
        mischelper = dhpr.mischelper()
        mischelper.update_trade_holiday()
        mischelper.update_assetlist()
        mischelper.update_adjfactors()

    # hlper.dmpBarsToFile(folder='./', codes=["SZSE.399005","SZSE.399006","SZSE.399303"], period='day')

    # dbHelper = MysqlHelper("127.0.0.1","root","","test", 5306)
    # dbHelper.initDB()

    # hlper.dmpBarsToDB(dbHelper, codes=["CFFEX.IF.2103"], period="day")
    # hlper.dmpAdjFactorsToDB(dbHelper, codes=["SSE.600000",'SSE.600001'])

    if Once_bulk_import:
        # load misc info for market

        # parallel processing asset's minute k-bar by year
        database_helper = dhpr.database_helper()
        database_helper.process_assets_from_folders()

if __name__ == "__main__":
    daily_update_metadata()
    pass