import json

class cfg: # Constants Configs
    import os
    TEST = 1
    no_workers = 10
    parallel_mode = 'process' # 'thread'/'process'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = script_dir + '/../../../data_stock'
    INTEGRITY_FILE = script_dir + '/../../database/DB1_0/integrity_table.parquet'
    METADATA_FILE = script_dir + '/../../database/DB1_0/metadata_table.parquet'
    METADATA_JSON_FILE = script_dir + '/../../database/DB1_0/metadata_table.json'
    HOLIDAYS_FILE = script_dir + '/../cfg/misc/holidays.json'
    TRADEDAYS_FILE = script_dir + '/../cfg/misc/tradedays.json'
    STOCKS_FILE = script_dir + '/../cfg/assets_list/stocks.json'
    ADJFACTORS_FILE = script_dir + '/../cfg/misc/adjfactors.json'
    TRADING_HOURS = [(930, 1130), (1300, 1457)] # Augest 2018: modify rules of after-hour-call-auction： 14:57 to 15:00
    # Method to update database path, demonstrating how methods can be included
    @classmethod
    def update_config(cls, new_cfg):
        cls.old_cfg = new_cfg

run_once = 1 # bulk import csv data from folders
run_9am = 0  # daily routine(9am): update asset tables
def database_maintenance():
    import function_datahelper as dhpr
    if run_9am:
        mischelper = dhpr.mischelper()
        mischelper.update_trade_holiday()
        mischelper.update_assetlist()
        mischelper.update_adjfactors()

    if run_once:
        database_helper = dhpr.database_helper()
        database_helper.main_process_assets_from_folders()

if __name__ == "__main__":
    database_maintenance()
    pass