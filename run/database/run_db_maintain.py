class cfg: # Constants Configs
    import os
    TEST = 1
    max_workers = 96
    concurrency_mode = 'process' # 'thread'/'process'

    # local config files
    script_dir          = os.path.dirname(os.path.abspath(__file__))
    HOLIDAYS_FILE       = script_dir + '/../cfg/misc/holidays.json'
    TRADEDAYS_FILE      = script_dir + '/../cfg/misc/tradedays.json'
    STOCKS_FILE         = script_dir + '/../cfg/assets_list/stocks.json'
    ADJFACTORS_FILE     = script_dir + '/../cfg/misc/adjfactors.json'

    # local/remote database
    USE_REMOTE          = 1 # local storage limited
    LOCAL_DATABASE_DIR  = script_dir + '/../../database'
    REMOTE_DATABASE_DIR = 'E:/DB'
    DATABASE_DIR        = REMOTE_DATABASE_DIR if USE_REMOTE else LOCAL_DATABASE_DIR
    # db1_0
    DB1_0_name          = '/DB1_0'
    DB1_0               = DATABASE_DIR + DB1_0_name
    INTEGRITY_FILE      = DB1_0 + '/integrity_table.parquet'
    METADATA_FILE       = DB1_0 + '/metadata_table.parquet'
    METADATA_JSON_FILE  = LOCAL_DATABASE_DIR + DB1_0_name + '/metadata_table.json'
    BAR_DIR             = DB1_0 + '/bars'
    TICK_DIR            = DB1_0 + '/tick'
    CROSS_VERIFY_DIR    = BAR_DIR + '/third_party/tdx_1d'
    # remote raw data
    RAW_CSV_DIR = 'E:/raw_m1' # 'E:/data_stock'
    
    # 0. non-zero/NaN/NaT OHLC
    # 1. timestamp continuity/order/completeness
    # 2. intra-day price continuity
    # 3. inter-day price jump limit (10%~30% with call-auction and adj)
    # 4. OHLC the same from a minute bar if volume is zero
    # 5. verify day-open/close/mid-break price from other sources
    CHECK_0 = True
    CHECK_1 = True
    CHECK_2 = True
    CHECK_3 = True
    CHECK_4 = True
    CHECK_5 = True
    CHECK_6 = True
    CHECK_7 = True
    
    # Method to update database path, demonstrating how methods can be included
    @classmethod
    def update_config(cls, new_cfg):
        cls.old_cfg = new_cfg

run_once = 1 # bulk import csv data from folders
run_9am = 0  # daily routine(9am): update asset tables
run_db_purge = 0 # WARNING： Consequences!!！
def database_maintenance():
    import function_db_maintain as dhpr
    update_helper = dhpr.update_helper()
    
    if run_9am:
        update_helper.update_trade_holiday()
        update_helper.update_assetlist()
        update_helper.update_adjfactors()
        
    if run_db_purge:
        import shutil
        import os
        from tqdm import tqdm
        print("Purging DB !!!!!!!!!!!!!!")
        rm_path = cfg.BAR_DIR + '/m1'
        for folder in tqdm(os.listdir(rm_path)):
            shutil.rmtree(rm_path + '/' + folder)        

    if run_once:
        database_helper = dhpr.database_helper()
        database_helper.main_process_assets_from_folders()

if __name__ == "__main__":
    database_maintenance()
    pass