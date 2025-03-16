# from Chan.ChanConfig import CChanConfig

class cfg_stk:  # Constants Configs
    import os
    
    period, n = 'm', 1  # bar period
    wt_period = period + str(n)
    num = 1  # number of assets (int/None)
    start = 202101010930
    end   = 202101050930
    capital = 10000000
    
    # run_mode(1 in 3):
    # train  = True
    
    # main functions:
    parallel = num >= 30
    profile  = False
    stat     = False
    plot     = False
    analyze  = False
    snoop    = False
    panel    = False
    # 
    exchg = ['SSE', 'SZSE', 'BJSE']
    product = 'STK'
    days_policy = 'CHINA'
    session_policy = 'SD0930'
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    WT_DATABASE_DIR = os.path.abspath(script_dir + '/../../../database') # raw data
    WT_STORAGE_DIR = os.path.abspath(script_dir + '/../../storage') # processed data
    # STOCK_CSV_DIR = 'E:/raw_m1/stk'
    # STOCK_CSV_DIR = 'D:/data/stock_csv' + product + '/csv'
    # STOCK_DB_DIR = 'D:/data/stock_db' + product + '/bars'
    STOCK_CSV_DIR = os.path.expanduser("~/work/data/stock_csv")
    STOCK_DB_BAR_DIR = WT_DATABASE_DIR + '/stock/bars'
    STOCK_DB_FUND_DIR = WT_DATABASE_DIR + '/stock/fundamentals'
    # 
    # config files
    wt_asset_file = script_dir + '/stk_assets.json'
    lxr_profile_file = script_dir + '/info/lxr_profile.json'
    lxr_industry_file = script_dir + '/info/lxr_industry.json'
    wt_adj_factor_file = script_dir + '/stk_adjfactors.json'
    wt_tradedays_file = script_dir + '/stk_tradedays.json'
    wt_holidays_file = script_dir + '/stk_holidays.json'
    # ML_MODEL_DIR = script_dir + '/../models'
    # 
    # # ML model =========================================
    # model_path = './models/BiLTSM.pkl'
    # 
    # indicators =========================================
    # stats_result = './strategy/CPT_Statistics/stats'
    dump_ind = plot # if normal else False
    # 
    # # Stats Analysis =========================================
    # # analyze_fourier = True
    # # analyze_bi = True
    # # analyze_vwma = True
    # # analyze_longshort = True
    # # analyze_n_month = 3
    # 
    # Fees =========================================
    FEE = 0.0015
    # NO_FEE = 5
    