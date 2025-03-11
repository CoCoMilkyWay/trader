# from Chan.ChanConfig import CChanConfig

class cfg_stk:  # Constants Configs
    import os
    
    period, n = 'm', 1  # bar period
    wt_period = period + str(n)
    num = 10  # number of assets (int/None)
    start = 202101010000
    end   = 202103010000
    capital = 10000000
    
    # run_mode(1 in 3):
    # train  = True
    
    # main functions:
    parallel = num >= 30
    profile  = False
    stat     = True
    plot     = False
    analyze  = False
    snoop    = False
    panel    = False
    # 
    mode = '/um'  # spot, um, cm
    # market = 'Binance'
    # 
    STOCK_CSV_DIR = 'D:/data/stock_csv' + mode + '/csv'
    STOCK_DB_DIR = 'D:/data/stock_db' + mode + '/bars'
    # # CM_DIR      = CRYPTO_CSV_DIR    +   '/cm'
    # 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ASSET_FILE = script_dir + '/stk_assets.json'
    WT_STORAGE_DIR = script_dir + '/../../storage'
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
    