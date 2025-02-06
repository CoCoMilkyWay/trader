from Chan.ChanConfig import CChanConfig

class cfg_cpt:  # Constants Configs
    import os

    period, n = 'm', 1  # bar period
    num = 70  # number of assets (int/None)
    start = 202405010000
    end   = 202408010000
    capital = 10000000

    # run_mode(1 in 3):
    train  = True
    
    # misc functions:
    profile = False
    analyze = False
    snoop   = False
    panel   = False
    
    mode = '/um'  # spot, um, cm
    market = 'Binance'
    
    CRYPTO_CSV_DIR = 'D:/data/crypto_csv' + mode + '/csv'
    CRYPTO_DB_DIR = 'D:/data/crypto_db' + mode + '/bars'
    # SPOT_DIR    = CRYPTO_CSV_DIR    +   '/spot'
    # UM_DIR      = CRYPTO_CSV_DIR    +   '/um'
    # CM_DIR      = CRYPTO_CSV_DIR    +   '/cm'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ASSET_FILE = script_dir + '/cpt_assets.json'
    WT_STORAGE_DIR = script_dir + '/../../storage'
    ML_MODEL_DIR = script_dir + '/../models'

    # ML model =========================================
    model_path = './models/BiLTSM.pkl'

    # indicators =========================================
    stats_result = './strategy/CPT_Statistics/stats'
    dump_ind = True # if normal else False

    # Stats Analysis =========================================
    analyze_fourier = True
    analyze_bi = True
    analyze_vwma = True
    analyze_longshort = True

    analyze_n_month = 3

    # Fees =========================================
    FEE = 0.0015
    NO_FEE = 5
