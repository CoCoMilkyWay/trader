class cfg_cpt:  # Constants Configs
    import os

    period, n = 'm', 1  # bar period
    start = 202306010000
    end   = 202306100000
    capital = 10000000

    # run_mode(1 in 3):
    normal = False
    train  = False
    stats  = True

    # misc functions:
    analyze = True
    plot    = True
    snoop   = False
    profile = True
    panel   = False
    
    mode = '/um'  # spot, um, cm
    market = 'Binance'

    NUM = 1  # N/None
    symbols = ['BTCUSDT',  '1000SHIBUSDT',  'ETHUSDT',  'BNBUSDT',  'DOGEUSDT',  'ADAUSDT',  'XRPUSDT',  'SOLUSDT',  'LTCUSDT',
               'BCHUSDT',  'MATICUSDT',  'LINKUSDT',  'AVAXUSDT',  'DOTUSDT',  'FTMUSDT',  'UNIUSDT',  'AAVEUSDT',  'XLMUSDT',  'TRXUSDT']
    if NUM:
        symbols = [
            symbol for symbol in symbols if symbol.endswith('USDT')][:NUM]
    CRYPTO_CSV_DIR = 'D:/data/crypto_csv' + mode + '/csv'
    CRYPTO_DB_DIR = 'D:/data/crypto_db' + mode + '/bars'
    # SPOT_DIR    = CRYPTO_CSV_DIR    +   '/spot'
    # UM_DIR      = CRYPTO_CSV_DIR    +   '/um'
    # CM_DIR      = CRYPTO_CSV_DIR    +   '/cm'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ASSET_FILE = script_dir + '/cpt_assets.json'
    WT_STORAGE_DIR = script_dir + '/../../storage'

    # ML model
    model_path = './models/BiLTSM.pkl'

    # indicators
    stats_result = './strategy/CPT_Statistics/stats'
    dump_ind = True if normal else False

    FEE = 0.0015
    NO_FEE = 5
    
