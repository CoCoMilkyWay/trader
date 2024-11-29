class cfg_cpt: # Constants Configs
    import os
    max_workers = 128
    concurrency_mode = 'process' # 'thread'/'process'
    
    mode = '/um'  # spot, um, cm
    market = 'Binance'
    
    NUM = 2  # N/None
    symbols = ['BTCUSDT',  'ETHUSDT',  'BNBUSDT',  'DOGEUSDT',  '1000SHIBUSDT',  'ADAUSDT',  'XRPUSDT',  'SOLUSDT',  'LTCUSDT',  'BCHUSDT',  'MATICUSDT',  'LINKUSDT',  'AVAXUSDT',  'DOTUSDT',  'FTMUSDT',  'UNIUSDT',  'AAVEUSDT',  'XLMUSDT',  'TRXUSDT']
    if NUM:
        symbols = [symbol for symbol in symbols if symbol.endswith('USDT')][:NUM]
    CRYPTO_CSV_DIR  = 'd:/data/crypto_csv'  + mode  + '/csv'
    CRYPTO_DB_DIR   = 'd:/data/crypto_db'   + mode  + '/bars'
    # SPOT_DIR    = CRYPTO_CSV_DIR    +   '/spot'
    # UM_DIR      = CRYPTO_CSV_DIR    +   '/um'
    # CM_DIR      = CRYPTO_CSV_DIR    +   '/cm'
    
    script_dir          = os.path.dirname(os.path.abspath(__file__))
    ASSET_FILE          = script_dir + '/cpt_assets.json'
    WT_STORAGE_DIR      = script_dir + '/../../storage'
