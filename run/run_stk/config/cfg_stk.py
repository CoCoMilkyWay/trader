# from Chan.ChanConfig import CChanConfig

class cfg_stk:  # Constants Configs
    import os
    strategy = 'Strategy_Fund'
    # strategy = 'Strategy_Alpha'

    # use m1, m5, m30, m60 (wtpy requires that both 1m and 5m are present)
    period_u, period_l, n = 'min', 'm', 5
    wt_period_u = period_u + str(n)
    wt_period_l = period_l + str(n)
    num = 10000  # number of assets (int/None)
    start = 202401010000 # at least 1 asset has data
    end = 202402010000 # can be in future
    capital = 10000000

    wt_tradedays = 'CHINA'
    wt_session = 'SD0930'

    # run_mode(1 in 3):
    # train  = True

    # main functions:
    profile = False
    stat = False
    plot = False
    analyze = False
    snoop = False
    panel = False
    #
    exchg = ['SSE', 'SZSE', 'BJSE']
    product = 'STK'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    WT_DATABASE_DIR = os.path.abspath(
        script_dir + '/../../../database')  # raw data
    WT_STORAGE_DIR = os.path.abspath(
        script_dir + '/../../storage')  # processed data

    if os.name == 'posix':  # Linux
        STOCK_CSV_DIR = os.path.expanduser("~/work/data/stock_csv")
        STOCK_DB_BAR_DIR = WT_DATABASE_DIR + '/stock/bars'
        STOCK_DB_FUND_DIR = WT_DATABASE_DIR + '/stock/fundamentals'
    else: # Windows
        STOCK_CSV_DIR = 'D:/data/stock_csv'
        STOCK_DB_BAR_DIR = 'D:/data/stock_db/bars'
        STOCK_DB_FUND_DIR = 'D:/data/stock_fundamental'

    # config files
    lxr_asset_file = script_dir + '/lxr_assets.json'
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
    dump_ind = plot  # if normal else False
    #
    # # Stats Analysis =========================================
    # # analyze_fourier = True
    # # analyze_bi = True
    # # analyze_vwma = True
    # # analyze_longshort = True
    # # analyze_n_month = 3
    #
    # MISC =========================================
    FEE = 0.0015
    # memory management (manual set for simplicity)
    max_trade_session_ratio = 250/365*(4.5/24)/n
