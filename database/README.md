- Notation:
    - DB0_0: database level 0, number 0

- level 1 database:
    - DB1_0: {1m OHLCV k-bar (stock, etf)}
        - dependency:
            - xt-qmt API / baostock connected
        - maintainance callback function:
            - one-time: bulk import from historical csv data
            - 1m callback in {(9:30-11:30), (13:00-14:57)} (during trading sesstion)
        - integrity table: (day*asset):
            0. bool: integrity-flag
            1. u8: update try times
            2. u8: rules_violated[0:7]
        - integrity rule check:
            0. non-zero/NaN/NaT OHLC
            1. timestamp continuity/order/completeness
            2. intra-day price continuity
            3. inter-day price jump limit (10%, 20%, 30% with call-auction and factor-adj)
            4. OHLC the same from a minute bar if volume is zero
            5. verify day-open/close/mid-break price from other sources(AKshare)
        - meta data json: (asset):
            0. string: asset_name (subject to change: ST->*ST->PT (special treatment/particular transfer))
            1. u32: yearly data integrity[0:31] : {2000, 2001, ..., 2031}
            2. start_date
            3. datetime64[ns]: end_date
            4. datetime64[ns]: first_traded
            5. datetime64[ns]: auto_close_date
            6. string: exchange: SH/SZ
            7. string: sub_exchange:
                - code[:2] == '60'                                     : "SSE.A"    # 沪市主板
                - code[:3] == '900'                                    : "SSE.B"    # 沪市主板
                - code[:2] == '68'                                     : "SSE.STAR" # 沪市二板:科创板(Sci-Tech innovAtion boaRd)
                - code[:3] in ['000', '001']                           : "SZSE.A"   # 深市主板
                - code[:3] == '200'                                    : "SZSE.B"   # 深市主板
                - code[:3] in ['300', '301']                           : "SZSE.SB"  # 深市二板:创业板(second-board)
                - code[:3] in ['002', '003']                           : "SZSE.A"   # 中小板(深市主板)
                - code[:3] in ['440', '430'] or code[:2] in ['83','87']: "NQ"       # 新三板(National Equities Exchange and Quotations)(2021场内改革)
            8. industry sector:(change over years, but only use present year data)
                - u32: 申万一级行业
                - u32: 申万二级行业 (三级不收录)
            9. string: reserved for further info (call_auction/fundamental data flag)

    - DB1_1: {call-auction}
        - maintainance callback function:
            - tick(L1/L2) data in {(9:15-9:25), (14:57-15:00)}

    - DB1_2: {1d fundamental (stock)}
- level 2 database:
    - DB2_0(DB1_0->): {5m, 15m, 1d OHLCV k-bar (stock, etf)}
        - dependency: None

- level 3 data:
    - DB3_0: cross-section
    - DB3_1: tick
    - DB3_2: timestamped-event