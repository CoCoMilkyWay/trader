commodities.json
品种信息表，记录不同交易所中不同品种的相关信息，基本格式如下：(由于json无法使用注释，因此这里采用python的格式写，如果要使用请把注释删除或直接使用demo中的文件)

{
    "CFFEX": {  # 交易所
        "IC": {  # 品种名
            "covermode": 0,  # 平仓类型，表示该品种支持的开平模式
                             # 0=开平，1=开平昨平今，2=平未了结的，3=不区分开平
            "pricemode": 0,  # 价格模式 0=市价限价 1=仅限价 2=仅市价
            "category": 1,  # 分类，参考CTP
                            # 0=股票 1=期货 2=期货期权 3=组合 4=即期
                            # 5=期转现 6=现货期权(股指期权) 7=个股期权(ETF期权)
                            # 20=数币现货 21=数币永续 22=数币期货 23=数币杠杆 24=数币期权
            "trademode": 0,  # 交易模式，0=多空都支持 1=只支持做多 2=只支持做多且T+1
            "precision": 1,  # 价格小数点位数
            "pricetick": 0.2,  # 最小价格变动单位
            "volscale": 200,  # 合约倍数
            "name": "中证",  # 名称
            "exchg": "CFFEX",  # 所属交易所
            "session": "SD0930",  # 交易时间，具体参考session配置文件
            "holiday": "CHINA"  # 节假日，具体参考holiday.json
        },
        "IF": {
            "covermode": 0,
            "pricemode": 0,
            "category": 1,
            "trademode": 0,
            "precision": 1,
            "pricetick": 0.2,
            "volscale": 300,
            "name": "沪深",
            "exchg": "CFFEX",
            "session": "SD0930",
            "holiday": "CHINA"
        },
    }
    "CZCE": {
        "AP": {
            "covermode": 0,
            "pricemode": 0,
            "category": 1,
            "trademode": 0,
            "precision": 0,
            "pricetick": 1.0,
            "volscale": 10,
            "name": "苹果",
            "exchg": "CZCE",
            "session": "FD0900",
            "holiday": "CHINA"
        },
    }
}
stk_comms.json
股票信息表，格式同commodities.json

{
    "SSE" : {
        "STK" : {
            "category" : 0,
            "covermode" : 0,
            "exchg" : "SSE",
            "holiday" : "CHINA",
            "name" : "上证股票",
            "precision" : 2,
            "pricemode" : 1,
            "pricetick" : 0.01,
            "session" : "SD0930",
            "volscale" : 1,
            "trademode": 2
        },
        "IDX" : {
            "category" : 0,
            "covermode" : 0,
            "exchg" : "SSE",
            "holiday" : "CHINA",
            "name" : "上证指数",
            "precision" : 2,
            "pricemode" : 1,
            "pricetick" : 0.01,
            "session" : "SD0930",
            "volscale" : 1
        }
    },
}
sopt_comms.json
股票期权信息表，格式同commodities.json

{
    "SSE": {
        "ETFO": {
            "covermode": 0,
            "pricemode": 0,
            "category": 7,
            "precision": 4,
            "pricetick": 0.0001,
            "volscale": 10000,
            "name": "上证ETF期权",
            "exchg": "SSE",
            "session": "SD0930",
            "holiday": "CHINA"
        }
    },
    "SZSE": {
        "ETFO": {
            "covermode": 0,
            "pricemode": 0,
            "category": 7,
            "precision": 4,
            "pricetick": 0.0001,
            "volscale": 10000,
            "name": "深证ETF期权",
            "exchg": "SZSE",
            "session": "SD0930",
            "holiday": "CHINA"
        }
    }
}
contracts.json
合约信息表，记录合约的具体信息，基本格式如下：

{
    "CFFEX": {
        "IC2108": {
            "name": "中证2108",     # 名称
            "code": "IC2108",       # 代码
            "exchg": "CFFEX",       # 交易所
            "product": "IC",        # 品种
            "maxlimitqty": 20,      # 限价单单笔最大委托数量
            "maxmarketqty": 10      # 市价单单笔最大委托数量
        },
        "IC2109": {
            "name": "中证2109",
            "code": "IC2109",
            "exchg": "CFFEX",
            "product": "IC",
            "maxlimitqty": 20,
            "maxmarketqty": 10
        },
    }
}
stocks.json
股票相关信息，格式同contracts.json

{
    "SSE": {                            # 交易所
        "000001": {                     # 代码
            "code": "000001",           # 代码
            "exchg": "SSE",             # 交易所
            "name": "上证综指",          # 名称
            "product": "IDX"            # 类型
        },
        "603383": {
            "area": "福建",             # 地区
            "code": "603383",           # 代码
            "exchg": "SSE",             # 交易所
            "indust": "软件服务",        # 所属行业
            "name": "顶点软件",          # 名称
            "product": "STK"             
        }
    }
}
stk_options.json
股票期权信息，格式同contracts.json

{
    "SSE": {                            # 交易所
        "10003373": {                   # 代码
            "name": "50ETF购12月3061A", # 名称
            "code": "10003373",         # 代码
            "exchg": "SSE",             # 交易所
            "product": "ETFO",          # 品种类型
            "maxlimitqty": 50,          # 限价单单笔最大委托数量
            "maxmarketqty": 10,         # 市价单单笔最大委托数量
            "option": {
                "optiontype": 49,       # 期权类型类型 49=看涨期权，50=看跌期权
                "underlying": "510050", # 底层品种代码
                "strikeprice": 3.061,   # 行权价格
                "underlyingscale": 1.0  # 底层倍数
            }
        }
    }
}
fee.json
佣金费率配置文件

{
    "CFFEX.IF": {                   # 交易所.品种代码      
        "open":0.000023,            # 开仓费用
        "close":0.000023,           # 平仓费用
        "closetoday":0.000345,      # 平今费用
        "byvolume":false            # 是否根据交易量决定，true表示根据交易量也就是交易笔数决定费率，false表示根据交易额决定费率
    }
}
holidays.json
假期列表

{
    "CHINA": [                      # 地区
        "20080101",                 # 日期，格式为yyyymmdd
        "20080206",
        "20080207",
        "20080208",
        "20080211",
    ]
}
hots.json
主力换月信息

{
 "DCE": {                           # 交易所
    "a": [                          # 品种，后面是每次主力换月信息的列表
            {
                "date": "20100423", # 换月时间
                "from": "a1009",    # 从何主力换月
                "newclose": 3992.0, # 换月后的主力合约的价格
                "oldclse": 3909.0,  # 换月前的主力合约价格
                "to": "a1101"       # 换到何主力
            },
            {
                "date": "20100722",
                "from": "a1101",
                "newclose": 3941.0,
                "oldclse": 3876.0,
                "to": "a1105"
            }
    ]
    }
}
seconds.json
次主力换月信息,格式同hots.json

session.json
交易时间的相关配置

{
    "FN2300":{                      # 交易时段ID
        "name":"期货夜盘2300",       # 交易时段名称 
        "offset": 300,              # 交易时段偏移，对于含有夜盘的交易时段，需要将夜盘偏移到第二天，使得所有交易段位于同一日，300表示偏移3个小时，21点则会偏移到0点，不会影响策略开发，仅底层使用
        "auction":{                 # 集合竞价时段
            "from": 2059,           # 时段开始时间
            "to": 2100              # 时段结束时间
        },
        "sections":[                # 交易时段
            {
                "from": 2100,
                "to": 2300  # 注意与"FN0230"区分，那个是指到次日2:30
            },
            {
                "from": 900,
                "to": 1015
            },
            {
                "from": 1030,
                "to": 1130
            },
            {
                "from": 1330,
                "to": 1500
            }
        ]
    },
    "FD0915":{
        "name":"期货白盘0915",
        "offset": 0,
        "auction":{
            "from": 929,
            "to": 930
        },
        "sections":[
            {
                "from": 930,
                "to": 1130
            },
            {
                "from": 1300,
                "to": 1515
            }
        ]
    }
}