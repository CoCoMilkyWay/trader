import re
from pprint import pprint

def Paser_SZSE(data_str):
    lines = data_str.strip().split('\n')
    result = {
        'FundID': None,
        'CreationRedemptionUnit': None,
        'StockNum': None,
        'PreTradingDay': None,
        'CashComponent': None,
        'NAV': None,
        'NAVperCU': None,
        'TradingDay': None,
        'EstimateCashComponent': None,
        'pcf': [],
    }
    stocks = []
    
    # Parse basic info and other sections
    current_section = None
    i = 0
    
    pre_trading_day = None
    trading_day = None
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and separator lines
        if not line or line.startswith('-----------------------'):
            i += 1
            continue
            
        # Check for section headers
        if "基本信息" in line:
            current_section = "basic_info"
            i += 1
            continue
        elif "信息内容" in line:
            # Extract date from section header if present
            if "日" in line:
                current_section = f"info"
                date = ''.join(re.search(r'(\d{4})-(\d{2})-(\d{2})日', line).groups()) # type: ignore
                if pre_trading_day:
                    trading_day = date
                else:
                    pre_trading_day = date
            elif "组合" in line:
                current_section = f"stocks"
            i += 1
            continue
            
        # Parse current section content
        if current_section in ["basic_info", "info"]:
            if "：" in line:
                parts = line.split('：', 1)  # Split on first '：' only
                if len(parts) == 2:
                    key, value = parts
                    key = key.strip()
                    value = value.strip()
                # basic info ==================================================
                if key == '基金代码':
                    result['FundID'] = int(value)
                elif key == '最小申购、赎回单位':
                    result['CreationRedemptionUnit'] = int(re.search(r'[-\d.]+(?=份)', value).group()) # type: ignore
                elif key == '全部申购赎回组合证券只数':
                    # exclude 159900: 申赎现金
                    result['StockNum'] = int(re.search(r'[-\d.]+(?=只)', value).group()) - 1 # type: ignore
                # pre-trading day =============================================
                elif key == '现金差额':
                    result['CashComponent'] = float(re.search(r'[-\d.]+(?=元)', value).group()) # type: ignore
                elif key == '基金份额净值':
                    result['NAV'] = float(re.search(r'[-\d.]+(?=元)', value).group()) # type: ignore
                elif key == '最小申购、赎回单位资产净值':
                    result['NAVperCU'] = float(re.search(r'[-\d.]+(?=元)', value).group()) # type: ignore
                # trading day =============================================
                elif key == '预估现金差额':
                    result['EstimateCashComponent'] = float(re.search(r'[-\d.]+(?=元)', value).group()) # type: ignore

        # Parse pcf table after header row
        elif current_section == "stocks" and "证券代码" not in line:
            parts = line.split()
            if len(parts) >= 8:  # Ensure we have enough columns
                if '159900' in parts[0]:
                    # Skip 159900: 申赎现金
                    i += 1
                    continue
                stock = [
                    parts[0].strip(),                                   # code
                    parts[1].strip(),                                   # name
                    int(parts[2]) if parts[2].isdigit() else parts[2],  # quantity
                    cash_flag(parts[3].strip()),                        # cash_substitution_flag
                    parse_pct(parts[4].strip()),                        # subscription_margin_rate
                    parse_pct(parts[4].strip()),                        # redemption_margin_rate
                    float(parts[5].replace(',', '')) if parts[6].replace(',', '').replace('.', '').isdigit() else parts[6],  # subscription_amount
                    0.0
                    # float(parts[6].replace(',', '')) if parts[7].replace(',', '').replace('.', '').isdigit() else parts[7],  # redemption_amount
                    # parts[7].strip() if len(parts) > 8 else "",         # market
                    # parts[8].strip() if len(parts) > 9 else "",         # mapping_code
                ]
                stocks.append(stock)
        
        i += 1
    
    result['PreTradingDay'] = int(pre_trading_day) # type: ignore
    result['TradingDay'] = int(trading_day) # type: ignore
    
    result['pcf'] = stocks
    
    # pprint(result, width=200)
    return result

def cash_flag(flag:str):
    if flag in ['禁止']:
        return '0' # 禁止现金替代
    elif flag in ['允许']:
        return '1' # 允许现金替代
    elif flag in ['必须']:
        return '2' # 必须现金替代
    elif flag in ['退补']:
        return '3' # 退补现金替代
    else:
        assert False, f"Invalid cash substitution flag: {flag}"

def parse_pct(s):
    s = s.strip()
    if s.endswith('%'):
        return float(s[:-1]) / 100
    return float(s)

"""
                                                  纳斯达克100指数ETF申购赎回清单
                                               ( 2025-04-21 )

 基本信息
-------------------------------------------------------------------------------------------------------------------------------------------------------
                                        基金名称：  纳斯达克100指数ETF                                                              
                                基金管理公司名称：  大成基金管理有限公司                                                
                                        基金代码：  159513                                                               
                                    目标指数代码：  NDX                                                                    
                                        基金类型：  跨境ETF  
-------------------------------------------------------------------------------------------------------------------------------------------------------

 2025-04-17日 信息内容
-------------------------------------------------------------------------------------------------------------------------------------------------------
                                        现金差额：  -29995.00元                                                              
                      最小申购、赎回单位资产净值：  1109010.02元                                                         
                                    基金份额净值：  1.1090元         
-------------------------------------------------------------------------------------------------------------------------------------------------------

 2025-04-21日 信息内容
-------------------------------------------------------------------------------------------------------------------------------------------------------
                                    预估现金差额：  -29405.46元
                            可以现金替代比例上限：  100.00%
                                是否需要公布IOPV：  是 
                              最小申购、赎回单位：  1000000份  
                        最小申购赎回单位现金红利：  0.00元
                      本市场申购赎回组合证券只数：  1只        
                        全部申购赎回组合证券只数：  102只(含"159900"证券)
                                    是否开放申购：  允许
                                    是否开放赎回：  允许
                        当天净申购的基金份额上限：  不设上限
                        当天净赎回的基金份额上限：  不设上限
            单个证券账户当天净申购的基金份额上限：  不设上限
            单个证券账户当天净赎回的基金份额上限：  不设上限
                    当天累计可申购的基金份额上限：  1000000份
                    当天累计可赎回的基金份额上限：  70000000份
        单个证券账户当天累计可申购的基金份额上限：  不设上限
        单个证券账户当天累计可赎回的基金份额上限：  不设上限
-------------------------------------------------------------------------------------------------------------------------------------------------------

 组合信息内容
-------------------------------------------------------------------------------------------------------------------------------------------------------
 证券代码                             证券简称                                股份数量    现金替代标志    申购现金替代保证金率     赎回现金替代保证金率     申购替代金额     赎回替代金额    挂牌市场  映射代码  是否实物对价申赎
 159900                               申赎现金                                      0         必须                0.00%                                  1,252,257.0000           0.0000     深圳市场                    
 AAPL                                 AAPL                                         69         允许               10.00%                                          0.0000           0.0000     其他市场 
...
"""