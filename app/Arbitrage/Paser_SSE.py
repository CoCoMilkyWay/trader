from pprint import pprint

def Paser_SSE(data_str):
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
    # pprint(data_str)
    # Parse header section (before TAGTAG)
    i = 0
    while i < len(lines) and 'TAGTAG' not in lines[i]:
        line = lines[i].strip()
        if line:
            parts = line.split('=', 1)  # Split on first '=' only
            if len(parts) == 2:
                key, value = parts
                # basic info ==================================================
                if key == 'Fundid1':
                    result['FundID'] = int(value)
                elif key == 'CreationRedemptionUnit':
                    result['CreationRedemptionUnit'] = int(value)
                elif key == 'Recordnum':
                    result['StockNum'] = int(value)
                # pre-trading day =============================================
                elif key == 'PreTradingDay':
                    result['PreTradingDay'] = int(value)
                elif key == 'CashComponent':
                    result['CashComponent'] = float(value)
                elif key == 'NAV':
                    result['NAV'] = float(value)
                elif key == 'NAVperCU':
                    result['NAVperCU'] = float(value)
                # trading day =============================================
                elif key == 'TradingDay':
                    result['TradingDay'] = int(value)
                elif key == 'EstimateCashComponent':
                    result['EstimateCashComponent'] = float(value)
        i += 1

    # Skip the TAGTAG line
    i += 1

    # Parse stocks data
    while i < len(lines) and 'ENDENDEND' not in lines[i]:
        line = lines[i].strip()
        if line:
            # Split by '|' delimiter
            parts = line.split('|')
            if len(parts) >= 7:
                stock = [
                    parts[0].strip(),                                       # code
                    parts[1].strip(),                                       # name
                    int(parts[2].strip()) if parts[2].strip() else 0,       # quantity
                    cash_flag(parts[3].strip()),                            # cash_substitution_flag
                    float(parts[4].strip()) if parts[4].strip() else 0.0,   # subscription_margin_rate
                    float(parts[5].strip()) if parts[5].strip() else 0.0,   # redemption_margin_rate
                    float(parts[6].strip()) if parts[6].strip() else 0.0,   # subscription_amount
                    float(parts[6].strip()) if parts[6].strip() else 0.0,   # redemption_amount
                ]
                stocks.append(stock)
        i += 1
    result['pcf'] = stocks

    # pprint(result, width=200)
    return result

def cash_flag(flag:str):
    # +-------+-------------------------------+
    # | 标志  | 描述
    # +-------+-------------------------------+
    # | 0     | 禁止现金替代
    # | 1     | 允许现金替代
    # | 2     | 沪深市必须现金替代
    # | 3     | 深市退补现金替代
    # | 4     | 深市必须现金替代
    # | 5     | 非沪深市场成份证券退补现金替代
    # | 6     | 非沪深市场成份证券必须现金替代
    # | 7     | 港市退补现金替代
    # | 8     | 港市必须现金替代
    # +-------+-------------------------------+
    
    if flag in ['0']:
        return '0' # 禁止现金替代
    elif flag in ['1']:
        return '1' # 允许现金替代
    elif flag in ['2', '4', '6', '8']:
        return '2' # 必须现金替代
    elif flag in ['3', '5', '7']:
        return '3' # 退补现金替代
    else:
        assert False, f"Invalid cash substitution flag: {flag}"


"""
'AllCashAmount': '',
'AllCashDiscountRate': '',
'AllCashFlag': '',
'AllCashPremiumRate': '',
'CashComponent': '-112.87',
'CreationRedemption': '1',
'CreationRedemptionUnit': '750000',
'EstimateCashComponent': '695.52',
'Fundid1': '513300',
'MaxCashRatio': '1.00000',
'NAV': '1.6662',
'NAVperCU': '1249663.78',
'PreTradingDay': '20250417',
'Publish': '1',
'RTGSFlag': '',
'Recordnum': '101',
'Reserved': '',
'TradingDay': '20250421',
'pcf': [['AAPL', 'AAPL', 75, '5', 0.1, 0.0, 107834.73],... ,] 
"""
