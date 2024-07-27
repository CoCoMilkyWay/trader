from xtquant import xtdata
import time
xtdata.reconnect()
code_list = ["688099.SH"]
period = "1d"

if 1:
    for i in code_list:
        xtdata.download_history_data(i,period=period,incrementally=True)
    print('finish downloading')
    xtdata.download_financial_data(code_list) # 下载财务数据到本地
    print('finish downloading financial')
    xtdata.download_sector_data() # 下载板块数据到本地
    # 更多数据的下载方式可以通过数据字典查询
    
    fieldList = ['CAPITALSTRUCTURE.total_capital', '利润表.净利润']
    stockList = ['600000.SH','000001.SZ']
    startDate = '20171209'
    endDate = '20171212'
