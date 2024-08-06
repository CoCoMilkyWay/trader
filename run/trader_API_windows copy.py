from xtquant import xtdata
import time
xtdata.reconnect()
code = "688089.SH"
code_list = [code]
period = "1m"

if 1:
    def on_progress(data):
        print(data)
        # {'finished': 1, 'total': 50, 'stockcode': '000001.SZ', 'message': ''}
    xtdata.download_history_data2(code_list, period, start_time='20010101093000', end_time='', callback=on_progress)
    data = xtdata.get_market_data([], code_list, period='1m', start_time='20230701')
    print('一次性取数据', data)

    print('finish downloading history')
    # xtdata.download_financial_data(code_list); print('finish downloading financial')# 下载财务数据到本地
    # xtdata.download_sector_data(); print('finish downloading sector')# 下载板块数据到本地
    instrument_detail = xtdata.get_instrument_detail(code, iscomplete=False); print('finish downloading instrument')
    print(instrument_detail)
    fieldList = ['CAPITALSTRUCTURE.total_capital', '利润表.净利润']
    stockList = ['600000.SH','000001.SZ']
    startDate = '20171209'
    endDate = '20171212'
