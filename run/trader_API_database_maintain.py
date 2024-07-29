# coding:utf-8
# from xtquant import xtdata
# market, start_time , end_time = 'SZ','20230103','20240103'
# print(xtdata.get_stock_list_in_sector("沪深A股")) # 
# print(xtdata.get_his_st_data('688099.SH'))

from xtquant import xtdata
sector_list = xtdata.get_sector_list()
# SH_trading_calendar = xtdata.get_trading_calendar('SH',start_time='',end_time='')






# 获取沪深A股全部股票的代码
# coding:utf-8
# import pandas as pd
# from xtquant import xtdata
# period = 'snapshotindex'
# stock_list = ['300001.SZ']
# start_time='20240101'; end_time='20240201' #
# xtdata.download_history_data(stock_list[0], period=period, incrementally=True, start_time=start_time, end_time=end_time)
# data = xtdata.get_local_data(field_list=[], stock_list=stock_list, period=period, start_time=start_time, end_time=end_time)
# auction = xtdata.get_market_data_ex(field_list=[], stock_list=stock_list, period=period,start_time=start_time, end_time=end_time)
#
# print(pd.DataFrame(data[stock_list[0]]))
# print(pd.DataFrame(auction[stock_list[0]]))
