#coding:utf-8
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


from xtquant import xtdata
import time


def my_download(stock_list,period,start_date = '', end_date = ''):
  '''
  用于显示下载进度
  '''
  if "d" in period:
    period = "1d"
  elif "m" in period:
    if int(period[0]) < 5:
      period = "1m"
    else:
      period = "5m"
  elif "tick" == period:
    pass
  else:
    raise KeyboardInterrupt("周期传入错误")


  n = 1
  num = len(stock_list)
  for i in stock_list:
    print(f"当前正在下载{n}/{num}")
    
    xtdata.download_history_data(i,period,start_date, end_date)
    n += 1
  print("下载任务结束")

def do_subscribe_quote(stock_list:list, period:str):
  for i in stock_list:
    xtdata.subscribe_quote(i,period = period)
  time.sleep(1) # 等待订阅完成

if __name__ == "__main__":

  start_date = ""# 格式"YYYYMMDD"，开始下载的日期，date = ""时全量下载
  end_date = "" 
  period = "1d" 

  need_download = 1  # 取数据是空值时，将need_download赋值为1，确保正确下载了历史数据
  
  code_list = ["600000.SH"] # 股票列表

  if need_download: # 判断要不要下载数据, gmd系列函数都是从本地读取历史数据,从服务器订阅获取最新数据
    my_download(code_list, period, start_date, end_date)
  
  ############ 仅获取历史行情 #####################
  data1 = xtdata.get_market_data_ex([],code_list,period = period, start_time = start_date, end_time = end_date, count = -1)

  print(data1[code_list[0]].head())# 行情数据查看



