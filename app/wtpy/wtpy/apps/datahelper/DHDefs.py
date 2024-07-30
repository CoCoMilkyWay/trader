from datetime import datetime

class DBHelper:

    def __init__(self):
        pass

    def initDB(self):
        '''
        初始化数据库，主要是建表等工作
        '''
        pass

    def writeBars(self, bars:list, period="day"):
        '''
        将K线存储到数据库中
        @bars   K线序列
        @period K线周期
        '''
        pass

    def writeFactors(self, factors:dict):
        '''
        将复权因子存储到数据库中
        @factors   复权因子
        '''
        pass


class BaseDataHelper:

    def __init__(self):
        self.isAuthed = False
        pass

    def __check__(self):
        if not self.isAuthed:
            raise Exception("This module has not authorized yet!")

    def auth(self, **kwargs):
        '''
        模块认证
        '''
        pass

    def dmpCodeListToFile(self, filename:str, hasIndex:bool=True, hasStock:bool=True):
        '''
        将代码列表导出到文件
        @filename   要输出的文件名，json格式
        @hasIndex   是否包含指数
        @hasStock   是否包含股票
        '''
        pass

    def dmpAdjFactorsToFile(self, codes:list, filename:str):
        '''
        将除权因子导出到文件
        @codes  股票列表，格式如["SSE.600000","SZSE.000001"]
        @filename   要输出的文件名，json格式
        '''
        pass

    def dmpHolidayssToFile(self, filename_holidays:str, filename_tradedays:str):
        '''
        将节假日导出到数据库(固定使用Akshare API)
        @filename   要输出的文件名，json格式
        '''
        import akshare as ak
        import pandas as pd
        import json
        tradedays_df = ak.tool_trade_date_hist_sina()
        # Convert trade_date column to datetime
        tradedays_df['trade_date'] = pd.to_datetime(tradedays_df['trade_date'])
        # Generate the complete range of weekdays
        start_date = tradedays_df['trade_date'].min()
        end_date = tradedays_df['trade_date'].max()
        all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
        # Convert the trade dates to a set for faster operations
        trade_dates_set = set(tradedays_df['trade_date'])
        # Filter out the trade dates to find holidays
        holidays = sorted([date for date in all_weekdays if date not in trade_dates_set])
        tradedays = sorted([date for date in trade_dates_set])
        # Convert holidays list to a DataFrame
        holidays_df = pd.DataFrame(holidays, columns=['CHINA'])
        holidays_df['CHINA'] = holidays_df['CHINA'].dt.strftime('%Y%m%d')
        tradedays_df = pd.DataFrame(tradedays, columns=['CHINA'])
        tradedays_df['CHINA'] = tradedays_df['CHINA'].dt.strftime('%Y%m%d')
        # Create a JSON object with "CHINA" as the key and the formatted dates as a list
        holidays_json_object = {
            "CHINA": holidays_df['CHINA'].tolist()
        }
        tradedays_object = {
            "CHINA": tradedays_df['CHINA'].tolist()
        }
        # Convert the dictionary to a JSON string
        holidays_json_string = json.dumps(holidays_json_object, indent=4)
        tradedays_json_string = json.dumps(tradedays_object, indent=4)
        # Save to a file
        with open(filename_holidays, 'w') as file:
            file.write(holidays_json_string)
        with open(filename_tradedays, 'w') as file:
            file.write(tradedays_json_string)
            
    def dmpBarsToFile(self, folder:str, codes:list, start_date:datetime=None, end_date:datetime=None, period="day"):
        '''
        将K线导出到指定的目录下的csv文件，文件名格式如SSE.600000_d.csv
        @folder 要输出的文件夹
        @codes  股票列表，格式如["SSE.600000","SZSE.000001"]
        @start_date 开始日期，datetime类型，传None则自动设置为1990-01-01
        @end_date   结束日期，datetime类型，传None则自动设置为当前日期
        @period K线周期，支持day、min1、min5
        '''
        pass

    def dmpAdjFactorsToDB(self, dbHelper:DBHelper, codes:list):
        '''
        将除权因子导出到数据库
        @codes  股票列表，格式如["SSE.600000","SZSE.000001"]
        @dbHelper   数据库辅助模块
        '''
        pass
    
    def dmpBarsToDB(self, dbHelper:DBHelper, codes:list, start_date:datetime=None, end_date:datetime=None, period:str="day"):
        '''
        将K线导出到数据库
        @dbHelper 数据库辅助模块
        @codes  股票列表，格式如["SSE.600000","SZSE.000001"]
        @start_date 开始日期，datetime类型，传None则自动设置为1990-01-01
        @end_date   结束日期，datetime类型，传None则自动设置为当前日期
        @period K线周期，支持day、min1、min5
        '''
        pass

    def dmpBars(self, codes:list, cb, start_date:datetime=None, end_date:datetime=None, period:str="day"):
        '''
        将K线导出到指定的目录下的csv文件，文件名格式如SSE.600000_d.csv
        @cb     回调函数，格式如cb(exchg:str, code:str, firstBar:POINTER(WTSBarStruct), count:int, period:str)
        @codes  股票列表，格式如["SSE.600000","SZSE.000001"]
        @start_date 开始日期，datetime类型，传None则自动设置为1990-01-01
        @end_date   结束日期，datetime类型，传None则自动设置为当前日期
        @period K线周期，支持day、min1、min5
        '''
        pass