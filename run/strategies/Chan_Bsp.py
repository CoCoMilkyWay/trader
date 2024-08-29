import numpy as np
import pandas as pd
import os, sys
from wtpy import BaseCtaStrategy
from wtpy import CtaContext

from Chan.Chan import CChan
from Chan.ChanConfig import CChanConfig
from Chan.Common.CEnum import DATA_SRC, KL_TYPE, AUTYPE
from Chan.DataAPI.wtAPI import parse_time_column
from Chan.KLine.KLine_Unit import CKLine_Unit

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
def stdio(str):
    print(str)
    return str

class Chan_bsp(BaseCtaStrategy):
    
    def __init__(self, name:str, code:str, barCnt:int, period:str, isForStk:bool = False, lv_list = [KL_TYPE.K_DAY]):
        BaseCtaStrategy.__init__(self, name)
        self.__period__ = period
        self.__bar_cnt__ = barCnt
        self.__code__ = code
        
        self.__is_stk__ = isForStk
        self.barnum = 0
        
        self.config = CChanConfig({
            "trigger_step": True,
        })
        self.chan = CChan(
            code=code,
            # begin_time=begin_time,
            # end_time=end_time,
            # data_src=data_src,
            lv_list=lv_list,
            config=self.config,
            # autype=AUTYPE.QFQ,
        )
    
    def on_init(self, context:CtaContext):
        code = self.__code__    #品种代码
        if self.__is_stk__:
            code = code + "-"   # 如果是股票代码，后面加上一个+/-，+表示后复权，-表示前复权
            
        #这里演示了品种信息获取的接口
        #　pInfo = context.stra_get_comminfo(code)
        #　print(pInfo)
        
        context.stra_prepare_bars(code, self.__period__, self.__bar_cnt__, isMain = True)
        context.stra_sub_ticks(code)
        context.stra_log_text("Chan inited")
        
        #读取存储的数据
        self.xxx = context.user_load_data('xxx',1)
        
    def on_tick(self, context: CtaContext, stdCode: str, newTick: dict):
        # print(newTick)
        pass
    
    def on_bar(self,  context: CtaContext, stdCode: str, newTick: dict):
        self.barnum += 1
    
    def on_calculate(self, context:CtaContext):
        code = self.__code__    #品种代码
        
        #读取最近n条线(dataframe对象)
        theCode = code
        if self.__is_stk__:
            theCode = theCode + "-" # 如果是股票代码，后面加上一个+/-，+表示后复权，-表示前复权
        np_bars = context.stra_get_bars(theCode, self.__period__, self.__bar_cnt__, isMain = True)
        
        open = np_bars.opens[-1]
        high = np_bars.highs[-1]
        low = np_bars.lows[-1]
        close = np_bars.closes[-1]
        date = context.get_date()
        
        def create_item_dict(data, column_name):
            for i in range(len(data)):
                if i == 0:
                    data[0] = parse_time_column(str(int(data[0])))
                # data[i] = str2float(data[i])
            return dict(zip(column_name, data))
        
        def parse_time_column(inp):
            # 2020_1102_0931
            if len(inp) == 12:
                year = int(inp[:4])
                month = int(inp[4:6])
                day = int(inp[6:8])
                hour = int(inp[8:10])
                minute = int(inp[10:12])
            else:
                raise Exception(f"unknown time column from csv:{inp}")
            return CTime(year, month, day, hour, minute)
        
        klu = CKLine_Unit(create_item_dict(row_list, self.columns))
        
        self.chan.trigger_load({KL_TYPE.K_DAY: [klu]})  # 喂给CChan新增k线
        bsp_list = chan.get_bsp()
        if not bsp_list:
            continue
        print(open, high, low, close, date)
        
        # date = pd.to_datetime(str(context.get_date()), format="%Y%m%d").date