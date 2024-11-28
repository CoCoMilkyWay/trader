import pandas as pd
from wtpy import BaseCtaStrategy
from wtpy import CtaContext
import math

class Ichimoku(BaseCtaStrategy):
    def __init__(self, name:str, code:str, barCnt:int, period:str, day1:int,day2:int,day3:int,capital:float,margin_rate:float,money_pct:float,isForStk:bool = False):
        BaseCtaStrategy.__init__(self, name)

        self.__day1__ = day1
        self.__day2__ = day2
        self.__day3__ = day3
        self.__period__ = period
        self.__bar_cnt__ = barCnt
        self.__code__ = code
        self.__money_pct__ = money_pct
        self.__margin_rate__ = margin_rate
        self.__is_stk__ = isForStk
        self.__capital__ = capital

    def on_init(self, context:CtaContext):
        code = self.__code__    #品种代码
        if self.__is_stk__:
            code = code + "Q"
        #这里演示了品种信息获取的接口
        #pInfo = context.stra_get_comminfo(code)
        #print(pInfo)
        context.stra_log_text("Rbreaker inited")
        context.stra_get_bars(code, self.__period__, self.__bar_cnt__, isMain=True)
        pInfo = context.stra_get_comminfo(code)
        self.__volscale__ = pInfo.volscale

    
    def on_calculate(self, context:CtaContext):
        if self.__is_stk__:
            trdUnit = 100

        #读取最近50条1分钟线(dataframe对象)
        theCode = self.__code__
        if self.__is_stk__:
            theCode = theCode + "Q"
        df = context.stra_get_bars(theCode, self.__period__, self.__bar_cnt__, isMain = True)

        #把策略参数读进来，作为临时变量，方便引用
        day1 = self.__day1__
        day2 = self.__day2__
        day3 = self.__day3__
        code = self.__code__    #品种代码

        money_pct = self.__money_pct__
        margin_rate = self.__margin_rate__
        trdUnit = 1
        capital = self.__capital__
        volscale = self.__volscale__

        data = pd.DataFrame([])

        high_9 = pd.DataFrame(df['high']).rolling(window=9).max()
        low_9 = pd.DataFrame(df['low']).rolling(window=9).min()
        data['conversation_line'] = (high_9 + low_9) / 2

        high_26 = pd.DataFrame(df['high']).rolling(window=26).max()
        low_26 = pd.DataFrame(df['low']).rolling(window=26).min()
        data['base_line'] = (high_26 + low_26) / 2

        data['lead_a'] = ((data['conversation_line'] + data['base_line']) / 2).shift(26)

        high_52 = pd.DataFrame(df['high']).rolling(window=52).max()
        low_52 = pd.DataFrame(df['low']).rolling(window=52).min()
        data['lead_b'] = ((high_52 + low_52) / 2).shift(26)

        curPos = context.stra_get_position(code)
        curPrice = context.stra_get_price(code)
        trdUnit_price = volscale * curPrice * margin_rate
        self.cur_money = capital + context.stra_get_fund_data(0)

        if curPos == 0 and curPos * curPrice * volscale / capital < 100:
            if (curPrice > max(data['lead_a'][len(data)-1],data['lead_b'][len(data)-1])) and data['conversation_line'][len(data)-1] > data['base_line'][len(data)-1]:
                context.stra_enter_long(code, math.floor(self.cur_money * money_pct *margin_rate/ trdUnit_price), 'enterlong')
                context.stra_log_text("向上突破%.2f，多仓进场" % (curPrice))
            elif (curPrice < min(data['lead_a'][len(data)-1],data['lead_b'][len(data)-1])) and data['conversation_line'][len(data)-1] < data['base_line'][len(data)-1]:
                context.stra_enter_short(code, math.floor(self.cur_money * money_pct *margin_rate/ trdUnit_price), 'entershort')
                context.stra_log_text("向下突破%.2f，空仓进场" % (curPrice))

        elif curPos > 0:
            if curPrice < data['conversation_line'][len(data)-1]:
                context.stra_set_position(code, 0, 'clear')
                context.stra_log_text("向下突破%.2f，多仓出场" % (curPrice))

        elif curPos < 0:
            if curPrice > data['conversation_line'][len(data)-1]:
                context.stra_set_position(code, 0, 'clear')
                context.stra_log_text("向上突破%.2f，空仓出场" % (curPrice))

        context.stra_log_text(
            "当前权益%.2f仓位%.2f" % (context.stra_get_fund_data(0), curPos * curPrice * volscale / capital))