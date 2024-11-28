from wtpy import BaseCtaStrategy
from wtpy import CtaContext
import numpy as np
import math

class BollingStra(BaseCtaStrategy):
    
    def __init__(self, name:str, code:str, barCnt:int,
                 period:str, margin_rate:float, money_pct:float, capital, k:float, days:int,type:str):
        BaseCtaStrategy.__init__(self, name)

        self.__period__ = period
        self.__bar_cnt__ = barCnt
        self.__code__ = code
        self.__margin_rate__ = margin_rate # 保证金比率
        self.__money_pct__ = money_pct # 每次使用的资金比率
        self.__capital__ = capital
        self.__k__ = k  # n日均线加上k倍指标作为上轨和下轨 (AVG时，k是百分数如0.01)
        self.__days__ = days  # 取的均线天数
        self.__type__ = type  # 策略种类，有三种，AVG均线策略，BOLL布林带，ATR平均真实波幅

    def on_init(self, context:CtaContext):
        code = self.__code__    # 品种代码
        type = self.__type__
        context.stra_get_bars(code, 'd1', self.__bar_cnt__, isMain=False)
        context.stra_get_bars(code, self.__period__, self.__bar_cnt__, isMain = True)
        context.stra_log_text("%sStra inited" % (type))
        pInfo = context.stra_get_comminfo(code)
        self.__volscale__ = pInfo.volscale


    def on_session_begin(self, context:CtaContext, curTDate:int):
        self.trade_next_day = 2

    def on_calculate(self, context:CtaContext):
        code = self.__code__    #品种代码
        # 把策略参数读进来，作为临时变量，方便引用
        curPrice = context.stra_get_price(code)
        margin_rate = self.__margin_rate__
        money_pct = self.__money_pct__
        volscale = self.__volscale__
        capital = self.__capital__
        days = self.__days__
        k = self.__k__
        type = self.__type__
        trdUnit_price = volscale * margin_rate * curPrice
        curPos = context.stra_get_position(code)
        if curPos == 0:
            self.cur_money = capital + context.stra_get_fund_data(0)

        df_bars = context.stra_get_bars(code, 'd1', self.__bar_cnt__, isMain=False)
        closes = df_bars.closes
        ma_days = np.average(closes[-days:-1])
        if type not in ['AVG', 'BOLL', 'ATR']:
            print('输入的type错误,应输入AVG,BOLL,ATR中的一个')
            exit()
        if type == 'AVG':
            up_price = ma_days * (1+k)
            down_price = ma_days * (1-k)
        elif type == 'BOLL':
            std = np.std(closes[-days:-1])
            up_price = ma_days + k * std
            down_price = ma_days - k * std
        elif type == 'ATR':
            highs = df_bars.highs
            lows = df_bars.lows
            closes = df_bars.closes
            TR_SUM = 0
            for i in range(days):
                TR_SUM += max(highs[-1 - i] - lows[-1 - i], highs[-1 - i] - closes[-2 - i],
                              closes[-2 - i] - lows[-1 - i])
            ATR = TR_SUM / days
            up_price = ma_days + k * ATR
            down_price = ma_days - k * ATR
        # 获取昨日收盘价
        if curPrice > up_price and curPos == 0:
            self.trade_next_day = 1
        elif curPrice < down_price and curPos == 0:
            self.trade_next_day = -1
        elif curPrice < ma_days and curPos > 0:
            context.stra_set_position(code, 0, 'clear')
            context.stra_log_text('收盘价回复到均线位置，平仓')
        elif curPrice > ma_days and curPos < 0:
            context.stra_set_position(code, 0, 'clear')
            context.stra_log_text('收盘价回复到均线位置，平仓')
        curTime = context.stra_get_time()
        cur_money = capital + context.stra_get_fund_data(code)
        if cur_money < self.cur_money * 0.99 and curPos != 0:
            self.trade_next_day = 0
        if curTime >= 1455 and curTime <= 1500:
            if self.trade_next_day == 1:
                context.stra_enter_long(code,math.floor(self.cur_money*money_pct/trdUnit_price)
                                        ,'enterlong')
                self.cur_money = capital + context.stra_get_fund_data(0)
                context.stra_log_text('下一交易日做多%s手'% (math.floor(self.cur_money*money_pct/trdUnit_price)))
            elif self.trade_next_day == -1:
                context.stra_enter_short(code,math.floor(self.cur_money*money_pct/trdUnit_price)
                                        ,'entershort')
                self.cur_money = capital + context.stra_get_fund_data(0)
                context.stra_log_text('下一交易日做空%s手'% (math.floor(self.cur_money*money_pct/trdUnit_price)))
            elif self.trade_next_day == 0:
                context.stra_set_position(code, 0, 'clear')
                context.stra_log_text('亏损超过百分之一，下一交易日平仓')
