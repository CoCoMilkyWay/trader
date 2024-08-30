from wtpy import BaseCtaStrategy
from wtpy import CtaContext
import numpy as np
import math
import statsmodels.api as sm
import pandas as pd

class RSRS(BaseCtaStrategy):
    def __init__(self, name:str, code:str, barCnt:int,
                 period:str, margin_rate:float, money_pct:float, capital):
        BaseCtaStrategy.__init__(self, name)

        self.__period__ = period
        self.__bar_cnt__ = barCnt
        self.__code__ = code
        self.__margin_rate__ = margin_rate # 保证金比率
        self.__money_pct__ = money_pct # 每次使用的资金比率
        self.__capital__ = capital

    def on_init(self, context:CtaContext):
        code = self.__code__    # 品种代码

        context.stra_get_bars(code, self.__period__, self.__bar_cnt__, isMain = True)

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
        count = self.__bar_cnt__

        trdUnit_price = volscale * curPrice * margin_rate  # 计算交易一手所选的期货所需的保证金
        #读取当前仓位,价格
        curPos = context.stra_get_position(code)
        curPrice = context.stra_get_price(code)

        df_bars = context.stra_get_bars(code, 'd1', count, isMain=False)

        closes = df_bars.closes
        highs = df_bars.highs
        lows = df_bars.lows

        data = pd.DataFrame([])
        data['high'] = highs
        data['low'] = lows

        fit = sm.formula.ols('high~low', data=data).fit()

        value = fit.params['low']
        #context.stra_log_text("value={}".format(value))
        if curPos == 0:
            if value > 1:
                self.cur_money = capital + context.stra_get_fund_data(0)
                context.stra_enter_long(code, math.floor(self.cur_money * money_pct * margin_rate / trdUnit_price)
                                        , 'enterlong')
                context.stra_log_text("斜率%.2f" % (value) +
                                      "做多%s手" % (math.floor(
                    self.cur_money * money_pct * margin_rate / trdUnit_price)))
            elif value < 0.97:
                self.cur_money = capital + context.stra_get_fund_data(0)
                context.stra_enter_short(code, math.floor(self.cur_money * money_pct * margin_rate / trdUnit_price)
                                         , 'entershort')
                context.stra_log_text("斜率%.2f" % (value) +
                                      "做空%s手" % (math.floor(
                    self.cur_money * money_pct * margin_rate / trdUnit_price)))  # 输出日志到终端

        elif curPos > 0:
            if value < 0.95:
                context.stra_set_position(code, 0, 'clear')
                context.stra_log_text("斜率%.2f" % (value) +"多头平仓")
        elif curPos < 0:
            if value > 1:
                context.stra_set_position(code, 0, 'clear')
                context.stra_log_text("斜率%.2f" % (value) +"空头平仓")