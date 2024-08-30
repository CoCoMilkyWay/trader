from wtpy import BaseCtaStrategy
from wtpy import CtaContext
import numpy as np
import math
import pandas as pd

class Dualave(BaseCtaStrategy):
    def __init__(self, name:str, code:str, barCnt:int,
                 period:str, margin_rate:float, money_pct:float, capital, days1:int,days2:int):
        BaseCtaStrategy.__init__(self, name)

        self.__period__ = period
        self.__bar_cnt__ = barCnt
        self.__code__ = code
        self.__margin_rate__ = margin_rate # 保证金比率
        self.__money_pct__ = money_pct # 每次使用的资金比率
        self.__capital__ = capital
        self.__days1__ = days1  # 取的短期均线天数
        self.__days2__ = days2  # 取的长期均线天数

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
        days1 = self.__days1__
        days2 = self.__days2__

        trdUnit_price = volscale * curPrice * margin_rate  # 计算交易一手所选的期货所需的保证金
        #读取当前仓位,价格
        curPos = context.stra_get_position(code)
        curPrice = context.stra_get_price(code)

        df_bars = context.stra_get_bars(code, 'd1', self.__bar_cnt__, isMain=False)

        closes = df_bars.closes

        ma_days1 = [np.average(closes[-days1-i:-i]) for i in range(1, 3)]
        ma_days2 = [np.average(closes[-days2-i:-i]) for i in range(1, 3)]

        context.stra_log_text(
            "当前净值%.2f仓位%.2f" % (context.stra_get_fund_data(1), curPos * curPrice * volscale / capital))
        if curPos == 0:
            if ma_days2[-2] > ma_days1[-2] and ma_days2[-1] <= ma_days1[-1]: # 当前未持仓且日线金叉时直接做多
                # 做多50%资金的手数，为了不使杠杆率过高，总资金乘以一手期货的保证金率再计算手数
                self.cur_money = capital + context.stra_get_fund_data(0)  # 用总盈亏加上初始资金计算出当前总资金
                context.stra_enter_long(code, math.floor(self.cur_money * money_pct *margin_rate/ trdUnit_price)
                                        , 'enterlong')
                self.benchmark_price = context.stra_get_price(code)  # 记录进场时的基准价格
                self.dual_pos = money_pct
                context.stra_log_text("进场基准价格%.2f" % (self.benchmark_price) +
                                      "下一交易日做多%s手" % (math.floor(self.cur_money * money_pct * margin_rate/ trdUnit_price)))  # 输出日志到终端
                self.cur_money = capital + context.stra_get_fund_data(0)  #用总盈亏加上初始资金计算出当前总资金

            elif ma_days2[-2] < ma_days1[-2] and ma_days2[-1] >= ma_days1[-1]:
                self.cur_money = capital + context.stra_get_fund_data(0)  # 用总盈亏加上初始资金计算出当前总资金
                context.stra_enter_short(code, math.floor(self.cur_money * money_pct * margin_rate / trdUnit_price)
                                        , 'entershort')
                self.benchmark_price = context.stra_get_price(code)  # 记录进场时的基准价格
                self.dual_pos = money_pct
                context.stra_log_text("进场基准价格%.2f" % (self.benchmark_price) +
                                      "下一交易日做空%s手" % (math.floor(self.cur_money * money_pct * margin_rate / trdUnit_price)))  # 输出日志到终端

        elif curPos != 0:
            if ma_days2[-2] < ma_days1[-2] and ma_days2[-1] >= ma_days1[-1]:

                self.cur_money = capital + context.stra_get_fund_data(0)  # 用总盈亏加上初始资金计算出当前总资金
                context.stra_set_position(code, 0, 'clear') #平多
                context.stra_enter_short(code, math.floor(self.cur_money * money_pct * margin_rate / trdUnit_price)
                                        , 'entershort')
                self.benchmark_price = context.stra_get_price(code)  # 记录进场时的基准价格
                self.dual_pos = money_pct
                context.stra_log_text("均线交叉，空多交换")
                context.stra_log_text("进场基准价格%.2f" % (self.benchmark_price) +
                                      "下一交易日做空%s手" % (math.floor(self.cur_money * money_pct* margin_rate / trdUnit_price)))  # 输出日志到终端

            elif ma_days2[-2] > ma_days1[-2] and ma_days2[-1] <= ma_days1[-1]: # 当前未持仓且日线金叉时直接做多
                # 做多50%资金的手数，为了不使杠杆率过高，总资金乘以一手期货的保证金率再计算手数
                self.cur_money = capital + context.stra_get_fund_data(0)  # 用总盈亏加上初始资金计算出当前总资金
                context.stra_set_position(code, 0, 'clear')
                context.stra_enter_long(code, math.floor(self.cur_money * money_pct * margin_rate / trdUnit_price)
                                         , 'enterlong')
                self.benchmark_price = context.stra_get_price(code)  # 记录进场时的基准价格
                self.dual_pos = money_pct
                context.stra_log_text("均线交叉，空多交换")
                context.stra_log_text("进场基准价格%.2f" % (self.benchmark_price) +
                                      "下一交易日做多%s手" % (math.floor(self.cur_money * money_pct* margin_rate / trdUnit_price)))  # 输出日志到终端




