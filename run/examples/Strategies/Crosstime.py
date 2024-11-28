import numpy as np
from wtpy import BaseCtaStrategy
from wtpy import CtaContext
import math


class CrosstimeStra(BaseCtaStrategy):

    def __init__(self, name: str, code1: str, code2: str, barCnt: int, period: str, capital: float, margin_rate: float,
                 money_pct: float, isForStk: bool = False):
        BaseCtaStrategy.__init__(self, name)

        self.__period__ = period
        self.__bar_cnt__ = barCnt
        self.__code1__ = code1
        self.__code2__ = code2
        self.__money_pct__ = money_pct
        self.__margin_rate__ = margin_rate
        self.__is_stk__ = isForStk
        self.__capital__ = capital

    def on_init(self, context: CtaContext):
        code1 = self.__code1__  # 品种代码
        code2 = self.__code2__  # 品种代码

        # 这里演示了品种信息获取的接口
        # pInfo = context.stra_get_comminfo(code)
        # print(pInfo)
        context.stra_log_text("Cross inited")
        context.stra_get_bars(code1, self.__period__, self.__bar_cnt__, isMain=False)
        context.stra_get_bars(code2, self.__period__, self.__bar_cnt__, isMain=True)
        pInfo = context.stra_get_comminfo(code1)
        pInfo = context.stra_get_comminfo(code2)
        self.__volscale__ = 100

    def on_calculate(self, context: CtaContext):
        # 把策略参数读进来，作为临时变量，方便引用
        money_pct = self.__money_pct__
        margin_rate = self.__margin_rate__
        trdUnit = 1
        capital = self.__capital__
        volscale = self.__volscale__

        code1 = self.__code1__  # 品种代码
        code2 = self.__code2__  # 品种代码

        df_bars2 = context.stra_get_bars(code2, 'd1', self.__bar_cnt__ + 1, isMain=False)
        df_bars1 = context.stra_get_bars(code1, 'd1', self.__bar_cnt__ + 1, isMain=False)

        df_diff_price = df_bars1.closes - df_bars2.closes
        new_diff_price = df_diff_price[-1]

        ave_price = np.average(df_diff_price[:-2])
        std_price = np.std(df_diff_price[:-2])

        up_price = ave_price + 0.75 * std_price
        down_price = ave_price - 0.75 * std_price

        stop_up_price = ave_price + 2 * std_price
        stop_down_price = ave_price - 2 * std_price

        # 读取当前仓位
        curPos1 = context.stra_get_position(code1)
        curPrice1 = context.stra_get_price(code1)
        curPos2 = context.stra_get_position(code2)
        curPrice2 = context.stra_get_price(code2)
        trdUnit_price1 = volscale * curPrice1 * margin_rate
        trdUnit_price2 = volscale * curPrice2 * margin_rate

        context.stra_log_text('动态权益:%.2f,当前仓位：%.2f  %.2f' \
                              % (context.stra_get_fund_data(1), curPos1, curPos2))

        if curPos1 == 0 and curPos2 == 0:  # 空仓情况下
            if new_diff_price > up_price:
                self.cur_money = capital + context.stra_get_fund_data(0)
                context.stra_enter_long(code1, math.floor(self.cur_money * money_pct / trdUnit_price2),
                                        'enterlong')
                context.stra_enter_short(code2, math.floor(self.cur_money * money_pct / trdUnit_price1),
                                         'entershort')
                context.stra_log_text("价差突破%.2f高点，做多近月做空远月" % (up_price))

            elif new_diff_price < down_price:
                self.cur_money = capital + context.stra_get_fund_data(0)
                context.stra_enter_long(code1, math.floor(self.cur_money * money_pct / trdUnit_price1),
                                        'enterlong')
                context.stra_enter_short(code2, math.floor(self.cur_money * money_pct / trdUnit_price2),
                                         'entershort')
                context.stra_log_text("价差突破%.2f低点，做空近月做多远月" % (up_price))

        else:
            if curPos1 > 0 and new_diff_price < ave_price:
                context.stra_set_position(code2, 0, 'clear')
                context.stra_set_position(code1, 0, 'clear')
                context.stra_log_text("价差多头组合回归均值，平仓")
            elif curPos1 > 0 and new_diff_price > stop_up_price:
                context.stra_set_position(code2, 0, 'clear')
                context.stra_set_position(code1, 0, 'clear')
                context.stra_log_text("达到止盈点，平仓")

            elif curPos2 > 0 and new_diff_price > ave_price:
                context.stra_set_position(code1, 0, 'clear')
                context.stra_set_position(code2, 0, 'clear')
                context.stra_log_text("价差空头组合反向突破，平仓")
            elif curPos2 > 0 and new_diff_price < stop_down_price:
                context.stra_set_position(code1, 0, 'clear')
                context.stra_set_position(code2, 0, 'clear')
                context.stra_log_text("达到止盈点，平仓")
