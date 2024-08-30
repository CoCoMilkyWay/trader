from wtpy import BaseCtaStrategy
from wtpy import CtaContext
import numpy as np
import math


class TurtleStra(BaseCtaStrategy):
    def __init__(self, name:str, code:str, barCnt:int, period:str, barN_atr:int, barN_ma:int,k1:float, \
                 k2:float, margin_rate:float, money_pct:float,capital, day1:int, day2:int):
        BaseCtaStrategy.__init__(self, name)

        self.__period__ = period  # 策略运行的时间区间
        self.__bar_cnt__ = barCnt   # 拉取的K线条数
        self.__code__ = code
        self.__margin_rate__ = margin_rate  # 保证金比率
        self.__money_pct__ = money_pct  # 每次使用的资金比率
        self.__capital__ = capital  # 起始资金
        self.__k1__ = k1  # 上轨的系数
        self.__k2__ = k2  # 下轨的系数
        self.__barN_atr__ = barN_atr  # 计算atr时的k线根数
        self.__barN_ma__ = barN_ma   # 计算均线时的k线根数

        self.__day1__ = day1
        self.__day2__ = day2

    def on_init(self, context:CtaContext):
        code = self.__code__    #品种代码

        context.stra_get_bars(code, self.__period__, self.__bar_cnt__, isMain = True)
        context.stra_log_text("Turtle inited")
        pInfo = context.stra_get_comminfo(code)
        self.__volscale__ = pInfo.volscale

    def on_calculate(self, context:CtaContext):
        code = self.__code__    #品种代码
        # 把策略参数读进来，作为临时变量，方便引用
        margin_rate = self.__margin_rate__
        k1 = self.__k1__
        k2 = self.__k2__
        barN_atr = self.__barN_atr__
        barN_ma = self.__barN_ma__
        money_pct = self.__money_pct__
        volscale = self.__volscale__
        capital = self.__capital__
        curPrice = context.stra_get_price(code)
        curPos = context.stra_get_position(code)
        curTime = context.stra_get_time()
        trdUnit_price = volscale * margin_rate * curPrice
        cur_money = capital + context.stra_get_fund_data(0)
        TR_SUM = 0


        #过去day1和day2天的均值
        day1 = self.__day1__
        day2 = self.__day2__



        df_bars = context.stra_get_bars(code, self.__period__, self.__bar_cnt__, isMain = True)
        df_bars_day = context.stra_get_bars(code, 'd1', day2+1, isMain=False)

        closes = df_bars.closes
        highs = df_bars.highs
        lows = df_bars.lows
        opens = df_bars.opens
        highs2 = df_bars_day.highs
        lows2 = df_bars_day.lows
        closes2 = df_bars_day.closes

        ma_high1 = max(highs2[-day1:])
        ma_high2 = max(highs2[-day2:])
        ma_low1 = min(lows2[-day1:])
        ma_low2 = min(lows2[-day2:])

        # 计算之前day1的ATR
        for i in range(day1):
            TR_SUM += max(highs2[-1-i]-lows2[-1-i], highs2[-1-i]-closes2[-2-i], closes2[-2-i]-lows2[-1-i])
        ATR = TR_SUM / day1

        # 计算中轨，为n根bar收盘价的均线
        ma = np.average(closes[-barN_ma:-1])
        up_price = ma + k1 * ATR
        down_price = ma - k2 * ATR

        #计算买入的数量:
        #unit = (0.01 * cur_money) / ATR
        #unit = math.floor(cur_money * money_pct *margin_rate/ trdUnit_price)
        if capital > abs(curPos*curPrice*volscale):
            unit = abs(math.floor((capital-abs(curPos*curPrice)) * money_pct/ curPrice/volscale))
        else:
            unit = 0

        context.stra_log_text("ATR=%.2f  当前净值%.2f仓位%.2f" % (ATR, cur_money / capital, curPos * curPrice * volscale / capital))
        if curPos == 0:
            if curPrice > ma_high1:
                context.stra_enter_long(code, 1 * unit, 'enterlong')
                self.benchmark_price = context.stra_get_price(code)
                context.stra_log_text("向上突破%.2f日最高点：%.2f  多仓进场%.2f手" % (day1, curPrice,unit))
                return
            elif curPrice < ma_low1:
                context.stra_enter_short(code, 1 * unit, 'entershort')
                self.benchmark_price = context.stra_get_price(code)
                context.stra_log_text("向下突破%.2f日最低点：%.2f  空仓进场%.2f手" % (day1, curPrice, unit))
                return

        elif curPos > 0:
            if curPrice > self.benchmark_price + ATR*0.5:
                context.stra_enter_long(code, 1 * unit, 'enterlong')
                self.benchmark_price = context.stra_get_price(code)
                context.stra_log_text("向上突破0.5ATR：%.2f  多头加仓%.2f手" % (curPrice,unit))
                return
            elif curPrice < self.benchmark_price - 2*ATR:
                context.stra_set_position(code, 0, 'clear')
                self.benchmark_price = context.stra_get_price(code)
                #context.stra_exit_long(code, 1 * unit, 'exitlong')
                context.stra_log_text("向下突破2ATR：%.2f  多头平仓%.2f手" % (curPrice,curPos))
                return

        elif curPos < 0:
            if curPrice < self.benchmark_price - ATR*0.5:
                context.stra_enter_short(code, 1 * unit, 'entershort')
                self.benchmark_price = context.stra_get_price(code)
                context.stra_log_text("向下突破0.5ATR：%.2f  空头加仓%.2f手" % (curPrice,unit))
                return
            elif curPrice > self.benchmark_price + ATR*2:
                context.stra_set_position(code, 0, 'clear')
                self.benchmark_price = context.stra_get_price(code)
                #context.stra_exit_short(code, 1 * unit, 'exitshort')
                context.stra_log_text("向上突破2ATR：%.2f  空头平仓%.2f手" % (curPrice,curPos))
                return