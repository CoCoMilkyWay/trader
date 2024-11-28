from wtpy import BaseCtaStrategy
from wtpy import CtaContext
import numpy as np
import math


class ATRStra(BaseCtaStrategy):
    def __init__(self, name:str, code:str, barCnt:int, period:str, barN_atr:int, barN_ma:int,k1:float, \
                 k2:float, margin_rate:float, money_pct:float,capital, cleartimes:list):
        BaseCtaStrategy.__init__(self, name)

        self.__period__ = period  # 策略运行的时间区间
        self.__bar_cnt__ = barCnt   # 拉取的K线条数
        self.__code__ = code
        self.__margin_rate__ = margin_rate  # 保证金比率
        self.__money_pct__ = money_pct  # 每次使用的资金比率
        self.__cleartimes__ = cleartimes  # 尾盘清仓的时间区间
        self.__capital__ = capital  # 起始资金
        self.__k1__ = k1  # 上轨的系数
        self.__k2__ = k2  # 下轨的系数
        self.__barN_atr__ = barN_atr  # 计算atr时的k线根数
        self.__barN_ma__ = barN_ma   # 计算均线时的k线根数

    def on_init(self, context:CtaContext):
        code = self.__code__    #品种代码

        context.stra_get_bars(code, self.__period__, self.__bar_cnt__, isMain = True)
        context.stra_log_text("ATRStra inited")
        pInfo = context.stra_get_comminfo(code)
        self.__volscale__ = pInfo.volscale

    def on_calculate(self, context:CtaContext):
        code = self.__code__    #品种代码

        df_bars = context.stra_get_bars(code, self.__period__, self.__bar_cnt__, isMain = True)
        closes = df_bars.closes
        highs = df_bars.highs
        lows = df_bars.lows
        opens = df_bars.opens

        # 日内策略，尾盘清仓
        curPos = context.stra_get_position(code)
        curPrice = context.stra_get_price(code)
        curTime = context.stra_get_time()
        bCleared = False
        for tmPair in self.__cleartimes__:
            if curTime >= tmPair[0] and curTime <= tmPair[1]:
                if curPos != 0:  # 如果持仓不为0，则要检查尾盘清仓
                    context.stra_set_position(code, 0, "clear")  # 清仓直接设置仓位为0
                    context.stra_log_text("尾盘清仓")
                bCleared = True
                break

        if bCleared:
            return

        # 把策略参数读进来，作为临时变量，方便引用
        margin_rate = self.__margin_rate__
        k1 = self.__k1__
        k2 = self.__k2__
        barN_atr = self.__barN_atr__
        barN_ma = self.__barN_ma__
        money_pct = self.__money_pct__
        volscale = self.__volscale__
        capital = self.__capital__
        trdUnit_price = volscale * margin_rate * curPrice
        cur_money = capital + context.stra_get_fund_data(0)
        TR_SUM = 0
        # 计算之前N个bar的ATR
        for i in range(barN_atr):
            TR_SUM += max(highs[-1-i]-lows[-1-i], highs[-1-i]-closes[-2-i], closes[-2-i]-lows[-1-i])
        ATR = TR_SUM / barN_atr
        # 计算中轨，为n根bar收盘价的均线
        ma = np.average(closes[-barN_ma:-1])
        up_price = ma + k1 * ATR * 3
        down_price = ma - k2 * ATR * 3

        if curPos == 0:
            if curPrice > up_price:
                context.stra_enter_long(code, math.floor(cur_money * money_pct* margin_rate/trdUnit_price), 'enterlong')
                context.stra_log_text('当前价格:%.2f > 由atr计算出的上轨：%.2f，做多%s手' % (curPrice,
                                     up_price, math.floor(cur_money * money_pct * margin_rate/trdUnit_price)))
                return
            if curPrice < down_price:
                context.stra_enter_short(code, math.floor(cur_money * money_pct* margin_rate/trdUnit_price), 'entershort')
                context.stra_log_text('当前价格:%.2f < 由atr计算出的下轨:%.2f,做空%s手'%(curPrice,
                                        down_price, math.floor(cur_money * money_pct* margin_rate/trdUnit_price)))
                return
        elif curPos != 0:
            if curPos > 0 and curPrice < ma:
                context.stra_set_position(code, 0, 'clear')
                context.stra_log_text('当前价格:%.2f < 由均线计算出的中轨:%.2f,平多' % (curPrice, ma,))
                return
            elif curPos < 0 and curPrice > ma:
                context.stra_set_position(code, 0, 'clear')
                context.stra_log_text('当前价格:%.2f > 由均线计算出的中轨:%.2f,平空' % (curPrice, ma,))
                return
