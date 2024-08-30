from wtpy import BaseCtaStrategy
from wtpy import CtaContext
import numpy as np
import math


class GridStra(BaseCtaStrategy):
    
    def __init__(self, name:str, code:str, barCnt:int, period:str, short_days:int, long_days:int, num:int, p1:float, p2:float, capital, margin_rate= 0.1,stop_loss = 0.8):
        BaseCtaStrategy.__init__(self, name)

        self.__short_days__ = short_days  # 计算短时均线时使用的天数
        self.__long_days__ = long_days  # 计算长时均线时使用的天数
        self.__num__ = num  # 单边网格格数
        self.__p1__ = p1    # 上涨时每格相比基准格的涨幅
        self.__p2__ = p2    # 下跌时每格相比基准格的跌幅(<0)
        self.__period__ = period    # 交易k线的时间级，如5分钟，1分钟
        self.__bar_cnt__ = barCnt   # 拉取的bar的次数
        self.__code__ = code        # 策略实例名称
        self.__capital__ = capital  # 起始资金
        self.__margin_rate__ = margin_rate  # 保证金率
        self.__stop_loss__ = stop_loss  # 止损系数，止损点算法为网格最低格的价格*stop_loss


    def on_init(self, context:CtaContext):
        code = self.__code__    # 品种代码

        context.stra_get_bars(code, 'd1', self.__bar_cnt__, isMain=False)  # 在策略初始化阶段调用一次后面要拉取的K线
        context.stra_get_bars(code, self.__period__, self.__bar_cnt__, isMain=True)
        context.stra_log_text("GridStra inited")

        pInfo = context.stra_get_comminfo(code)  # 调用接口读取品类相关数据
        self.__volscale__ = pInfo.volscale
        # 生成网格交易每格的边界以及每格的持仓比例
        self.__price_list__ = [1]
        self.__position_list__ = [0.5]
        num = self.__num__
        p1 = self.__p1__
        p2 = self.__p2__
        for i in range(num):
            self.__price_list__.append(1+(i+1)*p1)
            self.__price_list__.append(1+(i+1)*p2)
            self.__position_list__.append(0.5+(i+1)*0.5/num)
            self.__position_list__.append(0.5-(i+1)*0.5/num)
        self.__price_list__.sort()
        self.__position_list__.sort(reverse=True)

        print(self.__price_list__)
        print(self.__position_list__)


    def on_calculate(self, context:CtaContext):
        code = self.__code__    #品种代码

        #把策略参数读进来，作为临时变量，方便引用
        margin_rate = self.__margin_rate__
        price_list = self.__price_list__
        position_list = self.__position_list__
        capital = self.__capital__
        volscale = self.__volscale__
        stop_loss = self.__stop_loss__
        short_days = self.__short_days__
        long_days = self.__long_days__
        theCode = code

        # 读取日线数据以计算均线的金叉
        df_bars = context.stra_get_bars(theCode, 'd1', self.__bar_cnt__, isMain=False)
        closes = df_bars.closes
        ma_short_days1 = np.average(closes[-short_days:-1])
        ma_long_days1 = np.average(closes[-long_days:-1])
        ma_short_days2 = np.average(closes[-short_days - 1:-2])
        ma_long_days2 = np.average(closes[-long_days - 1:-2])
        # 读取最近50条5分钟线
        context.stra_get_bars(theCode, self.__period__, self.__bar_cnt__, isMain=True)

        #读取当前仓位,价格
        curPos = context.stra_get_position(code)
        curPrice = context.stra_get_price(code)
        if curPos == 0:
            self.cur_money = context.stra_get_fund_data(0) + capital  # 当没有持仓时，用总盈亏加上初始资金计算出当前总资金

        trdUnit_price = volscale * curPrice * margin_rate  # 计算交易一手所选的期货所需的保证金

        if curPos == 0:
            if (ma_short_days1 > ma_long_days1) and (ma_long_days2 > ma_short_days2):  # 当前未持仓且日线金叉时作为基准价格进场
                # 做多50%资金的手数，为了不使杠杆率过高，总资金乘以一手期货的保证金率再计算手数
                context.stra_enter_long(code, math.floor(self.cur_money*margin_rate*0.5/trdUnit_price), 'enterlong')
                self.benchmark_price = context.stra_get_price(code)  # 记录进场时的基准价格
                self.grid_pos = 0.5  # 记录grid_pos表示网格当前实际仓位
                context.stra_log_text("进场基准价格%.2f" % (self.benchmark_price))  # 输出日志到终端
                return

        elif curPos != 0:
            #  当前有持仓时，先根据当前价格计算出交易后需要达到的目标仓位target_pos
            for i in range(len(price_list)-1):
                if (price_list[i] <= (curPrice / self.benchmark_price)) & ((curPrice / self.benchmark_price) < price_list[i+1]):
                    #  当前价格处于上涨或下跌区间时 仓位都应该选择靠近基准的那一端
                    if curPrice / self.benchmark_price < 1:
                        target_pos = position_list[i+1]
                    else:
                        target_pos = position_list[i]
            # 计算当价格超出网格上边界或低于网格下边界时的目标仓位
            if curPrice / self.benchmark_price < price_list[0]:
                target_pos = 1
            if curPrice / self.benchmark_price > price_list[-1]:
                target_pos = 0
                context.stra_exit_long(code, context.stra_get_position(code), 'exitlong')
                context.stra_log_text("价格超出最大上边界，全部平多")
                self.grid_pos = target_pos
            # 止损逻辑 当价格低于下边界乘以设定的stoploss比例时止损
            if curPrice / self.benchmark_price < price_list[0] * stop_loss:
                target_pos = 0
                context.stra_exit_long(code, context.stra_get_position(code), 'exitlong')
                context.stra_log_text("价格低于最大下边界*%s，止损，全部平多" % (stop_loss))
                self.grid_pos = target_pos
            # 当目标仓位大于当前实际仓位时做多
            if target_pos > self.grid_pos:
                context.stra_enter_long(code, math.floor((target_pos-self.grid_pos)*self.cur_money/trdUnit_price), 'enterlong')
                context.stra_log_text("做多%d手,目标仓位%.2f,当前仓位%.2f,当前手数%d" % (math.floor((target_pos-self.grid_pos)*self.cur_money/trdUnit_price), target_pos, self.grid_pos,\
                                                                      context.stra_get_position(code)))
                self.grid_pos = target_pos
                return
            # 当目标仓位小于当前实际仓位时平多
            elif target_pos < self.grid_pos:
                context.stra_exit_long(code, math.floor((self.grid_pos-target_pos)*self.cur_money/trdUnit_price), 'exitlong')
                context.stra_log_text("平多%d手,目标仓位%.2f,当前仓位%.2f,当前手数%d" % (math.floor((self.grid_pos-target_pos)*self.cur_money/trdUnit_price), target_pos, self.grid_pos,\
                                                                      context.stra_get_position(code)))
                self.grid_pos = target_pos
                return
