from wtpy import BaseCtaStrategy
from wtpy import CtaContext
import numpy as np
import pandas as pd
def load_pred():
    import pickle
    with open('pred.pkl', 'rb') as f:
        data = pickle.load(f)
    datetime = data.index.get_level_values(0).tolist()
    instrument = data.index.get_level_values(1).tolist()
    pred = data.xs('SH000300', level='instrument')
    return datetime, instrument, pred

score_high = 0.02
score_low = 0.02
class ML_pred(BaseCtaStrategy):
    
    def __init__(self, name:str, code:str, barCnt:int, period:str, isForStk:bool = False):
        BaseCtaStrategy.__init__(self, name)
        self.__period__ = period
        self.__bar_cnt__ = barCnt
        self.__code__ = code

        self.__is_stk__ = isForStk
        self.barnum = 0
        self.datetime, self.instrument, self.pred = load_pred()
        self.pred.index = self.pred.index.strftime('%Y%m%d').astype(int)

    def on_init(self, context:CtaContext):
        code = self.__code__    #品种代码
        if self.__is_stk__:
            code = code + "-"   # 如果是股票代码，后面加上一个+/-，+表示后复权，-表示前复权

        #这里演示了品种信息获取的接口
        #　pInfo = context.stra_get_comminfo(code)
        #　print(pInfo)

        context.stra_prepare_bars(code, self.__period__, self.__bar_cnt__, isMain = True)
        context.stra_sub_ticks(code)
        context.stra_log_text("ML_pred inited")

        #读取存储的数据
        self.xxx = context.user_load_data('xxx',1)

    def on_tick(self, context: CtaContext, stdCode: str, newTick: dict):
        # print(newTick)
        pass
    
    def on_bar(self,  context: CtaContext, stdCode: str, newTick: dict):
        self.barnum += 1
    
    def on_calculate(self, context:CtaContext):
        long = False
        exit_long = False
        
        code = self.__code__    #品种代码

        trdUnit = 1
        if self.__is_stk__:
            trdUnit = 100

        #读取最近n条线(dataframe对象)
        theCode = code
        if self.__is_stk__:
            theCode = theCode + "-" # 如果是股票代码，后面加上一个+/-，+表示后复权，-表示前复权
        np_bars = context.stra_get_bars(theCode, self.__period__, self.__bar_cnt__, isMain = True)

        days = 1
        #平仓价序列、最高价序列、最低价序列
        closes = np_bars.closes
        highs = np_bars.highs
        lows = np_bars.lows
        #读取days天之前到上一个交易日位置的数据
        hh = np.amax(highs[-1])
        hc = np.amax(closes[-1])
        ll = np.amin(lows[-1])
        lc = np.amin(closes[-1])
        #读取今天的开盘价、最高价和最低价
        # lastBar = df_bars.get_last_bar()
        openpx = np_bars.opens[-1]
        highpx = np_bars.highs[-1]
        lowpx = np_bars.lows[-1]
        
        #把策略参数读进来，作为临时变量，方便引用
        barnum = self.barnum
        
        # date = pd.to_datetime(str(context.get_date()), format="%Y%m%d").date
        date = context.get_date() # int
        
        try:
            score = self.pred.loc[date, 'score']
            if score > score_high:
                long = True
            elif score < score_low:
                exit_long = True
        except:
            return
        
        #读取当前仓位
        curPos = context.stra_get_position(code)/trdUnit
        if curPos == 0:
            if long:
                context.stra_enter_long(code, 1, 'enterlong')
                context.stra_log_text(stdio(f"{barnum}: 模型分数{score:.2f}>={score_high:.2f}，多仓进场"))
                #修改并保存
                self.xxx = 1
                context.user_save_data('xxx', self.xxx)
            return
        elif curPos > 0:
            if exit_long:
                context.stra_exit_long(code, 1, 'exitlong')
                context.stra_log_text(stdio(f"{barnum}: 模型分数{score:.2f}<={score_low:.2f}，多仓出场"))
                #raise Exception("except on purpose")
            return

def stdio(str):
    print(str)
    return str