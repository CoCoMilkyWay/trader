from typing import List, Dict, Callable

from wtpy import WtBtEngine, EngineType
from wtpy import SelContext
from wtpy import BaseSelStrategy

from config.cfg_stk import cfg_stk


class Wondertrader_Thread:
    """
    Because WTCPP/WTPY is mainly a single-asset CTA-style trading engine
    For efficient market-wide back-testing with server CPUs,
    it is easiest to instance N WTCPP worker threads then manage data-coherence manually

    disable all other CTA-style features offered by WTCPP
    """

    def __init__(
        self,
        wt_assets: List[str],
        callback_functions: List[Callable],
    ):

        engine = WtBtEngine(EngineType.ET_SEL, logCfg='./config/logcfg.yaml')
        engine.init(folder='.', cfgfile='./config/configbt.yaml')
        engine.configBacktest(cfg_stk.start, cfg_stk.end)
        engine.commitBTConfig()

        str_name = f'bt_stock'

        straInfo = Strategy_Interface(
            name=str_name,
            codes=wt_assets,
            period=cfg_stk.wt_period_l,
            callback_functions=callback_functions,
        )
        engine.set_sel_strategy(
            straInfo,
            date=0, time=cfg_stk.n, period=cfg_stk.period_u,
            isRatioSlp=False)
        engine.run_backtest()


class Strategy_Interface(BaseSelStrategy):
    def __init__(self, name: str, codes: List[str], period: str, callback_functions: List[Callable]):
        BaseSelStrategy.__init__(self, name)
        self.__period__ = period
        self.__codes__ = codes
        [self.cb_on_init,
         self.cb_on_tick,
         self.cb_on_bar,
         self.cb_on_calculate,
         self.cb_on_backtest_end,
         ] = callback_functions

    def on_init(self, context: SelContext):
        for idx, code in enumerate(self.__codes__):
            r = context.stra_prepare_bars(code, self.__period__, 1)
        self.cb_on_init()

    def on_tick(self, context: SelContext, code: str, newTick: dict):
        self.cb_on_tick()
        return

    def on_bar(self, context: SelContext, code: str, period: str, newBar: dict):
        self.cb_on_bar()
        return

    def on_calculate(self, context: SelContext):
        self.cb_on_calculate()
        return

    def on_backtest_end(self, context: SelContext):
        self.cb_on_backtest_end()
        return
