from typing import Tuple, List
from config.cfg_cpt import cfg_cpt

class ChandelierIndicator:
    def __init__(self, length:int=22, atr_period:int=22, mult:float=3):
        self.length = length
        self.atr_period = atr_period
        self.mult = mult

        # Price histories
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.closes: List[float] = []
        self.tr: List[float] = []

        # State variables
        self.prev_shortcs = float('inf')  # Initial shortvs[1]
        self.prev_longcs = 0.0            # Initial longvs[1]
        self.prev_direction = 0           # Initial direction[1]
        self.prev_close = 0.0

        # History for plotting/analysis
        if cfg_cpt.dump_ind:
            self.his_ts: List[float] = []
            self.his_longcs: List[float] = []
            self.his_shortcs: List[float] = []
            self.switch_idx: int = 0
            self.his_switch_ts: List[float] = []
            self.his_switch_vs: List[float] = []

    def _calculate_atr(self) -> float:
        if len(self.tr) < self.atr_period:
            return 0.0
        return sum(self.tr[-self.atr_period:]) / self.atr_period

    def update(self, high: float, low: float, close: float, ts: float) -> Tuple[bool, bool]:
        # Update price histories
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)

        # Calculate True Range
        tr = max(
            high - low,
            abs(high - self.prev_close),
            abs(low - self.prev_close)
        )
        self.tr.append(tr)

        # Need enough data
        if len(self.highs) < max(self.length, self.atr_period):
            self.prev_close = close
            return False, False

        # Trim histories to required length
        while len(self.highs) > max(self.length, self.atr_period):
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)
            self.tr.pop(0)

        # Calculate ATR
        atr = self._calculate_atr()

        # Calculate base stops
        lowest = min(self.lows[-self.length:])
        highest = max(self.highs[-self.length:])
        
        short_stop = lowest + self.mult * atr
        long_stop = highest - self.mult * atr

        # Calculate final stops with memory (trailing stops)
        shortcs = short_stop if close > self.prev_shortcs else min(short_stop, self.prev_shortcs)
        longcs = long_stop if close < self.prev_longcs else max(long_stop, self.prev_longcs)

        # Calculate switches
        long_switch = close >= self.prev_shortcs and self.prev_close < self.prev_shortcs
        short_switch = close <= self.prev_longcs and self.prev_close > self.prev_longcs

        # Calculate direction
        direction = (1 if self.prev_direction <= 0 and long_switch else
                    -1 if self.prev_direction >= 0 and short_switch else
                    self.prev_direction)

        switch = direction != self.prev_direction

        # Update state
        self.prev_shortcs = shortcs
        self.prev_longcs = longcs
        self.prev_direction = direction
        self.prev_close = close

        # Store history if enabled
        if cfg_cpt.dump_ind:
            self.his_ts.append(ts)
            self.his_longcs.append(longcs)
            self.his_shortcs.append(shortcs)
            
            if switch:
                self.switch_idx += 1
                self.his_switch_ts.append(ts)
                self.his_switch_vs.append(longcs if long_switch else shortcs)

        return long_switch, short_switch