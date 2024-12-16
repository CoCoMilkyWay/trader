from typing import Tuple, List
from config.cfg_cpt import cfg_cpt

class ChandelierIndicator:
    def __init__(self, length=22, atr_period=14, mult=2.0, use_close=False):
        self.high_length = length
        self.low_length = length
        self.atr_length = atr_period
        self.multiplier = mult
        self.use_close = use_close

        # Price histories
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.closes: List[float] = []
        self.tr: List[float] = []

        # State variables
        self.direction = 1  # Initialize direction as up
        self.prev_close = None
        
        # History for plotting/analysis
        if cfg_cpt.dump_ind:
            self.his_ts: List[float] = []
            self.his_longcs: List[float] = []
            self.his_shortcs: List[float] = []
            self.switch_idx: int = 0
            self.his_switch_ts: List[float] = []
            self.his_switch_vs: List[float] = []

    def _calculate_atr(self) -> float:
        """Calculate RMA-based ATR"""
        if len(self.tr) < self.atr_length:
            return 0.0
        
        # Use RMA (Rolling Moving Average) for ATR calculation
        alpha = 1.0 / self.atr_length
        atr = self.tr[0]
        for i in range(1, self.atr_length):
            atr = (alpha * self.tr[i]) + ((1 - alpha) * atr)
        return atr

    def update(self, high: float, low: float, close: float, ts: float) -> Tuple[bool, bool]:
        # Update price histories
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)

        # Calculate True Range
        if self.prev_close is not None:
            tr = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close)
            )
        else:
            tr = high - low
        self.tr.append(tr)

        # Keep required history length
        max_period = max(max(self.high_length, self.low_length), self.atr_length)
        while len(self.highs) > max_period:
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)
            self.tr.pop(0)

        if len(self.highs) < max_period:
            self.prev_close = close
            return False, False

        # Calculate ATR
        atr = self._calculate_atr()
        if atr is None:
            self.prev_close = close
            return False, False

        atr_mult = atr * self.multiplier

        # Calculate stops
        if self.use_close:
            roll_length = max(self.high_length, self.low_length)
            highest = max(self.closes[-roll_length:])
            lowest = min(self.closes[-roll_length:])
        else:
            highest = max(self.highs[-self.high_length:])
            lowest = min(self.lows[-self.low_length:])

        long_stop = highest - atr_mult
        short_stop = lowest + atr_mult

        # Calculate trend changes
        prev_direction = self.direction
        if close > long_stop:
            self.direction = 1
        elif close < short_stop:
            self.direction = -1

        # Determine switches
        long_switch = self.direction == 1 and prev_direction <= 0
        short_switch = self.direction == -1 and prev_direction >= 0

        # Store history if enabled
        if cfg_cpt.dump_ind:
            self.his_ts.append(ts)
            self.his_longcs.append(long_stop)
            self.his_shortcs.append(short_stop)
            
            if long_switch or short_switch:
                self.switch_idx += 1
                self.his_switch_ts.append(ts)
                self.his_switch_vs.append(long_stop if long_switch else short_stop)

        self.prev_close = close
        return long_switch, short_switch