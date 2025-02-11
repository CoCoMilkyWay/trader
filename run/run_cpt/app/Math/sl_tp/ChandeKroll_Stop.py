from typing import Tuple, List
from config.cfg_cpt import cfg_cpt

class ChandeKrollStop:
    def __init__(self, atr_length: int = 10, atr_coef: float = 1, stop_len: int = 9):
        """
        Initialize Chande Kroll Stop indicator
        
        Args:
            p (int): ATR Length (default: 10)
            x (int): ATR Coefficient (default: 1)
            q (int): Stop Length (default: 9)
        """
        self.p = atr_length  # ATR Length
        self.x = atr_coef  # ATR Coefficient
        self.q = stop_len  # Stop Length
        
        # Price histories
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.tr: List[float] = []
        
        # Intermediate calculations
        self.first_high_stops: List[float] = []
        self.first_low_stops: List[float] = []
        
        # Previous values for switch detection
        self.prev_close = 0.0
        self.prev_stop_long = 0.0
        self.prev_stop_short = float('inf')
        
        if cfg_cpt.dump_ind:
            self.his_ts = []
            self.his_upper = []
            self.his_lower = []

    def _calculate_tr(self, high: float, low: float, close: float) -> float:
        """Calculate True Range"""
        return max(
            high - low,
            abs(high - self.prev_close),
            abs(low - self.prev_close)
        )

    def _calculate_atr(self) -> float:
        """Calculate Average True Range"""
        if len(self.tr) < self.p:
            return 0.0
        return sum(self.tr[-self.p:]) / self.p

    def update(self, high: float, low: float, close: float, ts: float) -> Tuple[bool, bool]:
        """
        Update the Chande Kroll Stop indicator with new price data
        
        Args:
            high (float): Current period's high price
            low (float): Current period's low price
            close (float): Current period's close price
            
        Returns:
            Tuple[bool, bool]: (long_switch, short_switch) signals
        """
        # Update price histories
        self.highs.append(high)
        self.lows.append(low)
        
        # Calculate and store True Range
        tr = self._calculate_tr(high, low, close)
        self.tr.append(tr)
        
        # Keep histories memory-bound to required length
        required_length = max(self.p + self.q - 1, self.p)
        if len(self.highs) > required_length:
            self.highs.pop(0)
            self.lows.pop(0)
            self.tr.pop(0)
            
        # Need enough data
        if len(self.highs) < required_length:
            self.prev_close = close
            return False, False
            
        # Calculate ATR
        atr = self._calculate_atr()
        
        # Calculate first level stops
        highest_p = max(self.highs[-self.p:])
        lowest_p = min(self.lows[-self.p:])
        
        first_high_stop = highest_p - self.x * atr
        first_low_stop = lowest_p + self.x * atr
        
        # Store and bound intermediate calculations
        self.first_high_stops.append(first_high_stop)
        self.first_low_stops.append(first_low_stop)
        
        if len(self.first_high_stops) > self.q:
            self.first_high_stops.pop(0)
            self.first_low_stops.pop(0)
            
        # Calculate final stops
        stop_short = max(self.first_high_stops)
        stop_long = min(self.first_low_stops)
        
        # Detect switches
        long_switch = close >= self.prev_stop_short and self.prev_close < self.prev_stop_short
        short_switch = close <= self.prev_stop_long and self.prev_close > self.prev_stop_long
        
        # Update state
        self.prev_close = close
        self.prev_stop_long = stop_long
        self.prev_stop_short = stop_short
        
        if cfg_cpt.dump_ind:
            self.his_ts.append(ts)
            self.his_upper.append(stop_short)
            self.his_lower.append(stop_long)

        return long_switch, short_switch