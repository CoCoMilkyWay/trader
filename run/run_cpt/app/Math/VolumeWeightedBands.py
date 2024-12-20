from typing import Tuple, List
from config.cfg_cpt import cfg_cpt

class VolumeWeightedBands:
    def __init__(self, 
                 window_size: int = 800,
                 window_size_atr: int = 240,  # 4hrs * 60min default
                 band1_upper: float = 1.28, band1_lower: float = 1.28,
                 band2_upper: float = 2.01, band2_lower: float = 2.01,
                 band3_upper: float = 2.51, band3_lower: float = 2.51,
                 band4_upper: float = 3.09, band4_lower: float = 3.09,
                 band5_upper: float = 4.01, band5_lower: float = 4.01,
                 enable_band2: bool = False,
                 enable_band3: bool = False,
                 enable_band4: bool = False,
                 enable_band5: bool = False):
        
        # Rolling window configuration
        # assert(window_size >= 10)
        # assert(atr_window <= window_size), "ATR window must be smaller than or equal to main window size"
        if window_size_atr > window_size:
            window_size_atr = window_size
        self.window_size = window_size
        self.window_size_atr = window_size_atr
        self.min_rate = cfg_cpt.FEE 
        
        # Price and volume history for rolling calculation
        self.price_history = []
        self.volume_history = []
        
        # ATR calculation history
        self.high_history = []
        self.low_history = []
        self.close_history = []
        self.atr_history = []
        
        # Band multipliers
        self.band_params = [
            (band1_upper, band1_lower, True),
            (band2_upper, band2_lower, enable_band2),
            (band3_upper, band3_lower, enable_band3),
            (band4_upper, band4_lower, enable_band4),
            (band5_upper, band5_lower, enable_band5)
        ]
        
        if cfg_cpt.dump_ind:
            self.his_ts: List[float] = []
            self.his_vavg: List[float] = []
            self.his_b1up: List[float] = []
            self.his_b1lo: List[float] = []
            
    def _calculate_rolling_vwap(self) -> float:
        """Calculate volume-weighted average price"""
        if not self.price_history:
            return 0.0
        
        weighted_sum = sum(p * v for p, v in zip(self.price_history, self.volume_history))
        total_volume = sum(self.volume_history)
        
        return weighted_sum / total_volume if total_volume > 0 else 0.0
    
    def _calculate_rolling_dispersion(self, average: float) -> float:
        """Calculate volume-weighted price dispersion from the average (original method)"""
        if not self.price_history:
            return 0.0
        
        weighted_squared_deviations = [
            (price - average) ** 2 * volume 
            for price, volume in zip(self.price_history, self.volume_history)
        ]
        
        total_volume = sum(self.volume_history)
        if total_volume == 0:
            return 0.0
            
        weighted_variance = sum(weighted_squared_deviations) / total_volume
        return (weighted_variance ** 0.5) if weighted_variance > 0 else 0
    
    def _calculate_atr(self) -> float:
        """Calculate Average True Range"""
        if len(self.close_history) < 2:
            return 0.0
        
        tr_values = []
        for i in range(1, len(self.close_history)):
            high = self.high_history[i]
            low = self.low_history[i]
            prev_close = self.close_history[i-1]
            
            tr = max(
                high - low,  # Current high-low range
                abs(high - prev_close),  # Current high - previous close
                abs(low - prev_close)    # Current low - previous close
            )
            tr_values.append(tr)
            
        # Calculate simple moving average of TR values
        if not tr_values:
            return 0.0
            
        atr = sum(tr_values[-self.window_size_atr:]) / min(len(tr_values), self.window_size_atr)
        return atr
    
    def update(self, high: float, low: float, close: float, volume: float, ts:float) -> List:
        """
        Process new price bar and calculate bands
        Returns: (average, band1_up, band1_down, band2_up, band2_down, ...)
        """
        # Calculate typical price
        typical_price = (high + low) * 0.5
        
        # Maintain rolling windows
        self.price_history.append(typical_price)
        self.volume_history.append(volume)
        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)
        
        # Trim to window size
        if len(self.price_history) > self.window_size:
            self.price_history = self.price_history[-self.window_size:]
            self.volume_history = self.volume_history[-self.window_size:]
            self.high_history = self.high_history[-self.window_size:]
            self.low_history = self.low_history[-self.window_size:]
            self.close_history = self.close_history[-self.window_size:]
        
        # Calculate volume-weighted average
        average = self._calculate_rolling_vwap()
        
        # Calculate ATR-based dispersion
        dispersion = self._calculate_atr()
        min_dispersion = self.min_rate * average
        if abs(dispersion)< min_dispersion: # avoid bad values
            dispersion = min_dispersion
            disp_mult = 1
        else:
            disp_mult = abs(close-average)/dispersion
        
        # Generate band results
        result = [average, disp_mult]
        for band_config in self.band_params:
            upper_mult, lower_mult, is_enabled = band_config
            if is_enabled:
                result.extend([
                    average + upper_mult * dispersion, 
                    average - lower_mult * dispersion
                ])
        
        if cfg_cpt.dump_ind:
            self.his_ts.append(ts)
            self.his_vavg.append(result[0])
            self.his_b1up.append(result[2])
            self.his_b1lo.append(result[3])
        
        return result