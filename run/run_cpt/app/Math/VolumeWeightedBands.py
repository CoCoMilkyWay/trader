from typing import Tuple, List
from config.cfg_cpt import cfg_cpt

class VolumeWeightedBands:
    def __init__(self, 
                 window_size: int = 50,
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
        assert(window_size>=10)
        self.window_size = window_size
        
        # Price and volume history for rolling calculation
        self.price_history = []
        self.volume_history = []
        
        # Band multipliers
        self.band_params = [
            (band1_upper, band1_lower),
            (band2_upper, band2_lower, enable_band2),
            (band3_upper, band3_lower, enable_band3),
            (band4_upper, band4_lower, enable_band4),
            (band5_upper, band5_lower, enable_band5)
        ]
        
        if cfg_cpt.dump_ind:
            self.his_ts: List[float] = []
            self.his_vavg: List[float] = []
            self.his_tavg: List[float] = []
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
        """Calculate volume-weighted price dispersion from the average"""
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
    
    def update(self, high: float, low: float, close: float, volume: float, ts:float) -> Tuple[float,float]:
        """
        Process new price bar and calculate bands
        Returns: (average, band1_up, band1_down, band2_up, band2_down, ...)
        """
        # Calculate typical price
        typical_price = (high + low) * 0.5
        
        # Maintain rolling window
        self.price_history.append(typical_price)
        self.volume_history.append(volume)
        
        # Trim to window size
        if len(self.price_history) > self.window_size:
            self.price_history = self.price_history[-self.window_size:]
            self.volume_history = self.volume_history[-self.window_size:]
        
        # Calculate volume-weighted average
        average = self._calculate_rolling_vwap()
        
        # Calculate price dispersion
        dispersion = self._calculate_rolling_dispersion(average)
        disp_mult = 0.0
        if dispersion and average:
            disp_mult = abs(close-average)/dispersion
            if disp_mult > 100:
                print('wtf', average, disp_mult, dispersion)
                disp_mult = 100
        
        # print(average, disp_mult, dispersion)
        # Generate band results
        result = (average, disp_mult)
        # sum(self.price_history)/len(self.price_history)
        # for band_config in self.band_params:
        #     if len(band_config) == 3:  # Check if band is enabled
        #         upper_mult, lower_mult, is_enabled = band_config
        #         if is_enabled:
        #             result.extend([
        #                 average + upper_mult * dispersion, 
        #                 average - lower_mult * dispersion
        #             ])
        #         else:
        #             result.extend([0.0, 0.0])
        #     else:
        #         # Always include bands 1 and 2
        #         upper_mult, lower_mult = band_config
        #         result.extend([
        #             average + upper_mult * dispersion, 
        #             average - lower_mult * dispersion
        #         ])
        
        
        if cfg_cpt.dump_ind:
            self.his_ts.append(ts)
            self.his_vavg.append(result[0])
            # self.his_b1up.append(result[2])
            # self.his_b1lo.append(result[3])
        return result