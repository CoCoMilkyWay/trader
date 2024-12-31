import array
import math
from collections import deque

class aobv:
    def __init__(self,
                 opens: array.array,
                 highs: array.array,
                 lows: array.array,
                 closes: array.array,
                 volumes: array.array,
                 smooth_period: int = 13  # EMA smoothing period
                 ):
        # Initialize histogram array
        self.histogram = array.array('d', [])
        
        # Initialize rolling window for money flow volumes
        self.mf_volumes = deque(maxlen=smooth_period)
        
        # Store inputs
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.volumes = volumes
        self.smooth_period = int(smooth_period)

        # Initialize previous values
        self.previous_aobv = 0.0
        self.previous_smooth = 0.0

    def update(self):
        """
        Updates AOBV histogram using rolling window:
        Money Flow Volume = Volume * ((Close-Low) - (High-Close))/(High-Low)
        AOBV = Sum of last smooth_period Money Flow Volumes
        Smooth = EMA(AOBV)
        Histogram = AOBV - Smooth
        """
        # Get current OHLCV values
        high = self.highs[-1]
        low = self.lows[-1]
        close = self.closes[-1]
        volume = self.volumes[-1]

        # Calculate money flow multiplier
        if high != low:
            mf_multiplier = ((close - low) - (high - close)) / (high - low)
            mf_volume = mf_multiplier * volume
        else:
            mf_volume = 0.0

        # Add to rolling window
        self.mf_volumes.append(mf_volume)

        # Calculate rolling AOBV
        aobv_value = sum(self.mf_volumes)
        self.previous_aobv = aobv_value

        # Calculate smoothed AOBV using EMA
        alpha = 2.0 / (self.smooth_period + 1)
        if not self.histogram:  # First value
            smooth_value = aobv_value
        else:
            smooth_value = aobv_value * alpha + self.previous_smooth * (1 - alpha)
        self.previous_smooth = smooth_value

        # Calculate and store histogram
        sign = 1 if aobv_value > smooth_value else -1
        diff = math.log1p(abs(aobv_value - smooth_value))
        histogram = sign * diff
        self.histogram.append(histogram)

        # Maintain fixed length
        LEN = 100
        if len(self.histogram) > 2*LEN:
            del self.histogram[:-LEN]