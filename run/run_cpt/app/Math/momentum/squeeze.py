import array
import math
from Math.util.RecursiveLinReg import RecursiveLinReg

class squeeze:
    """
    A sample squeeze indicator class that computes:
      1. A squeeze rating based on Bollinger Band and Keltner Channel widths.
      2. A momentum measure given by the slope of a linear regression of close prices.
    
    The linear regression is computed recursively for performance.
    """
    def __init__(self,
                 closes: array.array,
                 bb_upper: array.array,    # Bollinger Band upper
                 bb_lower: array.array,    # Bollinger Band lower
                 kc_upper: array.array,    # Keltner Channel upper
                 kc_lower: array.array,    # Keltner Channel lower
                 linreg_length: int = 20   # Regression window length
                 ):
        self.closes = closes
        self.bb_upper = bb_upper
        self.bb_lower = bb_lower
        self.kc_upper = kc_upper
        self.kc_lower = kc_lower
        self.linreg_length = linreg_length
        
        # Output arrays
        self.momentum = array.array('d', [])        # Linear regression slope (momentum)
        self.squeeze_rating = array.array('d', [])    # Squeeze rating (transformed for even distribution)
        self.data_points = 0
        
        # Initialize our recursive linear regression helper.
        self.linreg_reg = RecursiveLinReg(linreg_length)
    
    def calculate_squeeze_rating(self, bb_width: float, kc_width: float) -> float:
        """
        Calculate the squeeze rating based on the ratio of Bollinger Band width to
        Keltner Channel width. Here we use the logarithm to spread the distribution.
        """
        if kc_width != 0:
            bb_kc_ratio = bb_width / kc_width
        else:
            bb_kc_ratio = 10.0  # A fallback when kc_width is zero.
        return math.log1p(bb_kc_ratio)
    
    def update(self):
        """
        Update the squeeze indicator by:
          1. Computing the latest squeeze rating.
          2. Updating the regression slope (momentum) recursively.
        """
        self.data_points += 1
        
        # Use the latest Bollinger Band and Keltner Channel values.
        bb_width = self.bb_upper[-1] - self.bb_lower[-1]
        kc_width = self.kc_upper[-1] - self.kc_lower[-1]
        
        # Compute and store the squeeze rating.
        rating = self.calculate_squeeze_rating(bb_width, kc_width)
        self.squeeze_rating.append(rating)
        
        # Update the recursive linear regression with the most recent close price.
        new_close = self.closes[-1]
        slope = self.linreg_reg.update(new_close)
        if slope is None:
            # Not enough data yet to compute the regression—store 0 momentum.
            self.momentum.append(0.0)
        else:
            self.momentum.append(slope)
        
        # Optional: trim stored output arrays to limit memory usage.
        LEN = 100
        if len(self.momentum) > 2 * LEN:
            del self.momentum[:-LEN]
            del self.squeeze_rating[:-LEN]

"""
//@version=5
indicator("Recursive LinReg Squeeze", overlay=false, shorttitle="R-LinReg Squeeze")

// ─────────────────────────────
//  INPUTS
// ─────────────────────────────
window_size = input.int(20, "Regression Window", minval=2)
length      = input.int(20, "BB / KC Length", minval=1)
mult        = input.float(2.0, "BB Multiplier", step=0.1)
kc_factor   = input.float(1.5, "KC Factor", step=0.1)

// ─────────────────────────────
//  BOLLINGER BANDS CALCULATION
// ─────────────────────────────
basis    = ta.sma(close, length)
dev      = mult * ta.stdev(close, length)
bb_upper = basis + dev
bb_lower = basis - dev
bb_width = bb_upper - bb_lower

// ─────────────────────────────
//  KELTNER CHANNEL CALCULATION
// ─────────────────────────────
kc_middle = ta.sma(close, length)
atr_val   = ta.atr(length)
kc_upper  = kc_middle + kc_factor * atr_val
kc_lower  = kc_middle - kc_factor * atr_val
kc_width  = kc_upper - kc_lower

// ─────────────────────────────
//  SQUEEZE RATING CALCULATION
// ─────────────────────────────
// Use natural logarithm (math.log) to spread the distribution.
// When kc_width is zero we use a fallback ratio of 10.
squeeze_rating = (kc_width != 0) ? math.log(1 + bb_width / kc_width) : math.log(11)

// ─────────────────────────────
//  RECURSIVE LINEAR REGRESSION CALCULATION
// ─────────────────────────────
// We want to compute the slope for the last 'window_size' close values.
// The standard regression slope for points (x, y) with x = 0, 1, ..., n-1 is:
//     slope = (n * S_xy - S_x * S_y) / D,   where D = n * S_xx - S_x^2,
// with S_y = sum(y) and S_xy = sum(i * y).

// We update S_y and S_xy recursively as new bars arrive.
// When a new bar comes in and the window is full, we remove the oldest value.
// The recursive update formulas are:
//
//     S_y_new  = S_y_old - y_old + new_value
//     S_xy_new = S_xy_old - (S_y_old - y_old) + (window_size - 1)*new_value
//
// where y_old is the value that left the window.
// Since we want the window to contain the most recent 'window_size' bars
// (with the oldest assigned weight 0 and the newest weight window_size-1),
// the oldest value is the one from 'window_size' bars ago (i.e. close[window_size]).

// Declare persistent (var) variables:
var float S_y  = 0.0    // running sum of y-values
var float S_xy = 0.0    // running weighted sum: sum(i * y)
var float S_x  = 0.0    // constant: sum of indices 0 to (window_size-1)
var float S_xx = 0.0    // constant: sum of squares of indices
var float D    = 0.0    // constant: window_size * S_xx - S_x^2

// On the very first bar, precompute S_x, S_xx and D.
if barstate.isfirst
    for i = 0 to window_size - 1
        S_x  += i
        S_xx += i * i
    D := window_size * S_xx - S_x * S_x

// The regression “slope” (our momentum measure):
var float slope = na

// For the first window_size bars, accumulate the sums.
// We assign the current bar a weight equal to its (zero‑based) index.
if bar_index < window_size
    S_y  += close
    S_xy += bar_index * close
    // When the window just becomes full, compute the slope.
    if bar_index == window_size - 1
        slope := (window_size * S_xy - S_x * S_y) / D
else
    // When the window is full, update recursively.
    // y_old is the close value from window_size bars ago.
    y_old = close[window_size]
    S_y  := S_y[1]  - y_old + close
    S_xy := S_xy[1] - (S_y[1] - y_old) + (window_size - 1) * close
    slope := (window_size * S_xy - S_x * S_y) / D

// ─────────────────────────────
//  PLOTTING
// ─────────────────────────────
// plot(slope,         title="Momentum (Slope)",     color=color.blue, linewidth=2)
plot(squeeze_rating, title="Squeeze Rating",       color=color.red,  linewidth=2)

"""