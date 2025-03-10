import array
import math
class candlestrength:
    """
    Analyze candlestick strength based on body position within true range.
    
    Strength patterns explained (│ = wick, █ = body):
    
    Bullish Patterns (close > open):
    Strength 0 (Very Bullish):     Strength 1:           Strength 2:           Strength 3:           Strength 4:
        
        █                             │                      │                      │                      │   
        █       Bottom third          █     Bottom third     │     Middle third     █    Middle third      -    Top third
        █                             █                      █                      │                      │   
    
    Bearish Patterns (close < open):
    Strength 8 (Very Bearish):     Strength 7:           Strength 6:           Strength 5:           Strength 4:
        █                             █                      █                      │                      │   
        █         Top third           █      Top third       │    Middle third      █    Middle third      -    Bottom third
        █                             │                      │                      │                      │   
    """
    
    def __init__(self,
                 opens: array.array,
                 highs: array.array,
                 lows: array.array,
                 closes: array.array,
                 volumes: array.array,
                 avs: array.array,
                 atrs: array.array,
                 ):
        """Initialize the CandleStrength analyzer."""
        self.opens      = opens 
        self.highs      = highs 
        self.lows       = lows  
        self.closes     = closes
        self.volumes    = volumes
        self.av         = avs
        self.atr        = atrs
        
        self.strength = array.array('b', [])
        self.tr_mult = array.array('d', [])
        self.v_mult = array.array('d', [])
        self.is_bullish = None
    
    def update(self):
        """
        Update and calculate the candlestick strength.
        
        Args:
            open_price (float): Opening price
            high (float): High price
            low (float): Low price
            close (float): Closing price
            
        Returns:
            int: Strength rating from 0 (most bullish) to 8 (most bearish)
        """
        open  = self.opens[-1]
        high  = self.highs[-1]
        low   = self.lows[-1]
        close = self.closes[-1]
        # Calculate true range and section sizes
        true_range = high - low
        section_size = true_range / 3
        
        # Calculate section boundaries
        lower_third = low + section_size
        upper_third = high - section_size
        
        # Determine body position
        body_high = max(open, close)
        body_low = min(open, close)
        
        # Determine if candle is bullish or bearish
        self.is_bullish = close > open
        
        # Calculate strength based on body position
        if self.is_bullish:
            if body_high <= lower_third:
                strength = 0  # Very bullish - full body in bottom third
            elif body_low <= lower_third:
                strength = 1  # Body extends into bottom third
            elif body_high <= upper_third:
                strength = 2  # Full body in middle third
            elif body_low <= upper_third:
                strength = 3  # Body extends into middle third
            else:
                strength = 4  # Body in top third
        else:  # bearish
            if body_low >= upper_third:
                strength = 8  # Very bearish - full body in top third
            elif body_high >= upper_third:
                strength = 7  # Body extends into top third
            elif body_low >= lower_third:
                strength = 6  # Full body in middle third
            elif body_high >= lower_third:
                strength = 5  # Body extends into middle third
            else:
                strength = 4  # Body in bottom third
        
        tr_mult = 1
        v_mult = 1
        if len(self.atr) > 1:
            atr = self.atr[-2] + 1e-8
            av = self.av[-2] + 1
            tr_mult = math.log1p(true_range / atr)
            v_mult = math.log1p(self.volumes[-1] / av)
        self.strength.append(strength)
        self.tr_mult.append(tr_mult)
        self.v_mult.append(v_mult)
        LEN = 100
        if len(self.strength) > 2*LEN:
            del self.strength[:-LEN]
            del self.tr_mult[:-LEN]
            del self.v_mult[:-LEN]
        return