# PA: Price Action
# bar的力度形态
# ref: https://www.youtube.com/watch?v=pK6S7CEqMAk

# this is mainly for Daily Bias
#   1.  Directional Move: How the candle moves (bullish or bearish).
#   2.  Size: The range between the open and close.
#   3.  Relative Position: How the candle appears on the chart in relation to adjacent candles.

# 3 critical higher time-frame candle levels:
#   1.  High or Low
#       If the price moved higher then retraced into a bearish (down) candle, its high serves as the first support level.
#       If the price moved lower then retraced into a bullish (up) candle, its low acts as the first resistance level.
#   2.  Open Price:
#       The open price of the bullish or bearish candle acts as the second support and resistance level respectively.
#   3.  Median Threshold (M.T.):
#   T   he M.T. represents the halfway point between the open and close of the candle, not the high and low, it acts as the third and final support or resistance level respectively.

# 3 types of candle zones:
#   1.  FVG(fair value gap): body of a candle not overlapped with previous/next candles' high/low
#   2.  VI(Volume Imbalance): gap between previous/next candles' body(open/close), but with wicks in gap
#   3.  Gap(Price & Volume Imbalance): gap between previous/next candles' high/low (wicks do not close)

# NOTE:
#   price has a tendency to revert back to central location
# > each bullish bar serve as resistance(at 3 critical candle levels)
#   each bearish bar serve as support(at 3 critical candle levels)
# > however there are a lot of false signals(break some level but trend continues)
# > use way to identify key levels with more strength:
#       1. a clear move away from the support/resistance candle
#           a. Price must move away twice the distance of the body of the support or resistance candle. ICT uses this approach.
#           b. A subsequent one or more candles must both open and close above the high or low of the support and resistance level respectively
#       2. creates a pivot point
#           a. by support/resist-ing the body or wick of the candle after it, effectively creates a short term low/high pivot point (mini nexus/zs)


























'''
candle_rankings = {
    "CDL3LINESTRIKE_Bull": 1,
    "CDL3LINESTRIKE_Bear": 2,
    "CDL3BLACKCROWS_Bull": 3,
    "CDL3BLACKCROWS_Bear": 3,
    "CDLEVENINGSTAR_Bull": 4,
    "CDLEVENINGSTAR_Bear": 4,
    "CDLTASUKIGAP_Bull": 5,
    "CDLTASUKIGAP_Bear": 5,
    "CDLINVERTEDHAMMER_Bull": 6,
    "CDLINVERTEDHAMMER_Bear": 6,
    "CDLMATCHINGLOW_Bull": 7,
    "CDLMATCHINGLOW_Bear": 7,
    "CDLABANDONEDBABY_Bull": 8,
    "CDLABANDONEDBABY_Bear": 8,
    "CDLBREAKAWAY_Bull": 10,
    "CDLBREAKAWAY_Bear": 10,
    "CDLMORNINGSTAR_Bull": 12,
    "CDLMORNINGSTAR_Bear": 12,
    "CDLPIERCING_Bull": 13,
    "CDLPIERCING_Bear": 13,
    "CDLSTICKSANDWICH_Bull": 14,
    "CDLSTICKSANDWICH_Bear": 14,
    "CDLTHRUSTING_Bull": 15,
    "CDLTHRUSTING_Bear": 15,
    "CDLINNECK_Bull": 17,
    "CDLINNECK_Bear": 17,
    "CDL3INSIDE_Bull": 20,
    "CDL3INSIDE_Bear": 56,
    "CDLHOMINGPIGEON_Bull": 21,
    "CDLHOMINGPIGEON_Bear": 21,
    "CDLDARKCLOUDCOVER_Bull": 22,
    "CDLDARKCLOUDCOVER_Bear": 22,
    "CDLIDENTICAL3CROWS_Bull": 24,
    "CDLIDENTICAL3CROWS_Bear": 24,
    "CDLMORNINGDOJISTAR_Bull": 25,
    "CDLMORNINGDOJISTAR_Bear": 25,
    "CDLXSIDEGAP3METHODS_Bull": 27,
    "CDLXSIDEGAP3METHODS_Bear": 26,
    "CDLTRISTAR_Bull": 28,
    "CDLTRISTAR_Bear": 76,
    "CDLGAPSIDESIDEWHITE_Bull": 46,
    "CDLGAPSIDESIDEWHITE_Bear": 29,
    "CDLEVENINGDOJISTAR_Bull": 30,
    "CDLEVENINGDOJISTAR_Bear": 30,
    "CDL3WHITESOLDIERS_Bull": 32,
    "CDL3WHITESOLDIERS_Bear": 32,
    "CDLONNECK_Bull": 33,
    "CDLONNECK_Bear": 33,
    "CDL3OUTSIDE_Bull": 34,
    "CDL3OUTSIDE_Bear": 39,
    "CDLRICKSHAWMAN_Bull": 35,
    "CDLRICKSHAWMAN_Bear": 35,
    "CDLSEPARATINGLINES_Bull": 36,
    "CDLSEPARATINGLINES_Bear": 40,
    "CDLLONGLEGGEDDOJI_Bull": 37,
    "CDLLONGLEGGEDDOJI_Bear": 37,
    "CDLHARAMI_Bull": 38,
    "CDLHARAMI_Bear": 72,
    "CDLLADDERBOTTOM_Bull": 41,
    "CDLLADDERBOTTOM_Bear": 41,
    "CDLCLOSINGMARUBOZU_Bull": 70,
    "CDLCLOSINGMARUBOZU_Bear": 43,
    "CDLTAKURI_Bull": 47,
    "CDLTAKURI_Bear": 47,
    "CDLDOJISTAR_Bull": 49,
    "CDLDOJISTAR_Bear": 51,
    "CDLHARAMICROSS_Bull": 50,
    "CDLHARAMICROSS_Bear": 80,
    "CDLADVANCEBLOCK_Bull": 54,
    "CDLADVANCEBLOCK_Bear": 54,
    "CDLSHOOTINGSTAR_Bull": 55,
    "CDLSHOOTINGSTAR_Bear": 55,
    "CDLMARUBOZU_Bull": 71,
    "CDLMARUBOZU_Bear": 57,
    "CDLUNIQUE3RIVER_Bull": 60,
    "CDLUNIQUE3RIVER_Bear": 60,
    "CDL2CROWS_Bull": 61,
    "CDL2CROWS_Bear": 61,
    "CDLBELTHOLD_Bull": 62,
    "CDLBELTHOLD_Bear": 63,
    "CDLHAMMER_Bull": 65,
    "CDLHAMMER_Bear": 65,
    "CDLHIGHWAVE_Bull": 67,
    "CDLHIGHWAVE_Bear": 67,
    "CDLSPINNINGTOP_Bull": 69,
    "CDLSPINNINGTOP_Bear": 73,
    "CDLUPSIDEGAP2CROWS_Bull": 74,
    "CDLUPSIDEGAP2CROWS_Bear": 74,
    "CDLGRAVESTONEDOJI_Bull": 77,
    "CDLGRAVESTONEDOJI_Bear": 77,
    "CDLHIKKAKEMOD_Bull": 82,
    "CDLHIKKAKEMOD_Bear": 81,
    "CDLHIKKAKE_Bull": 85,
    "CDLHIKKAKE_Bear": 83,
    "CDLENGULFING_Bull": 84,
    "CDLENGULFING_Bear": 91,
    "CDLMATHOLD_Bull": 86,
    "CDLMATHOLD_Bear": 86,
    "CDLHANGINGMAN_Bull": 87,
    "CDLHANGINGMAN_Bear": 87,
    "CDLRISEFALL3METHODS_Bull": 94,
    "CDLRISEFALL3METHODS_Bear": 89,
    "CDLKICKING_Bull": 96,
    "CDLKICKING_Bear": 102,
    "CDLDRAGONFLYDOJI_Bull": 98,
    "CDLDRAGONFLYDOJI_Bear": 98,
    "CDLCONCEALBABYSWALL_Bull": 101,
    "CDLCONCEALBABYSWALL_Bear": 101,
    "CDL3STARSINSOUTH_Bull": 103,
    "CDL3STARSINSOUTH_Bear": 103,
    "CDLDOJI_Bull": 104,
    "CDLDOJI_Bear": 104,
    'CDLCOUNTERATTACK': -1,
    'CDLLONGLINE': -1,
    'CDLSHORTLINE': -1,
    'CDLSTALLEDPATTERN': -1,
    'CDLKICKINGBYLENGTH': -1,
}
'''