# this is strat look-up-table from ML features (better understand ML decision)
# ref: https://toslc.thinkorswim.com/center/reference/Tech-Indicators/strategies

# 回测优先考虑外汇市场，流动性最高，TA属性最强

strategies_dict = {
    "Trend-Following Strategies": {
        # 趋势的建立：
        #   1. 破线(突破)
        #   2. 均线拐头（EMA特性：破线既(均线)拐头，SMA:抵扣(n天前)价算法）
        #   2. 均线金/死叉
        #   3. 均线多/空头排列(价格保持在均线上方运行(MACD DIFF/DEA 为正，绿柱保持))
        #       (均线密集区 = 市场成本统一 = 趋势开始)
        #   4. 趋势建立后，才有趋势强度(乖离率)的讨论
        #       (比如MACD DIFF,DEA反转; 如果时间级别太小，MACD反转太频繁，可以考虑MACD和价格的顶顶、底底背离)
        "Moving Averages": {
            # 2种类模式：trend-following / reversal
            # 考虑MACD(12,26,9 / 5,10,5), EMA5, EMA20, EMA50, EMA200
            # 用EMA50/EMA200判断长期趋势， 短期金叉死叉(EMA/MACD)作为交易信号
            #   1. 金叉死叉假信号多，在长期上升趋势中只做金叉，下降中只做死叉
            #   2. 交叉（开仓）前没有明显的reversal行为
            #   3. 滤掉逆周期的价格通道，避免横盘震荡
            #   4. 止损设置为前低/高，止盈为止损1.5倍, 取决于综合趋势强度，pnl可以适度增加到3倍
            "MovAvgStrat": {},
            "GoldenCrossBreakouts": {},
            "VWMABreakouts": {}
        },
        "Directional Indicators": {
            "ADXTrend": {},
            "ADXBreakoutsLE": {},
            "TAC_DMI": {}
        },
        "Long-Term Trend": {
            "EightMonthAvg": {},
            "LongHaul": {},
            "HACOLTStrat": {}
        },
        "Trend Channels and Zones": {
            "PriceZoneOscillatorLE": {},
            "PriceZoneOscillatorLX": {},
            "PriceZoneOscillatorSE": {},
            "PriceZoneOscillatorSX": {},
            "SimpleTrendChannel": {},
            "MiddleHighLowMAStrat": {}
        },
        "Volatility-Based Trend": {
            "ATRHighSMABreakoutsLE": {},
            "ATRTrailingStopLE": {},
            "ATRTrailingStopSE": {},
            "VHFTrend": {},
            "VolatilityBand": {}
        },
        "Other Trend-Following": {
            "R2Trend": {},
            "ERTrend": {},
            "OnsetTrend": {},
            "SVEHaTypCross": {},
            "UniversalOscillatorStrat": {}
        }
    },
    "Breakout Strategies": {
        "Momentum-Based Breakouts": {
            "MomentumLE": {},
            "FirstHourBreakout": {},
            "FourDayBreakoutLE": {},
            "KeyRevLE": {},
            "KeyRevLX": {}
        },
        "Bollinger Band Breakouts": {
            "BollingerBandsLE": {},
            "BollingerBandsSE": {},
            "BollingerBandsWithEngulfing": {}
        },
        "Range Breakouts": {
            "Donchian": {},
            "GoldenTriangleLE": {},
            "MeanReversionSwingLE": {}
        },
        "Gap-Based Breakouts": {
            "GapDownSE": {},
            "GapMomentumSystem": {},
            "GapReversalLE": {},
            "GapUpLE": {}
        }
    },
    "Mean Reversion Strategies": {
        "Simple Mean Reversion": {
            "SimpleMeanReversion": {}
        },
        "Regression-Based Mean Reversion": {
            "RegressionDivergenceStrat": {}
        },
        "Stochastic Oscillators": {
            "EhlersStoch": {},
            "Stochastic": {},
            "IFT_Stoch": {}
        },
        "Seasonality-Based Mean Reversion": {
            "HybridSeasonalSystem": {},
            "Halloween": {},
            "SeasonalTrading": {}
        }
    },
    "Reversal Strategies": {
        "Key Reversal Patterns": {
            "KeyRevLE": {},
            "KeyRevLX": {},
            "ReverseEMAStrat": {},
            "MajorBearMarketAwareStrat": {}
        },
        "Divergence-Based Reversals": {
            "BBDivergenceStrat": {},
            "RegressionDivergenceStrat": {},
            "ThreePeriodDivergence": {},
            "RSITrend": {}
        }
    },
    "Oscillator-Based Strategies": {
        "Relative Strength Index (RSI)": {
            "RSIStrat": {},
            "RSITrend": {}
        },
        "Other Oscillators": {
            "ElegantOscillatorStrat": {},
            "SimpleROCStrat": {},
            "VPNStrat": {},
            "SpectrumBarsLE": {},
            "UniversalOscillatorStrat": {}
        }
    },
    "Volatility-Based Strategies": {
        "ATRHighSMABreakoutsLE": {},
        "ATRTrailingStopLE": {},
        "ATRTrailingStopSE": {},
        "VolatilityBand": {},
        "VolSwitch": {},
        "VoltyExpanCloseLX": {}
    },
    "Support and Resistance Strategies": {
        "Pivot Points": {
            "CamarillaPointsStrat": {}
        },
        "Flag and Continuation Patterns": {
            "IntradayFlagFormationStrat": {}
        },
        "Swing-Based Strategies": {
            "SwingThree": {},
            "PriceSwing": {}
        }
    },
    "Market Sentiment Strategies": {
        "SentimentZone": {},
        "VIX_Timing": {}
    },
    "Stop-Loss and Profit Target Strategies": {
        "StopLossLX": {},
        "StopLossSX": {},
        "TrailingStopLX": {},
        "TrailingStopSX": {},
        "ProfitTargetLX": {},
        "ProfitTargetSX": {}
    },
    "Advanced and Miscellaneous Strategies": {
        "Advanced or Custom Strategies": {
            "AccumulationDistributionStrat": {},
            "AdvanceDeclineCumulative": {},
            "RSMKStrat": {},
            "TechnicalStockRatingStrat": {},
            "SVEZLRBPercBStrat": {}
        },
        "Multi-Asset and Multi-Currency Strategies": {
            "PairTradingLong": {},
            "PairTradingShort": {},
            "MultiCurrencyCorrelation": {}
        },
        "Specialized Strategies": {
            "GandalfProjectResearchSystem": {},
            "Stress": {},
            "ElegantOscillatorStrat": {}
        }
    }
}
