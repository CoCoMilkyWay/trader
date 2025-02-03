# +-------------------+-------------------------+------------------------+-------------------------+-------------------------+---------------------------+
# | Model             | Core Mechanism          | Key Parameters         | Risk Management         | Market Conditions       | Limitations               |
# +-------------------+-------------------------+------------------------+-------------------------+-------------------------+---------------------------+
# | Stoikov-          | - Reservation price     | - Risk aversion (γ)    | - Inventory penalty     | - Continuous trading    | - Assumes normal dist.    |
# | Avellaneda        | - Inventory risk        | - Volatility (σ)       | - Time horizon effect   | - Liquid markets        | - Simple arrival rates    |
# |                   | - Time decay            | - Order arrival rate   | - Position limits       | - Mean-reverting        | - Single asset focus      |
# +-------------------+-------------------------+------------------------+-------------------------+-------------------------+---------------------------+
# | Glosten-Milgrom   | - Information asymmetry | - Informed trader %    | - Adverse selection     | - Sequential trading    | - Single unit trades      |
# |                   | - Belief updating       | - Signal probability   | - Probability-based     | - Information-sensitive | - No inventory effect     |
# |                   | - Quote adjustment      | - True value dist.     | - Quote skewing         | - Discrete prices       | - Simplified time         |
# +-------------------+-------------------------+------------------------+-------------------------+-------------------------+---------------------------+
# | Ho-Stoll          | - Portfolio balance     | - Return variance      | - Portfolio VaR         | - Multiple assets       | - Static optimization     |
# |                   | - Dealer optimization   | - Correlation matrix   | - Cross-asset hedging   | - Correlated markets    | - Period-based            |
# |                   | - Transaction costs     | - Risk tolerance       | - Diversification       | - Regular flow          | - Known distributions     |
# +-------------------+-------------------------+------------------------+-------------------------+-------------------------+---------------------------+
# | Kyle              | - Strategic trading     | - Noise trader vol.    | - Price impact          | - Batch auctions        | - Single informed trader  |
# |                   | - Price discovery       | - Information value    | - Information leakage   | - Price discovery       | - Linear strategies only  |
# |                   | - Market efficiency     | - Liquidity param.     | - Market depth          | - Informed trading      | - Fixed time horizon      |
# +-------------------+-------------------------+------------------------+-------------------------+-------------------------+---------------------------+
# | Amihud-Mendelson  | - Inventory control     | - Holding costs        | - Position boundaries   | - Dealer markets        | - Simplified arrival      |
# |                   | - Cost optimization     | - Processing costs     | - Inventory bands       | - Regular flow          | - Static boundaries       |
# |                   | - Return maximization   | - Return rates         | - Band adjustment       | - Stable spreads        | - Fixed costs             |
# +-------------------+-------------------------+------------------------+-------------------------+-------------------------+---------------------------+
# | Roll              | - Bid-ask bounce        | - Effective spread     | - Minimal               | - Efficient markets     | - No inventory effects    |
# |                   | - Trade direction       | - Trade size           | - Trade-based only      | - Random walk prices    | - No information aspect   |
# |                   | - Price formation       | - Quote revision       | - Spread-based          | - Continuous trading    | - Simplified dynamics     |
# +-------------------+-------------------------+------------------------+-------------------------+-------------------------+---------------------------+
# | Modern ML/RL      | - State representation  | - Network structure    | - Dynamic adjustment    | - Any market regime     | - Black box decisions     |
# |                   | - Policy optimization   | - Reward function      | - Continuous learning   | - Changing conditions   | - Training data needed    |
# |                   | - Experience replay     | - Learning rate        | - Risk constraints      | - Multiple assets       | - Overfitting risk        |
# +-------------------+-------------------------+------------------------+-------------------------+-------------------------+---------------------------+
# | Hybrid Grid-MM    | - Grid + MM fusion      | - Grid spacing         | - Combined inventory    | - Range-bound markets   | - Complex implementation  |
# |                   | - Dynamic adjustment    | - Inventory params     | - Multi-level risk      | - Volatile markets      | - Parameter tuning        |
# |                   | - Adaptive spacing      | - Vol scaling          | - Position + Grid       | - Trending markets      | - Regime detection        |
# +-------------------+-------------------------+------------------------+-------------------------+-------------------------+---------------------------+

# Potential optimizations over grid strategies employing some MM concepts

# +------------------+--------------------------------+--------------------------------+--------------------------------+
# | Method           | Core Logic & Principles        | Implementation Details         | Market Impact & Requirements   |
# +------------------+--------------------------------+--------------------------------+--------------------------------+
# | Volatility-      | - Adapts grid width to market  | - Multiple volatility metrics  | FAVORABLE:                     |
# | Based            |   uncertainty                  | - Expands in high vol          | - Range-bound markets          |
# | Adjustment       | - Prevents over-trading in     | - Contracts in low vol         | - Clear regime shifts          |
# |                  |   volatile periods             | - Vol trend detection          |                                |
# |                  | - Risk-adjusted positioning    | - Size adjustment with vol     | CHALLENGES:                    |
# |                  | - Vol-regime detection         | - Stop distance scaling        | - Lag in detection             |
# |                  | - Uncertainty pricing          | - Rebalance triggers           | - Transition whipsaws          |
# +------------------+--------------------------------+--------------------------------+--------------------------------+
# | Market Making    | - Adapts MM principles         | - Order flow tracking          | FAVORABLE:                     |
# | Enhanced         | - Order book pressure use      | - Inventory-based quotes       | - Liquid markets               |
# |                  | - Flow + inventory signals     | - Dynamic spread adjust        | - Clear order flow             |
# |                  | - Price impact awareness       | - Risk position limits         |                                |
# |                  | - Adverse selection            | - Flow pattern analysis        | CHALLENGES:                    |
# |                  | - Market microstructure        | - Impact estimation            | - Complex implementation       |
# |                  |   integration                  | - Quote optimization           | - Heavy data requirements      |
# +------------------+--------------------------------+--------------------------------+--------------------------------+
# | Mean             | - Price return tendency        | - Equilibrium detection        | FAVORABLE:                     |
# | Reversion        | - Statistical arbitrage        | - Deviation-based weights      | - Stable pair statistics       |
# | Based            | - Probability entry system     | - Position size scaling        | - Clear mean reversion         |
# |                  | - Confidence intervals         | - Multi-timeframe analysis     |                                |
# |                  | - Deviation risk scaling       | - Correlation monitoring       | CHALLENGES:                    |
# |                  | - Statistical validation       | - Regime change detection      | - Regime shifts                |
# |                  |                                | - Boundary conditions          | - Statistical breakdowns       |
# +------------------+--------------------------------+--------------------------------+--------------------------------+
# | Time             | - Session patterns             | - Volume profile by time       | FAVORABLE:                     |
# | Adaptive         | - Calendar effects             | - Seasonality integration      | - Regular patterns             |
# |                  | - Activity cycles              | - Dynamic time windows         | - Session-based markets        |
# |                  | - Periodic volatility          | - Event adjustments            |                                |
# |                  | - Trading hour patterns        | - Multi-zone handling          | CHALLENGES:                    |
# |                  | - Seasonal adjustments         | - Activity thresholds          | - Pattern irregularity         |
# |                  |                                | - Rebalance scheduling         | - Event disruptions            |
# +------------------+--------------------------------+--------------------------------+--------------------------------+
# | Cost             | - Fee structure optimization   | - Break-even analysis          | FAVORABLE:                     |
# | Optimized        | - Rebate maximization          | - Minimum profit distance      | - High fee markets             |
# |                  | - Capital efficiency           | - Tier-based spacing           | - Clear fee structures         |
# |                  | - ROI optimization             | - Rebate targeting             |                                |
# |                  | - Transaction cost             | - Position size by cost        | CHALLENGES:                    |
# |                  |   minimization                 | - Capital allocation           | - Fee structure changes        |
# |                  |                                | - Efficiency metrics           | - Complex rebate systems       |
# +------------------+--------------------------------+--------------------------------+--------------------------------+
# | Position         | - Inventory risk management    | - Risk-adjusted spacing        | FAVORABLE:                     |
# | Weighted         | - Capital utilization          | - Position-based sizing        | - Clear risk limits            |
# |                  | - Risk-based adjustments       | - Level exposure limits        | - Stable market conditions     |
# |                  | - Balance optimization         | - PnL integration              |                                |
# |                  | - Exposure control             | - Risk monitoring              | CHALLENGES:                    |
# |                  | - VaR constraints              | - Rebalance triggers           | - Position complexity          |
# |                  |                                | - Stop placement               | - Multi-asset risk             |
# +------------------+--------------------------------+--------------------------------+--------------------------------+
# | Order Flow       | - Book pressure analysis       | - Imbalance detection          | FAVORABLE:                     |
# | Based            | - Volume distribution          | - Level importance scoring     | - Transparent books            |
# |                  | - Trade flow patterns          | - Density adjustments          | - Consistent flow              |
# |                  | - Depth utilization            | - Fill probability             |                                |
# |                  | - Queue positioning            | - Smart routing                | CHALLENGES:                    |
# |                  | - Toxicity analysis            | - Queue optimization           | - Dark pools                   |
# |                  |                                | - Flow metrics                 | - Book manipulation            |
# +------------------+--------------------------------+--------------------------------+--------------------------------+
# | Machine          | - Pattern recognition          | - Feature engineering          | FAVORABLE:                     |
# | Learning         | - Adaptive parameters          | - Regime classification        | - Clear patterns               |
# | Enhanced         | - Relationship modeling        | - Dynamic adjustments          | - Sufficient data              |
# |                  | - Auto parameter tuning        | - Ensemble methods             |                                |
# |                  | - Historical pattern           | - Prediction integration       | CHALLENGES:                    |
# |                  |   learning                     | - Risk constraints             | - Overfitting                  |
# |                  |                                | - Reinforcement learning       | - Regime adaptation            |
# +------------------+--------------------------------+--------------------------------+--------------------------------+
# | Geometric/       | - Price scale independence     | - Ratio-based spacing          | FAVORABLE:                     |
# | Arithmetic       | - Fixed vs percentage          | - Hybrid approaches            | - Wide price ranges            |
# | Spacing          | - Market structure matching    | - Scale-adjusted sizing        | - Clear price structures       |
# |                  | - Range adaptation             | - Density optimization         |                                |
# |                  | - Mixed approach               | - Boundary handling            | CHALLENGES:                    |
# |                  |   optimization                 | - Transition logic             | - Price regime changes         |
# |                  |                                | - Method selection             | - Implementation complexity    |
# +------------------+--------------------------------+--------------------------------+--------------------------------+


"""
有一种天天赚钱的策略叫马丁，现在我在fmz平台上研究这种类型的策略用于数字货币合约交易。纯马丁翻倍加仓的风险是很大的，盈亏比也很差，所以我现在用Garch模型分析和预测波动率，再通过拟合参数回归的方式调节头寸和加仓间隔，依据大量技术指标的组合共振以及对于近期尾部风险的建模，参考市场深度信息和瞬时变动加入多种风控系统并且动态对冲部分整体市场的贝塔风险，简而言之，就是不会无脑加仓的趋势跟踪+对冲马丁。

"""