"""
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| Model                           | Pros                                                        | Cons                                                        |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| LSTM (Long Short-Term Memory)   | - Handles long-range dependencies well.                     | - Computationally expensive for long sequences.             |
|                                 | - Mitigates vanishing gradient problem.                     | - Slower training due to more parameters.                   |
|                                 | - Suitable for complex sequences (e.g., time-series, text). |                                                             |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| GRU (Gated Recurrent Unit)      | - Simpler architecture (fewer parameters).                  | - Struggles with very long sequences compared to LSTM.      |
|                                 | - Faster training than LSTM, often similar performance.     | - Fewer gates may limit flexibility for complex tasks.      |
|                                 | - Effective at learning temporal dependencies.              |                                                             |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| RNN (Recurrent Neural Network)  | - Simple and easy to implement.                             | - Struggles with long-range dependencies(vanishing gradient)|
|                                 | - Good for short-term dependencies.                         | - Not suitable for complex tasks without enhancements.      |
|                                 | - Lower computational cost than LSTM/GRU.                   |                                                             |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| BiLSTM (Bidirectional LSTM)     | - Captures both past and future context.                    | - More computationally expensive than unidirectional LSTM.  |
|                                 | - Suitable for tasks where future context is important.     | - Increases train time due to processing in both directions.|
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| 1D CNN (1D Convolutional        | - Efficient for local patterns in sequential data.          | - Limited at capturing long-term deps comparing to RNNs.    |
| Neural Networks)                |                                                             |                                                             |
|                                 | - Faster than RNNs for certain cases.                       | - Requires tuning of kernel size for optimal performance.   |
|                                 | - Can handle high-dimensional inputs.                       |                                                             |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| Attention Mechanisms            | - Highly parallelizable, faster training.                   | - Requires large amounts of data for effective training.    |
| (e.g., Transformer)             |                                                             |                                                             |
|                                 | - Great at capturing long-range dependencies.               | - Computationally intensive with long sequences.            |
|                                 | - Scalable to large datasets and sequences.                 | - Complexity in hyperparameter tuning.                      |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+

+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Normalization Type     | Formula                                          | Best Used For                              | Characteristics                                  |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Z-Score Normalization  | (x - μ) / σ                                      | - Normally distributed data                | - Mean = 0                                       |
| (StandardScaler)       | Transforms to mean 0, std dev 1                  | - Features on similar scales               | - Std Dev = 1                                    |
|                        |                                                  | - When outliers are not extreme            | - Sensitive to outliers                          |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Min-Max Scaling        | (x - min(x)) / (max(x) - min(x))                 | - When you need bounded range              | - Preserves zero values                          |
| (MinMaxScaler)         | Scales to fixed range (default [0,1])            | - Neural network inputs                    | - Sensitive to outliers                          |
|                        |                                                  | - Image processing                         | - Does not handle new min/max in test data       |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Robust Scaling         | (x - median) / IQR                               | - Data with many outliers                  | - Uses median instead of mean                    |
| (RobustScaler)         | Uses median and interquartile range              | - Skewed distributions                     | - Less affected by extreme values                |
|                        |                                                  | - Financial data with extreme values       | - Preserves shape of distribution                |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| MaxAbs Scaling         | x / max(abs(x))                                  | - Sparse data                              | - Preserves sign of original values              |
| (MaxAbsScaler)         | Scales by maximum absolute value                 | - When zero is meaningful                  | - Does not center data                           |
|                        |                                                  | - Machine learning with sparse features    | - Bounded between -1 and 1                       |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Quantile Transformation| Transforms to uniform/normal distribution        | - Non-gaussian distributions               | - Makes data look like normal distribution       |
| (QuantileTransformer)  | Equalizes feature distributions                  | - When feature distributions differ        | - Can handle non-linear transformations          |
|                        |                                                  | - Machine learning with varied features    | - Destroys sparseness                            |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Power Transformation   | x^λ or log(x)                                    | - Right-skewed data                        | - Stabilizes variance                            |
| (PowerTransformer)     | Stabilizes variance and makes data more Gaussian | - Financial ratios                         | - Handles positive values                        |
|                        |                                                  | - Economic indicators                      | - Different methods like Yeo-Johnson, Box-Cox    |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
"""

"""
FEATURE PROPERTIES FOR ENSEMBLE VS NEURAL NETWORK METHODS
+-----------------------+--------------------------------+--------------------------------+
| Property              | Ensemble/Tree-Based Methods    | Neural Networks                |
+-----------------------+--------------------------------+--------------------------------+
| Value Range           | - Raw values acceptable        | - Scale to [-1, 1] or [0, 1]   |
|                       | - No scaling needed            | - Min-max or standard scaling  |
|                       | - Unbounded OK                 | - Remove values beyond ±3σ     |
+-----------------------+--------------------------------+--------------------------------+
| Distribution          | - Any distribution OK          | - Normal/Uniform preferred     |
|                       | - Multimodal OK                | - Transform skewed features    |
|                       | - No transformation needed     | - Log/Box-Cox for heavy tails  |
+-----------------------+--------------------------------+--------------------------------+
| Mean                  | - Any value OK                 | - ~0 after standardization     |
|                       | - No centering needed          | - Center before training       |
+-----------------------+--------------------------------+--------------------------------+
| Variance              | - Non-zero required            | - ~1 after standardization     |
|                       | - No upper bound               | - Similar scale across features|
|                       | - High variance OK             | - Bounded variance preferred   |
+-----------------------+--------------------------------+--------------------------------+
| Skewness              | - Any value acceptable         | - Between -1 and 1             |
|                       | - No transformation needed     | - Transform if outside range   |
|                       | - Handles asymmetry well       | - Symmetric preferred          |
+-----------------------+--------------------------------+--------------------------------+
| Kurtosis              | - Any value acceptable         | - Close to 3 (normal dist)     |
|                       | - Handles heavy tails well     | - Transform heavy tails        |
|                       | - No transformation needed     | - Avoid extreme peaks          |
+-----------------------+--------------------------------+--------------------------------+
| Missing Values        | - Up to 25-30% acceptable      | - Less than 5% preferred       |
|                       | - Handle natively              | - Requires imputation          |
|                       | - Can use as split point       | - Use mean/median/mode         |
+-----------------------+--------------------------------+--------------------------------+
| Outliers              | - Robust to outliers           | - Remove or cap at ±3σ         |
|                       | - Can keep as is               | - Use robust scaling           |
|                       | - May improve splits           | - Consider Winsorization       |
+-----------------------+--------------------------------+--------------------------------+
| Cardinality           | - High cardinality OK          | - Low cardinality preferred    |
| (Categorical)         | - Up to 50-100 levels OK       | - One-hot if < 15 levels       |
|                       | - Can handle rare cats         | - Embedding for high cardinal  |
+-----------------------+--------------------------------+--------------------------------+
| Interactions          | - Auto-detected                | - Need manual engineering      |
|                       | - No preprocessing needed      | - Cross-features helpful       |
|                       | - Handles non-linearity        | - Polynomial features help     |
+-----------------------+--------------------------------+--------------------------------+
| Feature Independence  | - Handles correlation well     | - Independent preferred        |
|                       | - No decorrelation needed      | - PCA/decorrelation helps      |
|                       | - Redundancy OK                | - Remove highly correlated     |
+-----------------------+--------------------------------+--------------------------------+
| Zero/Constant Values  | - Auto-filtered                | - Remove constant features     |
|                       | - Low variance OK              | - Remove near-zero variance    |
|                       | - Sparse data OK               | - Sparse inputs challenging    |
+-----------------------+--------------------------------+--------------------------------+
| Numeric Resolution    | - Any precision OK             | - 32-bit float typical         |
|                       | - Integer or float OK          | - Consistent precision needed  |
|                       | - Binary values OK             | - Normalize if large scale diff|
+-----------------------+--------------------------------+--------------------------------+
| Feature Scale         | - Any range acceptable         | - Similar scales crucial       |
| Relationships         | - No relative scaling needed   | - Normalize across features    |
|                       | - Preserves relationships      | - Balance feature importance   |
+-----------------------+--------------------------------+--------------------------------+
"""

"""
WEAKNESSES OF ENSEMBLE TREE-BASED METHODS
+---------------------------+------------------------------------------------+
| Category                  | Limitations & Issues                           |
+---------------------------+------------------------------------------------+
| Computational Resources   | - Large memory footprint for model storage     |
|                           | - Slow inference with deep trees               |
|                           | - Training time scales poorly (N*logN)         |
|                           | - High RAM usage during training               |
|                           | - Parallel training limited by CPU cores       |
+---------------------------+------------------------------------------------+
| Feature Space Handling    | - Inefficient with sparse high-dim features    |
|                           | - Cannot learn natural linear combinations     |
|                           | - Manual feature crossing often needed         |
|                           | - Many trees needed for simple linear patterns |
|                           | - Poor handling of cyclic/periodic features    |
+---------------------------+------------------------------------------------+
| Optimization & Training   | - Discrete optimization (no smooth gradients)  |
|                           | - No end-to-end training capability            |
|                           | - Hard to incorporate domain constraints       |
|                           | - Limited transfer learning ability            |
|                           | - Cannot freeze/fine-tune partial model        |
+---------------------------+------------------------------------------------+
| Generalization Issues     | - Cannot extrapolate beyond training range     |
|                           | - Poor performance in sparse regions           |
|                           | - Rectangular decision boundaries only         |
|                           | - Limited smooth boundary approximation        |
|                           | - Struggles with continuous symmetries         |
+---------------------------+------------------------------------------------+
| Statistical Limitations   | - No native probability calibration            |
|                           | - Uncertainty requires special methods         |
|                           | - Cannot incorporate Bayesian priors           |
|                           | - Biased importance for correlated features    |
|                           | - No natural confidence intervals              |
+---------------------------+------------------------------------------------+
| Performance Ceiling       | - Plateaus with increased data size            |
|                           | - Exponential data needs with dimensions       |
|                           | - Cannot learn certain function classes        |
|                           | - Limited representation power                 |
|                           | - Inefficient for some pattern types           |
+---------------------------+------------------------------------------------+
| Model Properties          | - Prone to overfitting with deep trees         |
|                           | - Hard to interpret individual predictions     |
|                           | - No natural online learning                   |
|                           | - Model size grows with complexity             |
|                           | - Limited model compression options            |
+---------------------------+------------------------------------------------+
| Deployment Challenges     | - Large deployment package size                |
|                           | - High latency for real-time inference         |
|                           | - Resource-intensive serving                   |
|                           | - Complex versioning with ensembles            |
|                           | - Difficult to deploy on edge devices          |
+---------------------------+------------------------------------------------+
"""

"""
https://github.com/UFund-Me/Qbot

经典策略
交易对象	选股	择时	风险控制 (组合、仓位管理)
股票/期货/虚拟货币	
布林线均值回归 ('2022)
移动均线+KDJ
简单移动均线
情绪指标ARBR
阿隆指标(趋势交易)
LightGBM 预测
SVM 预测
LSTM时序预测
强化学习预测
Q-Leaning预测
随机森林预测
RSI背离策略
麻雀优化算法SSA
随机相对强弱指数 StochRSI
小市值 ('2021)
市场低估值策略
RSRS择时
配对交易
传统指标（对应下方Qbot支持的指标 这里）
布林线均值回归 ('2022)
移动均线+KDJ
简单移动均线
双均线策略 ('2022)
情绪指标ARBR
阿隆指标(趋势交易)
LightGBM 预测
SVM 预测
LSTM时序预测
强化学习预测
Q-Leaning预测
随机森林预测
RSI背离策略
麻雀优化算法SSA
随机相对强弱指数 StochRSI
因子组合
RSI和CCI组合
MACD和ADX指标
MACD和KDJ指标
多因子交易
alphalens多因子交易
多策略整合
组合策略
指数增强 ('2022)
经典策略
多因子选股 ('2023)
指数增强 ('2022)
Alpha对冲 ('2022)
网格交易
双均线策略 ('2022)
拐点交易 ('2022)
趋势交易
海龟策略
动态平衡策略
Kurtosis Portfolio组合策略 ('2023)
指数增强 ('2022)
Alpha对冲 ('2022)
动态平衡策略
多因子自动组合策略
基金	
4433法则 ('2022)
对冲策略：指数型+债券型对冲
组合策略：多因子组合配置
组合策略：惠赢智能算法1
组合策略：择时多策略
组合策略：智赢多因子1
同上
智能策略
GBDT	RNN	Reinforcement Learning	🔥 Transformer	🔥 LLM
GBDT
XGBoost (KDD'2016)
LightGBM (NIPS'2017)
Catboost (NIPS'2018)
BOOST
DoubleEnsemble (ICDM'2020)
TabNet (ECCV'2022)
LR
Line Regression ('2020)
CNN
MLP (CVPRW'2020)
GRU (ICCVW'2021)
ImVoxelNet (WACV'2022)
TabNet (AAAI'2019)
RNN
LSTM (Neural Computation'2017)
ALSTM (IJCAI'2022)
ADARNN (KDD'2021)
ADD (CoRL'2020)
KRNN ()
Sandwich ()
TFT (IJoF'2019)
GATs (NIPS'2017)
SFM (KDD'2017)
Transformer (NeurIPS'2017)
TCTS (ICML'2021)
TRA (KDD'2021)
TCN (KDD'2018)
IGMTF (KDD'2021)
HIST (KDD'2018)
Localformer ('2021)
ChatGPT
FinGPT
Benchmark and Model zoo
Results and models are available in the model zoo. AI strategies is shown at here, local run python backend/pytrader/strategies/workflow_by_code.py, also provide Binder

👉 点击展开查看具体AI模型benchmark结果

交易指标/因子
包含但不限于alpha-101、alpha-191，以及基于deap实现的因子自动生成算法。

EMA(简单移动均线)
MACD(指数平滑异同平均线)
KDJ(随机指标)
RSRS(阻力支撑相对强度)
RSI(相对强弱指标)
StochRSI(随机相对强弱指数)
BIAS(乖离率)
BOLL(布林线指标)
OBV(能量潮)
SAR(抛物转向)
VOL(成交量)
PSY(心理线)
ARBR(人气和意愿指标)
CR(带状能力线)
BBI(多空指标)
EMV(简易波动指标)
TRIX(三重指数平滑移动平均指标)
DMA(平均线差)
DMI(趋向指标)
CCI(顺势指标)
ROC(变动速率指标, 威廉指标)
ENE(轨道线)  # 轨道线（ENE）由上轨线(UPPER)和下轨线(LOWER)及中轨线(ENE)组成，
            # 轨道线的优势在于其不仅具有趋势轨道的研判分析作用，也可以敏锐的觉察股价运行过程中方向的改变
SKDJ(慢速随机指标)
LWR(慢速威廉指标)  # 趋势判断指标
市盈率
市净率
主力意愿(收费)
买卖差(收费)
散户线(收费)
分时博弈(收费)
买卖力道(收费)
行情趋势(收费)
MTM(动量轮动指标)(收费)
MACD智能参数(收费)
KDJ智能参数(收费)
RSI智能参数(收费)
WR智能参数(收费)
Qbot智能预测(收费)
Qbot买卖强弱指标(收费)
"""