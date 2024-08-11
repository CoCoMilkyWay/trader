## 目录结构 & 文件说明
https://www.bilibili.com/video/BV1iB421k74C/?spm_id_from=333.788&vd_source=f78f53fd28f7a2e2c81dfd10d4ab858c
https://www.youtube.com/watch?v=ZX-Tp4zgJYc

### 止损
- 最大亏损法：这是最简单的止损方法，当买入个股的浮动亏损幅度达到某个百分点时进行止损。这个百分点根据你的风险偏好、交易策略和操作周期而定，例如超短线（T+1）的可以是1.5～3%，短线（5天左右）的可以是3～5%，中长线的可以是5～10%。这个百分点一旦定下来，就不可轻易改变，要坚决果断执行。
- 回撤止损： 如果买入之后价格先上升，达到一个相对高点后再下跌，那么可以设定从相对高点开始的下跌幅度为止损目标，这个幅度的具体数值也由个人情况而定，一般可以参照上面说的最大亏损法的百分点。另外还可以再加上下跌时间（即天数）的因素，例如设定在3天内回撤下跌5%即进行止损。回撤止损实际更经常用于止赢的情况。
- 横盘止损： 将买入之后价格在一定幅度内横盘的时间设为止损目标，例如可以设定买入后5天内上涨幅度未达到5%即进行止损。横盘止损一般要与最大亏损法同时使用，以全面控制风险。
- 移动均线止损： 短线、中线、长线投资者可分别用MA5、MA20、MA120移动均线作为止损点。此外，EMA、SMA均线的止损效果一般会比MA更好一些。MACD红柱开始下降也可以作为一个不错的止损点。成本均线止损： 成本均线比移动均线多考虑了成交量因素，总体来说效果一般更好一些。具体方法与移动均线基本相同。不过需要提醒的是，均线永远是滞后的指标，不可对其期望过高。另外在盘整阶段，你要准备忍受均线的大量伪信号。
- 布林通道止损： 在上升趋势中，可以用布林通道中位线作为止损点，也可以用布林带宽缩小作为止损点。
- 波动止损： 这个方法比较复杂，也是高手们经常用的，例如用平均实际价格幅度的布林通道，或者上攻力度的移动平均等作为止损目标。
- K线组合止损： 包括出现两阴夹一阳、阴后两阳阴的空头炮，或出现一阴断三线的断头铡刀，以及出现黄昏之星、穿头破脚、射击之星、双飞乌鸦、三只乌鸦挂树梢等典型见顶的K线组合等。
- K线形态止损： 包括股价击破头肩顶、M头、圆弧顶等头部形态的颈线位，一阴断三线的断头铡刀等。切线支撑位止损： 股价有效跌破趋势线或击穿支撑线，可以作为止损点。
- 筹码密集区止损： 筹码密集区对股价会产生很强的支撑或阻力作用，一个坚实的密集区被向下击穿后，往往会由原来的支撑区转化为阻力区。根据筹码密集区设置止损位，一旦破位立即止损出局。不过需要注意的是，目前市面上绝大多数的股票软件在处理筹码分布时没有考虑除权因素。如果你所持个股在近期（一般是一年以内）发生过幅度比较大的除权，则需防止软件标示出错误的筹码密集区位置。
- 筹码分布图上移止损： 筹码分布图上移的原因一般主要是高位放量，如果上移形成新的密集峰，则风险往往很大，应及时止损或止赢出局。
- 大盘止损： 对于大盘走势的判断是操作个股的前提，那种在熊市中“抛开大盘做个股”的说法是十分有害的。一般来说，大盘的系统性风险有一个逐渐累积的过程，当发现大盘已经处于高风险区，有较大的中线下跌可能时，应及时减仓，持有的个股即使处于亏损状态也应考虑卖出。

![alt text](image.png)

```
.
├── 📁 Bi # 笔
│   ├── 📄 BiConfig.py: 配置类
│   ├── 📄 BiList.py: 笔列表类
│   └── 📄 Bi.py: 笔类
├── 📁 Seg: 线段类
│   ├── 📄 Eigen.py: 特征序列类
│   ├── 📄 EigenFX.py: 特征序列分形类
│   ├── 📄 SegConfig.py: 线段配置类
│   ├── 📄 SegListComm.py: 线段计算框架通用父类
│   ├── 📄 SegListChan.py: 线段计算：根据原文实现
│   ├── 📄 SegListDef.py: 线段计算：根据定义实现
│   ├── 📄 SegListDYH.py: 线段计算：根据都业华1+1突破实现
│   └── 📄 Seg.py: 线段类
├── 📁 ZS: 中枢类
│   ├── 📄 ZSConfig.py: 中枢配置
│   ├── 📄 ZSList.py: 中枢列表类
│   └── 📄 ZS.py: 中枢类
├── 📁 KLine: K线类
│   ├── 📄 KLine_List.py: K线列表类
│   ├── 📄 KLine.py: 合并K线类
│   ├── 📄 KLine_Unit.py: 单根K线类
│   └── 📄 TradeInfo.py: K线指标类（换手率，成交量，成交额等）
├── 📁 BuySellPoint: 形态学买卖点类（即bsp）
│   ├── 📄 BSPointConfig.py: 配置
│   ├── 📄 BSPointList.py: 买卖点列表类
│   └── 📄 BS_Point.py: 买卖点类
├── 📁 Combiner: K线，特征序列合并器
│   ├── 📄 Combine_Item.py: 合并元素通用父类
│   └── 📄 KLine_Combiner.py: K线合并器
├── 📁 Common: 通用函数
│   ├── 📄 cache.py: 缓存装饰器，大幅提高计算性能
│   ├── 📄 CEnum.py: 所有枚举类，K线类型/方向/笔类型/中枢类型等
│   ├── 📄 ChanException.py: 异常类
│   ├── 📄 CTime.py: 缠论时间类（可处理不同级别联立）
│   ├── 📄 func_util.py: 通用函数
│   ├── 📄 send_msg_cmd.py: 消息推送
│   ├── 📄 tools.py: 工具类
│   ├── 📄 CommonThred.py: 线程类
│   └── 📄 TradeUtil.py: 交易通用函数
├── 📁 Config: 配置
│   ├── 📄 config.sh shell脚本读取配置
│   ├── 📄 demo_config.yaml: demo配置
│   ├── 📄 EnvConfig.py: python读取配置类
├── 📁 CustomBuySellPoint: 自定义动力学买卖点类（即cbsp）
│   ├── 📄 Strategy.py: 通用抽象策略父类
│   ├── 📄 CustomStrategy.py: demo策略1
│   ├── 📄 SegBspStrategy.py: demo策略2
│   ├── 📄 ExamStrategy.py: 生成买卖点判断试题的策略
│   ├── 📄 CustomBSP.py: 自定义买卖点
│   └── 📄 Signal.py: 信号类
├── 📁 DataAPI: 数据接口
│   ├── 📄 CommonStockAPI.py: 通用数据接口抽象父类
│   ├── 📄 AkShareAPI.py: akshare数据接口
│   ├── 📄 BaoStockAPI.py: baostock数据接口
│   ├── 📄 ETFStockAPI.py: ETF数据解耦接口
│   ├── 📄 FutuAPI.py: futu数据接口
│   ├── 📄 OfflineDataAPI.py: 离线数据接口
│   ├── 📄 MarketValueFilter.py: 股票市值过滤类
│   └── 📁 SnapshotAPI: 实时股价数据接口
│       ├── 📄 StockSnapshotAPI.py: 统一调用接口
│       ├── 📄 CommSnapshot.py: snapshot通用父类
│       ├── 📄 AkShareSnapshot.py: akshare接口，支持a股，etf，港股，美股
│       ├── 📄 FutuSnapshot.py: 富途接口，支持a股，港股，美股
│       ├── 📄 PytdxSnapshot.py: pytdx，支持A股，ETF
│       └── 📄 SinaSnapshot.py: 新浪接口，支持a股，etf，港股，美股
├── 📁 Math: 计算类
│   ├── 📄 BOLL.py: 布林线计算类
│   ├── 📄 MACD.py: MACD计算类
│   ├── 📄 Demark.py: Demark指标计算类
│   ├── 📄 OutlinerDetection.py: 离群点计算类
│   ├── 📄 TrendModel.py: 趋势类（支持均线，最大值，最小值）
│   └── 📄 TrendLine.py: 趋势线
├── 📁 ModelStrategy: 模型策略
│   ├── 📄 BacktestChanConfig.py: 回测配置
│   ├── 📄 backtest.py: 回测计算框架
│   ├── 📄 FeatureReconciliation.py: 特征离线在线一致性校验
│   ├── 📄 ModelGenerator.py: 训练模型通用父类
│   ├── 📁 models: 提供模型
│   │   ├── 📁 deepModel: 深度学习模型
│   │   │   ├── 📄 MLPModelGenerator.py: 深度学习模型
│   │   │   └── 📄 train_all_model.sh 拉起全流程训练预测评估脚本
│   │   ├── 📁 lightGBM
│   │   │   ├── 📄 LGBMModelGenerator.py: LGBM模型
│   │   │   └── 📄 train_all_model.sh 拉起全流程训练预测评估脚本
│   │   └── 📁 Xgboost
│   │       ├── 📄 train_all_model.sh 拉起全流程训练预测评估脚本
│   │       ├── 📄 XGBTrainModelGenerator.py: XGB模型
│   │       └── 📄 xgb_util.py: XGB便捷调用工具
│   └── 📁 parameterEvaluate: 策略参数评估
│       ├── 📄 eval_strategy.py: 评估策略收益类
│       ├── 📄 multi_cycle_test_data.sh: 多周期策略评估数据生成脚本
│       ├── 📄 multi_cycle_test.py: 多周期策略评估
│       ├── 📄 para_automl.py: Automl计算模型超参
│       ├── 📄 automl_verify.py: 验证Automl结果小脚本
│       ├── 📄 parse_automl_result.py: 解析Automl超参生成交易配置文件
│       └── 📁 AutoML FrameWork: Automl学习框架，本项目不专门提供
├── 📁 ChanModel: 模型
│   ├── 📄 CommModel.py: 通用模型抽象父类
│   ├── 📄 FeatureDesc.py: 特征注册
│   ├── 📄 Features.py: 特征计算
│   └── 📄 XGBModel.py: XGB模型 demo
├── 📁 OfflineData: 离线数据更新
│   ├── 📄 download_all_offline_data.sh 调度下载A股，港股，美股所有数据脚本
│   ├── 📄 ak_update.py: akshare更新港股美股A股离线数据
│   ├── 📄 bao_download.py: baostock下载全量A股数据
│   ├── 📄 bao_update.py: baostock增量更新数据
│   ├── 📄 etf_download.py: 下载A股ETF数据脚本
│   ├── 📄 futu_download.py: 更新futu港股数据
│   ├── 📄 offline_data_util.py: 离线数据更新通用工具类
│   └── 📁 stockInfo: 股票指标数据
│       ├── 📄 CalTradeInfo.py: 计算股票指标数据分布，分位数
│       ├── 📄 query_marketvalue.py: 计算股票市值分位数
│       └── 📄 run_market_value_query.sh 调度脚本
├── 📁 Plot: 画图类
│   ├── 📄 AnimatePlotDriver.py: 动画画图类
│   ├── 📄 PlotDriver.py: matplotlib画图引擎
│   ├── 📄 PlotMeta.py: 图元数据
│   └── 📁 CosApi: COS文件上传类
│       ├── 📄 minio_api.py: minio上传接口
│       ├── 📄 tencent_cos_api.py: 腾讯云cos上传接口
│       └── 📄 cos_config.py: 读取项目配置里面的cos配置参数
├── 📁 Trade: 交易引擎
│   ├── 📄 db_util.py: 数据库操作类
│   ├── 📄 FutuTradeEngine.py: futu交易引擎类
│   ├── 📄 MysqlDB.py: Mysql数据库类
│   ├── 📄 SqliteDB.py: SqliteDB数据库类
│   ├── 📄 OpenQuotaGen.py: 开仓交易手数策略类（用于控制仓位）
│   ├── 📄 TradeEngine.py: 交易引擎核心类
│   └── 📁 Script: 核心交易脚本
│       ├── 📄 update_data_signal.sh: 离线数据更新，信号计算调度脚本
│       ├── 📄 CheckOpenScore.py: 后验检查开仓是否准确
│       ├── 📄 ClosePreErrorOpen.py: 修复错误开仓
│       ├── 📄 MakeOpenTrade.py: 开仓
│       ├── 📄 OpenConfig_demo.yaml: 开仓参数配置
│       ├── 📄 OpenConfig.py: 开仓参数配置类
│       ├── 📄 RealTimeTracker.py: 实时跟踪是否遇到止损，止盈点
│       ├── 📄 RetradeCoverOrder.py: 修复未成功交易平仓单
│       ├── 📄 SignalMonitor.py: 信号计算
│       ├── 📄 StaticsChanConfig.py: 缠论计算配置
│       └── 📄 UpdatePeakPrice.py: 峰值股价更新（用于做动态止损）
├── 📁 Debug： debug工具
│   ├── 📁 cprofile_analysis: 性能分析
│   │   └── 📄 cprofile_analysis.sh 性能分析脚本
│   └── 📁 Notebook
│       └── 📄 xxx.ipynb  各种notebook
├── 📁 Script: 脚本汇总
│   ├── 📄 InitDB.py: 数据库初始化
│   ├── 📄 Install.sh 安装本框架脚本
│   ├── 📄 requirements.txt: pip requirements文件
│   ├── 📄 pip_upgrade.sh: pip更新股票数据相关的库
│   ├── 📄 run_backtest.sh 运行回测计算
│   ├── 📄 run_train_pipeline.sh 运行回测，指定模型训练预测评估，校验，全pipeline脚本
│   ├── 📁 cprofile_analysis: 性能分析
│   │   └── 📄 cprofile_analysis.sh 性能分析脚本
│   └── 📁 Notion: Notion数据表同步脚本
│      ├── 📄 DB_sync_Notion.py 交易数据库同步Notion脚本
│      └── 📁 notion: Notion API
│          ├── 📄 notion_api.py: Notion统一API接口
│          ├── 📄 block_driver.py: Notion块操作类
│          ├── 📄 prop_driver.py.py: Notion数据表属性操作类
│          ├── 📄 text.py: Notion 富文本操作类
│          └── 📄 secret.py: notion读取配置文件里面的参数
├── 📄 main.py: demo main函数
├── 📄 Chan.py: 缠论主类
├── 📄 ChanConfig.py: 缠论配置
├── 📄 ExamGenerator.py: 测试题生成API
├── 📄 LICENSE
└── 📄 README.md: 本文件



### CChanConfig重点关注配置
CChanConfig里面提供了很多的配置，其中很多人最容易被影响到自己计算结果的主要是这几个，它们的含义最好再仔细阅读一下readme相关解释：
- bi_strict：是否只用严格笔，默认为 Ture，其中这里的严格笔只考虑顶底分形之间相隔几个合并K线
- bi_fx_check：检查笔顶底分形是否成立的方法
- bi_end_is_peak: 笔的尾部是否是整笔中最低/最高, 默认为 True
- divergence_rate：1类买卖点背驰比例，即离开中枢的笔的 MACD 指标相对于进入中枢的笔，默认为 0.9
- min_zs_cnt：1类买卖点至少要经历几个中枢，默认为 1
- max_bs2_rate：2类买卖点那一笔回撤最大比例，默认为 0.618
    - 注：如果是 1.0，那么相当于允许回测到1类买卖点的位置
- zs_algo: 中枢算法，涉及到中枢是否允许跨段

## 取出缠论元素
- CChan这个类里面有个变量`kl_datas`，这是一个字典，键是 KL_TYPE（即级别，具体取值参见Common/CEnum），值是 CKLine_List 类；
- CKLine_List是所有元素取值的入口，关键成员是：
  - lst: List[CKLine]：所有的合并K线
  - bi_list：CBiList 类，管理所有的笔
  - seg_list：CSegListComm 类，管理所有的线段
  - zs_list：CZSList 类，管理所有的中枢
  - bs_point_lst：CBSPointList 类，管理所有的买卖点
  - 其余大部分人可能不关注的
    - segseg_list：线段的线段
    - segzs_list：线段中枢
    - seg_bs_point_lst：线段买卖点


### CKLine-合并K线
成员包括：
- idx：第几根
- CKLine.lst可以取到所有的单根K线变量 CKLine_Unit
- fx：FX_TYPE，分形类型
- dir：方向
- pre,next：前一/后一合并K线
- high：高点
- low：低点

#### CKLine_Unit-单根K线
成员包括：
- idx：第几根
- time
- low/close/open/high
- klc：获取所属的合并K线（即CKLine）变量
- sub_kl_list: List[CKLine_Unit] 获取次级别K线列表，范围在这根K线范围内的
- sup_kl: CKLine_Unit 父级别K线（CKLine_Unit）


### bi_list-笔管理类
成员包含：
- bi_list: List[CBi]，每个元素是一笔

这个类的实现基本可以不用关注，除非你想实现自己的画笔算法

#### CBi- 笔类
成员包含：
- idx：第几笔
- dir：方向，BI_DIR类
- is_sure：是否是确定的笔
- klc_lst：List[CKLine]，该笔全部合并K线
- seg_idx：所属线段id
- parent_seg:CSeg 所属线段
- next/pre：前一/后一笔

可以关注一下这里面实现的一些关键函数：
- _high/_low
- get_first_klu/get_last_klu：获取笔第一根/最后一根K线
- get_begin_klu/get_end_klu：获取起止K线
  - 注意一下：和get_first_klu不一样的地方在于，比如下降笔，这个获取的是第一个合并K线里面high最大的K线，而不是第一个合并K线里面的第一根K线；
- get_begin_val/get_end_val：获取笔起止K线的价格
  - 比如下降笔get_begin_val就是get_begin_klu的高点


### CSegListComm-线段管理类
- lst: List[CSeg] 每一个成员是一根线段

这个类的实现基本可以不用关注，除非你想实现自己的画段算法，参照提供的几个demo，实现这个类的子类即可；

#### CSeg：线段类
成员包括：
- idx
- start_bi：起始笔
- end_bi：终止笔
- is_sure：是否已确定
- dir：方向，BI_DIR类
- zs_lst: List[CZS] 线段内中枢列表
- pre/next：前一/后一线段
- bi_list: List[CBi] 线段内笔的列表

关注的一些关键函数和CBi里面一样，都已实现同名函数，如：
- _high/_low
- get_first_klu/get_last_klu
- get_begin_klu/get_end_klu
- get_begin_val/get_end_val


### CZSList-中枢管理类
- zs_lst: List[CZS] 中枢列表


#### CZS：中枢类
成员包括：
- begin/end：起止K线CKLine_Unit
- begin_bi/end_bi：中枢内部的第一笔/最后一笔
- bi_in：进中枢的那一笔（在中枢外面）
- bi_out：出中枢的那一笔（在中枢外面，不一定存在）
- low/high：中枢的高低点
- peak_low/peak/high：中枢内所有笔的最高/最低值
- bi_lst：中枢内笔列表
- sub_zs_lst：子中枢（如果出现过中枢合并的话）


### CBSPointList-买卖点管理类
- lst：List[CBS_Point] 所有的买卖点


#### CBS_Point：买卖点类
成员包括：
- bi：所属的笔（买卖点一定在某一笔末尾）
- Klu：所在K线
- is_buy：True为买点，False为卖点
- type：List[BSP_TYPE] 买卖点类别，是个数组，比如2，3类买卖点是同一个

