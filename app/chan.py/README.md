## 目录结构 & 文件说明
https://www.bilibili.com/video/BV11y411e7bA/?spm_id_from=333.788.recommend_more_video.0&vd_source=f78f53fd28f7a2e2c81dfd10d4ab858c
https://www.bilibili.com/video/BV1iB421k74C/?spm_id_from=333.788&vd_source=f78f53fd28f7a2e2c81dfd10d4ab858c



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
