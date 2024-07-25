## 数据
支持获取新闻公告数据
xtdata.get_market_data()系列函数 数据周期：announcement

支持获取涨跌停连板数据
xtdata.get_market_data()系列函数 数据周期：limitupperformance

支持获取港股通持股明细数据
xtdata.get_market_data()系列函数 数据周期：hktdetails,hktstatistics

支持获取外盘的行情数据（需购买相应服务）
行情订阅xtdata.subscribe_quote()和行情获取xtdata.get_market_data()系列函数，支持美股品种的获取

## 功能
支持python3.12版本

xtdata支持选择端口范围，在范围内自动连接

添加函数xtdata.get_full_kline() 批量获取当日K线数据（需要开启K线全推）

支持订阅vba模型（连接投研端）
xtdata.subscribe_formula()

token模式下初始化全推市场可选
xtdatacenter.set_wholequote_market_list()

token模式下行情连接优选机制调整
xtdatacenter.set_allow_optmize_address()会使用第一个地址作为全推连接

token模式下期货周末夜盘数据时间模式可选,可以选择展示为周一凌晨时间或真实的周六凌晨时间
xtdatacenter.set_future_realtime_mode()