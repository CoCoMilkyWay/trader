# 用前须知

## xtdata提供和MiniQmt的交互接口，本质是和MiniQmt建立连接，由MiniQmt处理行情数据请求，再把结果回传返回到python层。使用的行情服务器以及能获取到的行情数据和MiniQmt是一致的，要检查数据或者切换连接时直接操作MiniQmt即可。

## 对于数据获取接口，使用时需要先确保MiniQmt已有所需要的数据，如果不足可以通过补充数据接口补充，再调用数据获取接口获取。

## 对于订阅接口，直接设置数据回调，数据到来时会由回调返回。订阅接收到的数据一般会保存下来，同种数据不需要再单独补充。

# 代码讲解

# 从本地python导入xtquant库，如果出现报错则说明安装失败
from xtquant import xtdata
import time
xtdata.reconnect()
# 设定一个标的列表
code_list = ["000001.SZ"]
# 设定获取数据的周期
period = "1d"

# 下载标的行情数据
if 1:
    ## 为了方便用户进行数据管理，xtquant的大部分历史数据都是以压缩形式存储在本地的
    ## 比如行情数据，需要通过download_history_data下载，财务数据需要通过
    ## 所以在取历史数据之前，我们需要调用数据下载接口，将数据下载到本地
    for i in code_list:
        xtdata.download_history_data(i,period=period,incrementally=True) # 增量下载行情数据（开高低收,等等）到本地
    print('finish downloading')
    xtdata.download_financial_data(code_list) # 下载财务数据到本地
    print('finish downloading financial')
    xtdata.download_sector_data() # 下载板块数据到本地
    # 更多数据的下载方式可以通过数据字典查询
    
    fieldList = ['CAPITALSTRUCTURE.total_capital', '利润表.净利润']
    stockList = ['600000.SH','000001.SZ']
    startDate = '20171209'
    endDate = '20171212'
    ContextInfo.get_financial_data(fieldList, stockList, startDate, endDate, report_type = 'report_time')