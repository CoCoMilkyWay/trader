#coding:utf-8
import time, datetime, traceback, sys
from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant

class _a(): # data container
    pass
A = _a()
A.bought_list = []
A.hsa = xtdata.get_stock_list_in_sector('沪深A股')

def interact():
    """执行后进入repl模式"""
    import code
    code.InteractiveConsole(locals=globals()).interact()
xtdata.download_sector_data()


import sys
import pprint
class MultiStream:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        """ Write data to all streams. """
        for stream in self.streams:
            stream.write(data)
    def flush(self):
        """ Flush all streams. """
        for stream in self.streams:
            stream.flush()

import time
import sys
import os
from functools import wraps
def monitor_callback(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Access the previous call time and count stored as function attributes
        if not hasattr(wrapper, 'last_call_time'):
            wrapper.last_call_time = None
            wrapper.call_count = 0
            wrapper.cumulative_file_size = 0
            wrapper.total_interval = 0

        # Call the callback function
        result = func(*args, **kwargs)

        # Update the call count
        wrapper.call_count += 1

        # Calculate the callback interval
        current_time = time.time()
        if wrapper.last_call_time is not None:
            interval = current_time - wrapper.last_call_time
            wrapper.total_interval += interval
            print(f"Callback Interval: {interval:.2f} seconds")
        wrapper.last_call_time = current_time

        # Calculate the dictionary size (in bytes, assuming it's serialized)
        dict_size = sys.getsizeof(result)
        print(f"Dictionary Size: {dict_size} bytes")

        # Write the result to a file and calculate file size increment
        filename = f'callback_output_{wrapper.call_count}.dat'
        with open(filename, 'w') as f:
            f.write(str(result))
        file_size = os.path.getsize(filename)
        wrapper.cumulative_file_size += file_size

        # Print statistics
        if wrapper.call_count > 1:
            average_interval = wrapper.total_interval / (wrapper.call_count - 1)
            print(f"Average Callback Interval: {average_interval:.2f} seconds")
        print(f"Cumulative File Size: {wrapper.cumulative_file_size} bytes")

        return result
    return wrapper

filename = 'full-tick.txt'
def print_to_stdout_and_file(data, filename):
    with open(filename, 'a') as file:
        print(data, file=file)
        # multi_stream = MultiStream(sys.stdout, file)
        # pp = pprint.PrettyPrinter(indent=4, width=120, stream=multi_stream)
        # pp.pprint(data)

# @monitor_callback
def subscribed_data_callback(data):
    now = datetime.datetime.now()
    print(now, ': ', sys.getsizeof(data))
    print_to_stdout_and_file(data, filename)
    # print('subscribed_data_callback: ', sys.getsizeof(data))
    # for stock in data:
    #     if stock not in A.hsa:
    #         continue
    #     print_to_stdout_and_file(stock, filename)
    #     cuurent_price = data[stock][0]['lastPrice']
    #     pre_price = data[stock][0]['lastClose']
    #     ratio = cuurent_price / pre_price - 1 if pre_price > 0 else 0
    #     if ratio > 0.09 and stock not in A.bought_list:
    #         print(f"{now} 最新价 买入 {stock} 200股")
    #         # async_seq = xt_trader.order_stock_async(acc, stock, xtconstant.STOCK_BUY, 1, xtconstant.LATEST_PRICE, -1, 'strategy_name', stock)
    #         A.bought_list.append(stock)

class MyXtQuantTraderCallback(XtQuantTraderCallback):
    def on_disconnected(self):
        print(datetime.datetime.now(),'连接断开回调')
    def on_stock_order(self, order):
        print(datetime.datetime.now(), '委托回调', order.order_remark)
    def on_stock_trade(self, trade):
        print(datetime.datetime.now(), '成交回调', trade.order_remark)
    def on_order_error(self, order_error):
        # print("on order_error callback")
        # print(order_error.order_id, order_error.error_id, order_error.error_msg)
        print(f"委托报错回调 {order_error.order_remark} {order_error.error_msg}")
    def on_cancel_error(self, cancel_error):
        print(datetime.datetime.now(), sys._getframe().f_code.co_name)
    def on_order_stock_async_response(self, response):
        print(f"异步委托回调 {response.order_remark}")
    def on_cancel_order_stock_async_response(self, response):
        print(datetime.datetime.now(), sys._getframe().f_code.co_name)
    def on_account_status(self, status):
        print(datetime.datetime.now(), sys._getframe().f_code.co_name)


if __name__ == '__main__':
    print('[API]： Initializing new trading session')
    path = r'C:\Users\chuyin.wang\Desktop\share\gjzqqmt\国金证券QMT交易端\userdata_mini'
    session_id = int(time.time()) # per-strategy basis
    xt_trader = XtQuantTrader(path, session_id)
    xt_trader.set_relaxed_response_order_enabled(False) # exclusive query thread(async to exec callback)

    print('[API]： Registering STOCK account 8881848660')
    acc = StockAccount('8881848660', 'STOCK')

    print('[API]： Registering trading callback function')
    callback = MyXtQuantTraderCallback() # trader callback
    xt_trader.register_callback(callback)

    print('[API]： Starting Trader')
    xt_trader.start()
    connect_result = xt_trader.connect()
    if ~connect_result:
        print('[API]： Trader Connected')

    subscribe_result = xt_trader.subscribe(acc)
    if ~subscribe_result:
        print('[API]： Subscription Channel Connected(empty)')

    #这一行是注册全推回调函数 包括下单判断 安全起见处于注释状态 确认理解效果后再放开
    xtdata.subscribe_whole_quote(["SH", "SZ"], callback=subscribed_data_callback)
    print('[API]： Data Subscribed, Trading Start')

    # 阻塞主线程退出
    xt_trader.run_forever()

    # 如果使用vscode pycharm等本地编辑器 可以进入交互模式 方便调试 （把上一行的run_forever注释掉 否则不会执行到这里）
    # interact()
    
