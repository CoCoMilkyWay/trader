
from xtquant import xtdatacenter as xtdc 
xtdc.set_token("f0cfaab78b934cf57a4ed77add48d6b04d5e9c62")
xtdc.init(False)
port = 58601
xtdc.listen(port=port)
print(f"service on, port: {port}")

import time
from xtquant import xtdata

# token mode, no username/password
info = {"ip": "43.242.96.162", "port": 55300, "username": '', "pwd": ''}

connect_success = 0
def func(d):
    ip = d.get('ip', '')
    port = d.get('port')
    status = d.get('status', 'disconnected')

    global connect_success
    if ip == info['ip'] and port == info['port']:
        if status == 'connected':
            connect_success = 1
        else:
            connect_success = 2

# register callback function
xtdata.watch_quote_server_status(func)

# real-time data connection
qs = xtdata.QuoteServer(info)
qs.connect()

# waiting for connection
while connect_success == 0:
    time.sleep(0.3)

if connect_success == 2:
    print("connection failed")

