#!/usr/bin/env python

import time
import logging
from binance.lib.utils import config_logging
from binance.websocket.cm_futures.websocket_client import CMFuturesWebsocketClient

config_logging(logging, logging.DEBUG)


def message_handler(_, message):
    print(message)


my_client = CMFuturesWebsocketClient(on_message=message_handler)

my_client.index_kline(
    pair="btcusd",
    id=1,
    interval="1m",
    callback=message_handler,
)

time.sleep(10)

logging.debug("closing ws connection")
my_client.stop()
