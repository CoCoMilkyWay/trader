#!/usr/bin/env python

import logging
import time
from binance.lib.utils import config_logging
from binance.websocket.spot.websocket_api import SpotWebsocketAPIClient

config_logging(logging, logging.DEBUG)


def on_close(_):
    logging.info("Do custom stuff when connection is closed")


def message_handler(_, message):
    logging.info(message)

proxies = { 'https': 'http://127.0.0.1:7890' }
my_client = SpotWebsocketAPIClient(on_message=message_handler, on_close=on_close, proxies=proxies)


my_client.order_book(symbol="BNBBUSD")

time.sleep(2)

my_client.order_book(symbol="BNBBUSD", limit=5)

time.sleep(2)

logging.info("closing ws connection")
my_client.stop()
