#!/usr/bin/env python

import logging
from binance.spot import Spot as Client
from binance.lib.utils import config_logging
from binance.utils.prepare_env import get_api_key

config_logging(logging, logging.DEBUG)

api_key, api_secret = get_api_key()

client = Client(api_key, api_secret)
logging.info(client.margin_my_trades(symbol="BNBUSDT"))
