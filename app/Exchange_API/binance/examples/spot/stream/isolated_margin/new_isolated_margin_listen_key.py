#!/usr/bin/env python

import logging
from binance.spot import Spot as Client
from binance.lib.utils import config_logging
from binance.utils.prepare_env import get_api_key

config_logging(logging, logging.DEBUG)

api_key, _ = get_api_key()

client = Client(api_key)
logging.info(client.new_isolated_margin_listen_key(symbol="BTCUSDT"))
