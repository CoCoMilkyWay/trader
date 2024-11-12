#!/usr/bin/env python

import logging
from binance.spot import Spot as Client
from binance.lib.utils import config_logging
from examples.utils.prepare_env import get_api_key

config_logging(logging, logging.DEBUG)

api_key, api_secret = get_api_key()


client = Client(api_key, api_secret)
logging.info(client.loan_vip_repay(orderId=100000001, amount=100.5))
