#!/usr/bin/env python

import logging
from binance.spot import Spot as Client
from binance.lib.utils import config_logging
from binance.utils.prepare_env import get_api_key

config_logging(logging, logging.DEBUG)

api_key, api_secret = get_api_key()

client = Client(api_key, api_secret)

logger = logging.getLogger(__name__)

response = client.gift_card_create_code("BNB", 0.2)
logger.info(response)
