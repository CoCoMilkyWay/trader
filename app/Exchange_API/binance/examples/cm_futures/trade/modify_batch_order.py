#!/usr/bin/env python
import logging
from binance.cm_futures import CMFutures as Client
from binance.lib.utils import config_logging
from binance.error import ClientError

config_logging(logging, logging.DEBUG)

from binance.utils.prepare_env import get_api_key
api_key, api_secret = get_api_key()

client = Client(key, secret, base_url="https://dapi.binance.com")

params = [
    {
        "symbol": "BNBUSD_PERP",
        "side": "BUY",
        "type": "LIMIT",
        "quantity": "1",
        "price": "400",
        "timeInForce": "GTC",
    }
]

try:
    response = client.modify_batch_order(params)
    logging.info(response)
except ClientError as error:
    logging.error(
        "Found error. status: {}, error code: {}, error message: {}".format(
            error.status_code, error.error_code, error.error_message
        )
    )
