#!/usr/bin/env python
import logging
from binance.um_futures import UMFutures
from binance.lib.utils import config_logging
from binance.error import ClientError

config_logging(logging, logging.DEBUG)

from binance.utils.prepare_env import get_api_key
api_key, api_secret = get_api_key()

um_futures_client = UMFutures(key=api_key, secret=api_secret)

try:
    response = um_futures_client.async_download_trade_id(downloadId="1")
    logging.info(response)
except ClientError as error:
    logging.error(
        "Found error. status: {}, error code: {}, error message: {}".format(
            error.status_code, error.error_code, error.error_message
        )
    )
