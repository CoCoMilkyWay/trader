import logging
import time

from binance.lib.utils import config_logging
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient

config_logging(logging, logging.DEBUG)


def message_handler(_, message):
    logging.info(message)

stream_url="wss://stream.binance.com:9443"
# stream_url="wss://stream.binance.com:443"

# proxies = { 'https': 'http://127.0.0.1:7890' }

my_client = SpotWebsocketStreamClient(
    stream_url=stream_url,
    on_message=message_handler, 
    is_combined=True,
    # proxies=proxies,
    )


my_client.subscribe(
    stream=["bnbusdt@bookTicker", "ethusdt@kline_1m"],
)

time.sleep(10)
my_client.stop()
