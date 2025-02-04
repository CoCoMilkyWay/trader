from binance.lib.utils import check_required_parameter
from binance.lib.utils import check_required_parameters


def change_position_mode(self, dualSidePosition: str, **kwargs):
    """
    |
    | **Change Position Mode (TRADE)**
    | *Change user's position mode (Hedge Mode or One-way Mode) on EVERY symbol*

    :API endpoint: ``POST /dapi/v1/positionSide/dual``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Change-Position-Mode

    :parameter dualSidePosition: string
    :parameter recvWindow: optional int
    |
    """

    check_required_parameter(dualSidePosition, "dualSidePosition")
    params = {"dualSidePosition": dualSidePosition, **kwargs}
    url_path = "/dapi/v1/positionSide/dual"
    return self.sign_request("POST", url_path, params)


def get_position_mode(self, **kwargs):
    """
    |
    | **Get Current Position Mode (USER_DATA)**
    | *Get user's position mode (Hedge Mode or One-way Mode) on EVERY symbol*

    :API endpoint: ``GET /dapi/v1/positionSide/dual``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/account/Get-Current-Position-Mode

    :parameter recvWindow: optional int
    |
    """

    params = {**kwargs}
    url_path = "/dapi/v1/positionSide/dual"
    return self.sign_request("GET", url_path, params)


def new_order(self, symbol: str, side: str, type: str, **kwargs):
    """
    |
    | **New Order (TRADE)**
    | *Send a new order*

    :API endpoint: ``POST /dapi/v1/order``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/New-Order

    :parameter symbol: string
    :parameter side: string
    :parameter type: string
    :parameter positionSide: optional string. Default BOTH for One-way Mode; LONG or SHORT for Hedge Mode. It must be passed in Hedge Mode.
    :parameter timeInForce: optional string
    :parameter quantity: optional float
    :parameter reduceOnly: optional string
    :parameter price: optional float
    :parameter newClientOrderId: optional string. An unique ID among open orders. Automatically generated if not sent.
    :parameter stopPrice: optional float. Use with STOP/STOP_MARKET or TAKE_PROFIT/TAKE_PROFIT_MARKET orders.
    :parameter closePosition: optional string. true or false; Close-All, use with STOP_MARKET or TAKE_PROFIT_MARKET.
    :parameter activationPrice: optional float. Use with TRAILING_STOP_MARKET orders, default is the latest price (supporting different workingType).
    :parameter callbackRate: optional float. Use with TRAILING_STOP_MARKET orders, min 0.1, max 5 where 1 for 1%.
    :parameter workingType: optional string. stopPrice triggered by: "MARK_PRICE", "CONTRACT_PRICE". Default "CONTRACT_PRICE".
    :parameter priceProtect: optional string. "TRUE" or "FALSE", default "FALSE". Use with STOP/STOP_MARKET or TAKE_PROFIT/TAKE_PROFIT_MARKET orders.
    :parameter newOrderRespType: optional string. "ACK" or "RESULT", default "ACK".
    :parameter priceMatch: optional string. only avaliable for "LIMIT"/"STOP"/"TAKE_PROFIT" order; can be set to "OPPONENT"/"OPPONENT_5"/"OPPONENT_10"/"OPPONENT_20": /"QUEUE"/"QUEUE_5"/"QUEUE_10"/"QUEUE_20"; Can't be passed together with price.
    :parameter selfTradePreventionMode: optional string. "NONE":No STP /"EXPIRE_TAKER":expire taker order when STP triggers/"EXPIRE_MAKER":expire taker order when STP triggers/"EXPIRE_BOTH":expire both orders when STP triggers; default "NONE".
    :parameter recvWindow: optional int
    |
    """

    check_required_parameters([[symbol, "symbol"], [side, "side"], [type, "type"]])
    params = {"symbol": symbol, "side": side, "type": type, **kwargs}
    url_path = "/dapi/v1/order"
    return self.sign_request("POST", url_path, params)


def modify_order(
    self,
    symbol: str,
    side: str,
    orderId: int = None,
    origClientOrderId: str = None,
    **kwargs
):
    """
    |
    | **Modify Order (TRADE)**
    | *Order modify function, currently only LIMIT order modification is supported, modified orders will be reordered in the match queue.*

    :API endpoint: ``PUT /dapi/v1/order``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Modify-Order

    :parameter symbol: string
    :parameter side: string
    :parameter orderId: optional int
    :parameter origClientOrderId: optional string
    :parameter quantity: optional float
    :parameter price: optional float
    :parameter recvWindow: optional int

    **Notes**
        - Either orderId or origClientOrderId must be sent, and the orderId will prevail if both are sent.
        - Either quantity or price must be sent.
        - If the modification will cause the order to be cancelled immediately, the modification request will be rejected, in this case the user can force the modification by sending both quantity and price parameters and let the the order be cancelled immediately. So if you want to ensure the success of the modification request, we strongly recommend sending both quantity and price parameters at the same, for example:
        - When the new order quantity in the modification request is less than the partially filled quantity, if the user only sends quantity then the modification will fail, if the user sends both quantity and price then the modification will be successful and the order will be cancelled immediately.
        - When the new order price in the modification request prevents the GTX order from becoming a pending order (post only), if the user only sends price then the modification will fail, if the user sends both quantity and price then the modification will be successful and the order will be cancelled immediately.
    |
    """

    url_path = "/dapi/v1/order"
    params = {}

    if (orderId is None) and (origClientOrderId is None):
        check_required_parameters(
            [[symbol, "symbol"], [side, "side"], [orderId, "orderId"]]
        )
    elif orderId:
        params = {"symbol": symbol, "side": side, "orderId": orderId, **kwargs}
    else:
        params = {
            "symbol": symbol,
            "side": side,
            "origClientOrderId": origClientOrderId,
            **kwargs,
        }

    return self.sign_request("PUT", url_path, params)


def new_batch_order(self, batchOrders: list):
    """
    |
    | **Place Multiple Orders (TRADE)**
    | *Post a new batch order*

    :API endpoint: ``POST /dapi/v1/batchOrders``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Place-Multiple-Orders

    :parameter batchOrders: list
    :parameter recvWindow: optional int

    **Notes**
        - Paremeter rules are same with New Order
        - Batch orders are processed concurrently, and the order of matching is not guaranteed.
        - The order of returned contents for batch orders is the same as the order of the order list.
            batchOrders (list): order list. Max 5 orders
        - batchOrders is the list of order parameters in JSON
    |
    """

    params = {"batchOrders": batchOrders}
    url_path = "/dapi/v1/batchOrders"

    return self.sign_request("POST", url_path, params, True)


def modify_batch_order(self, batchOrders: list):
    """
    |
    | **Place Multiple Orders (TRADE)**
    | *Post a new batch order*

    :API endpoint: ``PUT /dapi/v1/batchOrders``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Modify-Multiple-Orders

    :parameter batchOrders: list
    :parameter recvWindow: optional int

    **Notes**
        - Paremeter rules are same with New Order
        - Batch orders are processed concurrently, and the order of matching is not guaranteed.
        - The order of returned contents for batch orders is the same as the order of the order list.
            batchOrders (list): order list. Max 5 orders
        - batchOrders is the list of order parameters in JSON
    |
    """

    params = {"batchOrders": batchOrders}
    url_path = "/dapi/v1/batchOrders"

    return self.sign_request("PUT", url_path, params)


def order_modify_history(
    self, symbol: str, orderId: int = None, origClientOrderId: str = None, **kwargs
):
    """
    |
    | **Get Order Modify History (USER_DATA)**
    | *Get order modification history*

    :API endpoint: ``GET /dapi/v1/orderAmendment``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Get-Order-Modify-History

    :parameter symbol: string
    :parameter orderId: optional int
    :parameter origClientOrderId: optional string
    :parameter startTime: optional int; Timestamp in ms to get modification history from INCLUSIVE
    :parameter endTime: optional int; Timestamp in ms to get modification history from INCLUSIVE
    :parameter limit: optional int
    :parameter recvWindow: optional int
    |
    """

    url_path = "/dapi/v1/orderAmendment"
    params = {}

    if (orderId is None) and (origClientOrderId is None):
        check_required_parameters(
            [
                [symbol, "symbol"],
                [orderId, "orderId"],
                ["origClientOrderId", origClientOrderId],
            ]
        )
    elif orderId:
        params = {"symbol": symbol, "orderId": orderId, **kwargs}
    else:
        params = {"symbol": symbol, "origClientOrderId": origClientOrderId, **kwargs}

    return self.sign_request("GET", url_path, params)


def query_order(
    self, symbol: str, orderId: int = None, origClientOrderId: str = None, **kwargs
):
    """
    |
    | **Query Order (USER_DATA)**
    | *Query a order*

    :API endpoint: ``GET /dapi/v1/order``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Query-Order

    :parameter symbol: string
    :parameter orderId: optional string
    :parameter origClientOrderId: optional string
    :parameter recvWindow: optional int
    |
    """

    url_path = "/dapi/v1/order"
    params = {}

    if (orderId is None) and (origClientOrderId is None):
        check_required_parameters(
            [
                [symbol, "symbol"],
                [orderId, "orderId"],
                ["origClientOrderId", origClientOrderId],
            ]
        )
    elif orderId:
        params = {"symbol": symbol, "orderId": orderId, **kwargs}
    else:
        params = {"symbol": symbol, "origClientOrderId": origClientOrderId, **kwargs}

    return self.sign_request("GET", url_path, params)


def cancel_order(
    self, symbol: str, orderId: int = None, origClientOrderId: str = None, **kwargs
):
    """
    |
    | **Cancel Order (TRADE)**
    | *Cancel an active order.*

    :API endpoint: ``DELETE /dapi/v1/order``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Cancel-Order

    :parameter symbol: string
    :parameter orderId: optional string
    :parameter origClientOrderId: optional string
    :parameter newClientOrderId: optional string
    :parameter recvWindow: optional int
    |
    """

    url_path = "/dapi/v1/order"
    params = {}

    if (orderId is None) and (origClientOrderId is None):
        check_required_parameters(
            [
                [symbol, "symbol"],
                [orderId, "orderId"],
                ["origClientOrderId", origClientOrderId],
            ]
        )
    elif orderId:
        params = {"symbol": symbol, "orderId": orderId, **kwargs}
    else:
        params = {"symbol": symbol, "origClientOrderId": origClientOrderId, **kwargs}

    return self.sign_request("DELETE", url_path, params)


def cancel_open_orders(self, symbol: str, **kwargs):
    """
    |
    | **Cancel All Open Orders (TRADE)**

    :API endpoint: ``DELETE /dapi/v1/allOpenOrders``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Cancel-All-Open-Orders

    :parameter symbol: string
    :parameter recvWindow: optional int, the value cannot be greater than 60000
    |
    """

    url_path = "/dapi/v1/allOpenOrders"
    params = {"symbol": symbol, **kwargs}

    return self.sign_request("DELETE", url_path, params)


def cancel_batch_order(
    self, symbol: str, orderIdList: list, origClientOrderIdList: list, **kwargs
):
    """
    |
    | **Cancel Multiple Orders (TRADE)**
    | *Cancel a new batch order*

    :API endpoint: ``DELETE /dapi/v1/batchOrders``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Cancel-Multiple-Orders

    :parameter symbol: string
    :parameter orderIdList: int list, max length 10 e.g. [1234567,2345678]
    :parameter origClientOrderIdList: string list, max length 10 e.g. ["my_id_1","my_id_2"], encode the double quotes. No space after comma.
    :parameter recvWindow: optional int

    **Notes**
        - Either orderIdList or origClientOrderIdList must be sent.
    |
    """

    url_path = "/dapi/v1/batchOrders"
    params = {}

    if (orderIdList is None) and (origClientOrderIdList is None):
        check_required_parameters(
            [
                [symbol, "symbol"],
                [orderIdList, "orderIdList"],
                [origClientOrderIdList, "origClientOrderIdList"],
            ]
        )
    elif orderIdList:
        params = {"symbol": symbol, "orderIdList": orderIdList, **kwargs}
    else:
        params = {
            "symbol": symbol,
            "origClientOrderIdList": origClientOrderIdList,
            **kwargs,
        }

    return self.sign_request("DELETE", url_path, params)


def countdown_cancel_order(self, symbol: str, countdownTime: int, **kwargs):
    """
    |
    | **Auto-Cancel All Open Orders (TRADE)**
    | *Cancel all open orders of the specified symbol at the end of the specified countdown.*

    :API endpoint: ``POST /dapi/v1/countdownCancelAll``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Auto-Cancel-All-Open-Orders

    :parameter symbol: string
    :parameter countdownTime: int list, countdown time, 1000 for 1 second. 0 to cancel the timer
    :parameter recvWindow: optional int

    **Notes**
        - The endpoint should be called repeatedly as heartbeats so that the existing countdown time can be canceled and replaced by a new one.
        - Example usage:
            - Call this endpoint at 30s intervals with an countdownTime of 120000 (120s).
            - If this endpoint is not called within 120 seconds, all your orders of the specified symbol will be automatically canceled.
            - If this endpoint is called with an countdownTime of 0, the countdown timer will be stopped.
        - The system will check all countdowns approximately every 10 milliseconds, so please note that sufficient redundancy should be considered when using this function.
        - We do not recommend setting the countdown time to be too precise or too small.
    |
    """

    check_required_parameters([[symbol, "symbol"], [countdownTime, "countdownTime"]])
    url_path = "/dapi/v1/countdownCancelAll"
    params = {"symbol": symbol, "countdownTime": countdownTime, **kwargs}

    return self.sign_request("POST", url_path, params)


def get_open_orders(
    self, symbol: str, orderId: int = None, origClientOrderId: str = None, **kwargs
):
    """
    |
    | **Query Current Open Order (USER_DATA)**
    | *Get all open orders on a symbol.*

    :API endpoint: ``GET /dapi/v1/openOrder``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Query-Current-Open-Order

    :parameter symbol: string
    :parameter orderId: optional string
    :parameter origClientOrderId: optional string
    :parameter recvWindow: optional int, the value cannot be greater than 60000

    **Notes**
        - Either orderId or origClientOrderId must be sent
        - If the queried order has been filled or cancelled, the error message "Order does not exist" will be returned.
    |
    """

    url_path = "/dapi/v1/openOrder"
    params = {}

    if (orderId is None) and (origClientOrderId is None):
        check_required_parameters(
            [
                [symbol, "symbol"],
                [orderId, "orderId"],
                [origClientOrderId, "origClientOrderId"],
            ]
        )
    elif orderId:
        params = {"symbol": symbol, "orderId": orderId, **kwargs}
    else:
        params = {"symbol": symbol, "origClientOrderId": origClientOrderId, **kwargs}

    return self.sign_request("GET", url_path, params)


def get_orders(self, **kwargs):
    """
    |
    | **Current All Open Orders (USER_DATA)**
    | *Get all open orders on a symbol. Careful when accessing this with no symbol.*
    | *If the symbol is not sent, orders for all symbols will be returned in an array.*

    :API endpoint: ``GET /dapi/v1/openOrders``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Current-All-Open-Orders

    :parameter symbol: string
    :parameter recvWindow: optional int, the value cannot be greater than 60000
    |
    """

    url_path = "/dapi/v1/openOrders"
    params = {**kwargs}

    return self.sign_request("GET", url_path, params)


def get_all_orders(self, **kwargs):
    """
    |
    | **All Orders (USER_DATA)**
    | *Get all account orders; active, canceled, or filled.*

    :API endpoint: ``GET /dapi/v1/allOrders``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/All-Orders

    :parameter symbol: string
    :parameter orderId: optional int
    :parameter startTime: optional int
    :parameter endTime: optional int
    :parameter limit: optional int; default 50, max 100.
    :parameter recvWindow: optional int; the value cannot be greater than 60000
    |
    """

    url_path = "/dapi/v1/allOrders"
    params = {**kwargs}

    return self.sign_request("GET", url_path, params)


def balance(self, **kwargs):
    """
    |
    | **Futures Account Balance (USER_DATA)**
    | *Get current account balance*

    :API endpoint: ``GET /dapi/v1/balance``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/account/Futures-Account-Balance

    :parameter recvWindow: optional int
    |
    """

    url_path = "/dapi/v1/balance"
    return self.sign_request("GET", url_path, {**kwargs})


def account(self, **kwargs):
    """
    |
    | **Account Information (USER_DATA)**
    | *Get current account information*

    :API endpoint: ``GET /dapi/v1/account``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/account/Account-Information

    :parameter recvWindow: optional int

    **Notes**
        - For One-way Mode user, the "positions" will only show the "BOTH" positions
        - For Hedge Mode user, the "positions" will show "BOTH", "LONG", and "SHORT" positions.
    |
    """

    url_path = "/dapi/v1/account"
    return self.sign_request("GET", url_path, {**kwargs})


def change_leverage(self, symbol: str, leverage: int, **kwargs):
    """
    |
    | **Change Initial Leverage (TRADE)**
    | *Change user's initial leverage in the specific symbol market.*
    | *For Hedge Mode, LONG and SHORT positions of one symbol use the same initial leverage and share a total notional value.*

    :API endpoint: ``POST /dapi/v1/leverage``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Change-Initial-Leverage

    :parameter symbol: string
    :parameter leverage: int; target initial leverage: int from 1 to 125
    :parameter recvWindow: optional int
    |
    """

    check_required_parameters([[symbol, "symbol"], [leverage, "leverage"]])
    url_path = "/dapi/v1/leverage"
    params = {"symbol": symbol, "leverage": leverage, **kwargs}
    return self.sign_request("POST", url_path, params)


def change_margin_type(self, symbol: str, marginType: str, **kwargs):
    """
    |
    | **Change Margin Type (TRADE)**
    | *Change user's margin type in the specific symbol market.For Hedge Mode, LONG and SHORT positions of one symbol use the same margin type.*
    | *With ISOLATED margin type, margins of the LONG and SHORT positions are isolated from each other.*

    :API endpoint: ``POST /dapi/v1/marginType``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Change-Margin-Type

    :parameter symbol: string
    :parameter leverage: string; ISOLATED, CROSSED
    :parameter recvWindow: optional int
    |
    """

    check_required_parameters([[symbol, "symbol"], [marginType, "marginType"]])
    url_path = "/dapi/v1/marginType"
    params = {"symbol": symbol, "marginType": marginType, **kwargs}
    return self.sign_request("POST", url_path, params)


def modify_isolated_position_margin(
    self, symbol: str, amount: float, type: int, **kwargs
):
    """
    |
    | **Modify Isolated Position Margin (TRADE)**

    :API endpoint: ``POST /dapi/v1/positionMargin``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Modify-Isolated-Position-Margin

    :parameter symbol: string
    :parameter amount: float
    :parameter type: int; 1: Add position margin，2: Reduce position margin
    :parameter positionSide: optional string; default BOTH for One-way Mode, LONG or SHORT for Hedge Mode. It must be sent with Hedge Mode.
    :parameter recvWindow: optional int
    |
    """

    check_required_parameters([[symbol, "symbol"], [amount, "amount"], [type, "type"]])
    url_path = "/dapi/v1/positionMargin"
    params = {"symbol": symbol, "amount": amount, "type": type, **kwargs}
    return self.sign_request("POST", url_path, params)


def get_position_margin_history(self, symbol: str, **kwargs):
    """
    |
    | **Get Position Margin Change History (TRADE)**
    | *Get position margin history on a symbol.*

    :API endpoint: ``GET /dapi/v1/positionMargin/history``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Get-Position-Margin-Change-History

    :parameter symbol: string
    :parameter type: optional int; 1: Add position margin，2: Reduce position margin
    :parameter startTime: optional string
    :parameter endTime: optional string
    :parameter limit: optional int; default 50
    :parameter recvWindow: optional int
    |
    """

    check_required_parameter(symbol, "symbol")
    url_path = "/dapi/v1/positionMargin/history"
    params = {"symbol": symbol, **kwargs}

    return self.sign_request("GET", url_path, params)


def get_position_risk(self, **kwargs):
    """
    |
    | **Position Information (USER_DATA)**
    | *Get current position information.*

    :API endpoint: ``GET /dapi/v1/positionRisk``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Position-Information

    :parameter symbol: string
    :parameter recvWindow: optional int
    |
    """

    url_path = "/dapi/v1/positionRisk"
    params = {**kwargs}

    return self.sign_request("GET", url_path, params)


def get_account_trades(self, **kwargs):
    """
    |
    | **Account Trade List (USER_DATA)**
    | *Get trades for a specific account and symbol.*

    :API endpoint: ``GET /dapi/v1/userTrades``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Account-Trade-List

    :parameter symbol: optional string
    :parameter pair: optional string
    :parameter startTime: optional string
    :parameter endTime: optional string
    :parameter fromId: optional int; trade ID to fetch from, default is to get the most recent trades.
    :parameter limit: optional int; default 50, max 100
    :parameter recvWindow: optional int

    **Notes**
        - Either symbol or pair must be sent
        - Symbol and pair cannot be sent together
        - Pair and fromId cannot be sent together
        - If a pair is sent,tickers for all symbols of the pair will be returned
        - The parameter fromId cannot be sent with startTime or endTime
    |
    """

    url_path = "/dapi/v1/userTrades"
    params = {**kwargs}

    return self.sign_request("GET", url_path, params)


def get_income_history(self, **kwargs):
    """
    |
    | **Get Income History (USER_DATA)**
    | *Get trades for a specific account and symbol.*

    :API endpoint: ``GET /dapi/v1/income``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/account/Get-Income-History

    :parameter symbol: optional string
    :parameter incomeType: optional string; "TRANSFER", "WELCOME_BONUS", "REALIZED_PNL", "FUNDING_FEE", "COMMISSION" and "INSURANCE_CLEAR"
    :parameter startTime: optional string; timestamp in ms to get funding from INCLUSIVE.
    :parameter endTime: optional string; timestamp in ms to get funding from INCLUSIVE.
    :parameter page: optional int
    :parameter limit: optional int; default 50, max 100
    :parameter recvWindow: optional int

    **Notes**
        - If incomeType is not sent, all kinds of flow will be returned
        - "trandId" is unique in the same incomeType for a user
    |
    """

    url_path = "/dapi/v1/income"
    params = {**kwargs}

    return self.sign_request("GET", url_path, params)


def get_download_id_transaction_history(self, startTime: int, endTime: int, **kwargs):
    """
    |
    | **Get Download Id For Futures Transaction History (USER_DATA)**
    | *Get download ID transaction history.*
    | *Request Limitation is 5 times per month, shared by front end download page and rest api*
    | *The time between startTime and endTime can not be longer than 1 year*

    :API endpoint: ``GET /dapi/v1/income/asyn``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/account/Get-Download-Id-For-Futures-Transaction-History

    :parameter startTime: int
    :parameter endTime: int
    :parameter recvWindow: optional int
    |
    """

    check_required_parameter(startTime, "startTime")
    check_required_parameter(endTime, "endTime")
    url_path = "/dapi/v1/income/asyn"
    params = {"startTime": startTime, "endTime": endTime, **kwargs}

    return self.sign_request("GET", url_path, params)


def leverage_brackets(self, symbol: str = None, pair: str = None, **kwargs):
    """
    |
    | **Notional and Leverage Brackets (USER_DATA)**
    | *Get notional and leverage bracket.*

    :API endpoint: ``GET /dapi/v1/leverageBracket``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/account/Notional-Bracket-for-Pair

    :API endpoint: ``GET /dapi/v2/leverageBracket``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/account/Notional-Bracket-for-Symbol

    :parameter symbol: optional string
    :parameter pair: optional string
    :parameter recvWindow: optional int
    |
    """

    url_path = ""
    params = {}

    if (symbol is None) and (pair is None):
        url_path = "/dapi/v2/leverageBracket"
        params = {**kwargs}
    elif symbol is None:
        url_path = "/dapi/v2/leverageBracket"
        params = {"symbol": symbol, **kwargs}
    else:
        url_path = "/dapi/v1/leverageBracket"
        params = {"pair": pair, **kwargs}

    return self.sign_request("GET", url_path, params)


def adl_quantile(self, **kwargs):
    """
    |
    | **Position ADL Quantile Estimation (USER_DATA)**
    | *Get Position ADL Quantile Estimation*

    :API endpoint: ``GET /dapi/v1/adlQuantile``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Position-ADL-Quantile-Estimation

    :parameter symbol: optional string
    :parameter recvWindow: optional int

    **Notes**
        - Values update every 30s.
        - Values 0, 1, 2, 3, 4 shows the queue position and possibility of ADL from low to high.
        - For positions of the symbol are in One-way Mode or isolated margined in Hedge Mode, "LONG", "SHORT", and "BOTH" will be returned to show the positions' adl quantiles of different position sides.
        - If the positions of the symbol are crossed margined in Hedge Mode:
        - "HEDGE" as a sign will be returned instead of "BOTH";
        - A same value caculated on unrealized pnls on long and short sides' positions will be shown for "LONG" and "SHORT" when there are positions in both of long and short sides.
    |
    """

    url_path = "/dapi/v1/adlQuantile"
    params = {**kwargs}

    return self.sign_request("GET", url_path, params)


def force_orders(self, **kwargs):
    """
    |
    | **User's Force Orders (USER_DATA)**
    | *Get User's Force Orders*

    :API endpoint: ``GET /dapi/v1/forceOrders``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/Users-Force-Orders

    :parameter symbol: optional string
    :parameter autoCloseType: optional string; "LIQUIDATION" for liquidation orders, "ADL" for ADL orders.
    :parameter startTime: optional string
    :parameter endTime: optional string
    :parameter limit: optional int; default 50, max 100
    :parameter recvWindow: optional int

    **Notes**
        - If "autoCloseType" is not sent, orders with both of the types will be returned
        - If "startTime" is not sent, data within 200 days before "endTime" can be queried
    |
    """

    url_path = "/dapi/v1/forceOrders"
    params = {**kwargs}

    return self.sign_request("GET", url_path, params)


def commission_rate(self, symbol: str, **kwargs):
    """
    |
    | **User Commission Rate (USER_DATA)**
    | *Get commission rate of symbol*

    :API endpoint: ``GET /dapi/v1/commissionRate``
    :API doc: https://developers.binance.com/docs/derivatives/coin-margined-futures/account/User-Commission-Rate

    :parameter symbol: optional string
    :parameter recvWindow: optional int
    |
    """

    check_required_parameter(symbol, "symbol")
    url_path = "/dapi/v1/commissionRate"
    params = {"symbol": symbol, **kwargs}

    return self.sign_request("GET", url_path, params)
