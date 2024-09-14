# -*- coding: utf-8 -*-
from pandas import DataFrame, Series
from pandas_ta._typing import DictLike, Int
from pandas_ta.utils import (
    non_zero_range,
    rma_pandas,
    v_offset,
    v_pos_default,
    v_series
)



def kdj(
    high: Series, low: Series, close: Series,
    length: Int = None, signal: Int = None,
    offset: Int = None, **kwargs: DictLike
) -> Series:
    """KDJ (KDJ)

    The KDJ indicator is actually a derived form of the Slow
    Stochastic with the only difference being an extra line
    called the J line. The J line represents the divergence
    of the %D value from the %K. The value of J can go
    beyond [0, 100] for %K and %D lines on the chart.

    Sources:
        https://www.prorealcode.com/prorealtime-indicators/kdj/
        https://docs.anychart.com/Stock_Charts/Technical_Indicators/Mathematical_Description#kdj

    Args:
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's
        length (int): Default: 9
        signal (int): Default: 3
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.DataFrame: k, d, and j columns
    """
    # Validate
    length = v_pos_default(length, 9)
    signal = v_pos_default(signal, 3)
    _length = length + signal + 1
    high = v_series(high, _length)
    low = v_series(low, _length)
    close = v_series(close, _length)

    if high is None or low is None or close is None:
        return

    offset = v_offset(offset)

    # Calculate
    highest_high = high.rolling(length).max()
    lowest_low = low.rolling(length).min()

    fastk = 100 * (close - lowest_low) / \
        non_zero_range(highest_high, lowest_low)

    k = rma_pandas(fastk, length=signal)
    d = rma_pandas(k, length=signal)
    j = 3 * k - 2 * d

    # Offset
    if offset != 0:
        k = k.shift(offset)
        d = d.shift(offset)
        j = j.shift(offset)

    # Fill
    if "fillna" in kwargs:
        k.fillna(kwargs["fillna"], inplace=True)
        d.fillna(kwargs["fillna"], inplace=True)
        j.fillna(kwargs["fillna"], inplace=True)

    # Name and Category
    _props = f"_{length}_{signal}"
    k.name = f"K{_props}"
    d.name = f"D{_props}"
    j.name = f"J{_props}"
    k.category = d.category = j.category = "momentum"

    data = {k.name: k, d.name: d, j.name: j}
    df = DataFrame(data, index=close.index)
    df.name = f"KDJ{_props}"
    df.category = "momentum"

    return df
