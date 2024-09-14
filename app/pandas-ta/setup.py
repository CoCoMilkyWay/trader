# -*- coding: utf-8 -*-
from setuptools import setup

setup(
    name="pandas_ta",
    packages=[
        "pandas_ta",
        "pandas_ta.candles",
        "pandas_ta.cycles",
        "pandas_ta.momentum",
        "pandas_ta.overlap",
        "pandas_ta.performance",
        "pandas_ta.statistics",
        "pandas_ta.transform",
        "pandas_ta.trend",
        "pandas_ta.utils",
        "pandas_ta.volatility",
        "pandas_ta.volume"
    ],
    package_data={
        "pandas_ta": ["py.typed"]
    },
    install_requires=[
        "numba",
        "numpy",
        "pandas",
        "pandas-datareader",
        "scipy"
    ]
)
