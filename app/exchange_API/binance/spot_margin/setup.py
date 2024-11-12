import os
from setuptools import setup, find_packages

NAME = "binance"

requirements = [
    "requests>=2.25.1",
    "websocket-client>=1.5.0",
    "pycryptodome>=3.15.0",
]

setup(
    name=NAME,
    install_requires=[req for req in requirements],
)
