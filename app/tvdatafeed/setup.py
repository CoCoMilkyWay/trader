from setuptools import setup

setup(
    name="tvdatafeed",
    packages=["tvDatafeed"],
    install_requires=[
        "setuptools",
        "pandas",
        "websocket-client",
        "requests"
    ],
)
