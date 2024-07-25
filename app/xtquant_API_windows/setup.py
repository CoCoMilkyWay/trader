from setuptools import setup, find_packages
import requests

setup(
    name='xtquant',
    version='240613.1.1',
    packages=find_packages(),
    description='xtquant_240613',
    long_description=open('README.md',encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    author='Willows',
    author_email='972210116@qq.com',
    url='http://dict.thinktrader.net/?id=7zqjlm',
    package_data={
        # 如果你的pyd文件在包的子目录中
        'xtquant': ['*.*', '**/*.*'],
    },
    python_requires='>=3.6, <3.13',
    license='GNU GPLv3 ',
    install_requires=[
        # 依赖列表
        "requests",
        "pandas",
        "numpy"
    ],
)