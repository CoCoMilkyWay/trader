from setuptools import setup, find_packages

setup(
    name='Chan',
    version='1.0',
    packages=find_packages(),
    description='chan.py',
    long_description=open('README.md',encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    package_data={
        # 如果你的pyd文件在包的子目录中
        # 'Chan': ['*.*', '**/*.*'],
        '': ["*"],
    },
    python_requires='>=3.11',
    install_requires=[
        "pandas",
        "numpy"
    ],
)