from setuptools import setup, find_packages

setup(
    name='Mining',
    packages=find_packages(),  # Automatically find sub-packages and modules
    package_data={
        'Mining': [
            '*.py', # Include all .py files in the Mining package
            ],
    },
    install_requires=[
        # List your dependencies here, for example:
        # 'numpy',
        # 'pandas',
    ],
    entry_points={
        'console_scripts': [
            # Optional: Define any command line scripts here
            # 'command-name=package_name.module:function_name',
        ],
    },
    # Optional: metadata to display on PyPI
    description='need to be installed because those files need to be visible to remote machines in distributed systems',
)