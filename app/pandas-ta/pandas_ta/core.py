# -*- coding: utf-8 -*-
from dataclasses import dataclass
from multiprocessing import cpu_count, Pool
from pathlib import Path
from time import perf_counter
from warnings import simplefilter

from numpy import log10, ndarray
from pandas.api.extensions import register_dataframe_accessor
from pandas.errors import PerformanceWarning
from pandas import DataFrame, Series
from pandas import options as pd_options

from pandas_ta._typing import *
from pandas_ta import *

if Imports["dotenv"]:
    from dotenv import load_dotenv


# Pandas TA - DataFrame Extension Analysis Indicators
@register_dataframe_accessor("ta")
class AnalysisIndicators(object):
    """
    This Pandas Extension is named 'ta' for Technical Analysis. In other words,
    it is a Numerical Time Series Feature Generator where the Time Series data
    with emphasis on Financial Market data; typical data includes columns
    named :"open", "high", "low", "close", "volume".

    This TA Library hopefully allows you to apply familiar and unique Technical
    Analysis Indicators easily with this DataFrame Extension. Even though 'ta'
    is a Pandas DataFrame Extension, you can still call Technical Analysis
    indicators individually if you are more comfortable with that approach or
    it allows you to easily and automatically apply the indicators with the
    study method. See: help(ta.study).

    By default, the 'ta' extension uses lower case column names: open, high,
    low, close, and volume. You can override the defaults by providing the it's
    replacement name when calling the indicator. For example, to call the
    indicator hl2().

    With 'default' columns: open, high, low, close, and volume.
    >>> df.ta.hl2()
    >>> df.ta(kind="hl2")

    With DataFrame columns: Open, High, Low, Close, and Volume.
    >>> df.ta.hl2(high="High", low="Low")
    >>> df.ta(kind="hl2", high="High", low="Low")

    If you do not want to use a DataFrame Extension, just call it normally.
    >>> sma10 = ta.sma(df["Close"]) # Default length=10
    >>> sma50 = ta.sma(df["Close"], length=50)
    >>> ichimoku, span = ta.ichimoku(df["High"], df["Low"], df["Close"])

    Args:
        kind (str, optional): Default: None. Kind is the 'name' of the indicator.
            It converts kind to lowercase before calling.
        timed (bool, optional): Default: False. Curious about the execution
            speed?
        kwargs: Extension specific modifiers.
            append (bool, optional): Default: False. When True, it appends the
            resultant column(s) to the DataFrame.

    Returns:
        Most Indicators will return a Pandas Series. Others like MACD, BBANDS,
        KC, et al will return a Pandas DataFrame. Ichimoku on the other hand
        will return two DataFrames, the Ichimoku DataFrame for the known period
        and a Span DataFrame for the future of the Span values.

    Let's get started!

    1. Loading the 'ta' module:
    >>> import pandas as pd
    >>> import ta as ta

    2. Load some data:
    >>> df = pd.read_csv("AAPL.csv", index_col="date", parse_dates=True)

    3. Help!
    3a. General Help:
    >>> help(df.ta)
    >>> df.ta()
    3b. Indicator Help:
    >>> help(ta.apo)
    3c. Indicator Extension Help:
    >>> help(df.ta.apo)

    4. Ways of calling an indicator.
    4a. Standard: Calling just the APO indicator without "ta" DataFrame extension.
    >>> ta.apo(df["close"])  # or ta.apo(df.close)
    4b. DataFrame Extension: Calling just the APO indicator with "ta" DataFrame extension.
    >>> df.ta.apo()
    4c. DataFrame Extension (kind): Calling APO using 'kind'
    >>> df.ta(kind="apo")
    4d. Study:
    >>> df.ta.study("All") # Default
    >>> df.ta.study(ta.Study("My Strat", ta=[{"kind": "apo"}])) # Custom

    5. Working with kwargs
    5a. Append the result to the working df.
    >>> df.ta.apo(append=True)
    5b. Timing an indicator.
    >>> apo = df.ta(kind="apo", timed=True)
    >>> print(apo.timed)
    """

    # DataFrame Extension Properties
    _adjusted = None
    _config = None
    _cores = cpu_count()
    _custom = None
    _df = DataFrame()
    _ds = "yf" if Imports["yfinance"] else None
    _exchange = "NYSE"
    _last_run = get_time(_exchange, to_string=True)
    _time_range = "years"

    def __init__(self, obj: SeriesFrame):
        v_dataframe(obj)
        self._df = obj
        self._last_run = get_time(self._exchange, to_string=True)


    # DataFrame Behavioral Methods
    def __call__(
            self, kind: str = None, timed: bool = False,
            version: bool = False, **kwargs: DictLike
        ):
        if version:
            print(f"Pandas TA - Technical Analysis Indicators - v{self.version}")
        try:
            if isinstance(kind, str):
                kind = kind.lower()
                fn = getattr(self, kind)

                if timed:
                    stime = perf_counter()

                # Run the indicator
                result = fn(**kwargs)  # = getattr(self, kind)(**kwargs)

                if timed:
                    result.timed = final_time(stime)
                    print(f"[+] {kind}: {result.timed}")

                self._last_run = get_time(self.exchange, to_string=True)
                return result
            else:
                self.help()

        except BaseException:
            pass

    # Public Get/Set DataFrame Properties
    @property
    def adjusted(self) -> str:
        """property: df.ta.adjusted"""
        return self._adjusted

    @adjusted.setter
    def adjusted(self, value: str) -> None:
        """property: df.ta.adjusted = 'adj_close'"""
        if value is not None and isinstance(value, str):
            self._adjusted = value
        else:
            self._adjusted = None

    @property
    def cores(self) -> Int:
        """Returns the number of CPU cores."""
        return self._cores

    @cores.setter
    def cores(self, value: Int) -> None:
        """property: df.ta.cores = 0 <= int <= cpu_count"""
        cpus = cpu_count()
        if value is not None and isinstance(value, int):
            self._cores = int(value) if 0 <= value <= cpus else cpus
        else:
            self._cores = cpus

    @property
    def exchange(self) -> str:
        """Returns the current Exchange. Default: "NYSE"."""
        return self._exchange

    @exchange.setter
    def exchange(self, value: str) -> None:
        """property: df.ta.exchange = "LSE" """
        if value is not None and isinstance(value, str) and value in EXCHANGE_TZ.keys():
            self._exchange = value

    @property
    def last_run(self) -> str:
        """Returns the time when the DataFrame was last run."""
        return self._last_run

    # Public Get DataFrame Properties
    @property
    def config(self) -> str:
        """Returns the Pandas TA JSON config path."""
        return f"{self._config}"

    @config.setter
    def config(self, value: str) -> None:
        """property: df.ta.config = None (Default)"""
        _p = Path(value).expanduser()
        if _p.exists() and  _p.suffix == ".json":
            self._config = _p
        else:
            self._config = None

    @property
    def time_range(self) -> Float:
        """Returns the time ranges of the DataFrame as a float. Default is in "years". help(ta.total_time)"""
        return total_time(self._df, self._time_range)

    @time_range.setter
    def time_range(self, value: str) -> None:
        """property: df.ta.time_range = "years" (Default)"""
        if value is not None and isinstance(value, str):
            self._time_range = value
        else:
            self._time_range = "years"

    @property
    def version(self) -> str:
        """Returns the version."""
        return version

    # Private DataFrame Methods
    def _add_prefix_suffix(self,
        result: MaybeSeriesFrame = None, **kwargs: DictLike
    ) -> MaybeSeriesFrame:
        """Add prefix and/or suffix to the result columns"""
        if result is None:
            return
        else:
            prefix = suffix = ""
            delimiter = kwargs.setdefault("delimiter", "_")

            if "prefix" in kwargs:
                prefix = f"{kwargs['prefix']}{delimiter}"
            if "suffix" in kwargs:
                suffix = f"{delimiter}{kwargs['suffix']}"

            if isinstance(result, Series):
                result.name = prefix + result.name + suffix
            else:
                result.columns = [prefix + column + suffix for column in result.columns]

    def _append(self,
        result: MaybeSeriesFrame = None, **kwargs: DictLike
    ) -> MaybeSeriesFrame:
        """Appends a Pandas Series or DataFrame columns to self._df."""
        if "append" in kwargs and kwargs["append"]:
            df = self._df
            if df is None or result is None: return
            else:
                simplefilter(action="ignore", category=PerformanceWarning)
                pd_options.mode.chained_assignment = None

                if "col_names" in kwargs and not isinstance(kwargs["col_names"], tuple):
                    kwargs["col_names"] = (kwargs["col_names"],) # Note: tuple(kwargs["col_names"]) doesn't work

                if isinstance(result, DataFrame):
                    # If specified in kwargs, rename the columns.
                    # If not, use the default names.
                    if "col_names" in kwargs and isinstance(kwargs["col_names"], tuple):
                        if len(kwargs["col_names"]) >= len(result.columns):
                            for col, ind_name in zip(result.columns, kwargs["col_names"]):
                                df[ind_name] = result.loc[:, col]
                        else:
                            print(f"[!] Not enough col_names were specified : got {len(kwargs['col_names'])}, expected {len(result.columns)}.")
                            return
                    else:
                        # df = result.copy(deep=True) # Breaks Extension Indicators?
                        for i, column in enumerate(result.columns):
                            df[column] = result.iloc[:, i]
                else:
                    ind_name = (
                        kwargs["col_names"][0] if "col_names" in kwargs and
                        isinstance(kwargs["col_names"], tuple) else result.name
                    )
                    df[ind_name] = result
                pd_options.mode.chained_assignment = "warn"

    def _check_na_columns(self):
        """Returns the columns in which all it's values are na."""
        return [x for x in self._df.columns if all(self._df[x].isna())]

    def _get_column(self, series: Union[Series, str, None]):
        """Attempts to get the correct series or 'column' and return it."""
        df = self._df
        if df is None: return

        # Explicitly passing a pd.Series to override default.
        if isinstance(series, Series):
            return series
        # Apply default if no series nor a default.
        elif series is None:
            return df[self.adjusted] if self.adjusted is not None else None
        # Ok.  So it's a str.
        elif isinstance(series, str):
            # Return the df column since it's in there.
            if series in df.columns:
                return df[series]
            else:
                # Attempt to match the 'series' because it was likely
                # misspelled.
                matches = df.columns.str.match(series, case=False)
                match = [i for i, x in enumerate(matches) if x]
                # If found, awesome.  Return it or return the 'series'.
                NOT_FOUND = f"[X] The '{series}' column was not found in"
                cols = ", ".join(list(df.columns))

                if len(df.columns): NOT_FOUND += f": {cols}"
                else:               NOT_FOUND += " the DataFrame"

                if len(match):
                    return df.iloc[:, match[0]]
                else:
                    print(NOT_FOUND)

    def _indicators_by_category(self, name: str) -> List:
        """Returns indicators by Categorical name."""
        return Category[name] if name in self.categories() else None

    def _mp_worker(self, arguments: Tuple):
        """Multiprocessing Worker to handle different Methods."""
        method, args, kwargs = arguments

        if method != "ichimoku":
            return getattr(self, method)(*args, **kwargs)
        else:
            return getattr(self, method)(*args, **kwargs)[0]

    def _post_process(self,
        result: Union[Series, DataFrame], **kwargs: DictLike
    ) -> Union[Series, DataFrame]:
        """Applies any additional modifications to the DataFrame
        * Applies prefixes and/or suffixes
        * Appends the result to main DataFrame
        """
        verbose = kwargs.pop("verbose", False)
        if not isinstance(result, (Series, DataFrame)):
            if verbose:
                print(f"[X] The result is not a Series or DataFrame.")
            return self._df
        else:
            # Append only specific columns to the dataframe (via
            # 'col_numbers':(0,1,3) for example)
            result = (
                result.iloc[:, [int(n) for n in kwargs["col_numbers"]]]
                if isinstance(result, DataFrame) and
                "col_numbers" in kwargs and
                kwargs["col_numbers"] is not None else result
            )
            # Add prefix/suffix and append to the dataframe
            self._add_prefix_suffix(result=result, **kwargs)

            if "append" in kwargs and isinstance(kwargs["append"], bool):
                if not kwargs["append"]:
                    # Issue 388 - No appending, just print to stdout
                    # No DatetimeIndex could break execution.
                    print(result)
                else:
                    # Default: Appends result to DataFrame
                    self._append(result=result, **kwargs)
        return result

    def _study_mode(self, *args: Args) -> Tuple:
        """Helper method to determine the mode and name of the study.
        Returns tuple: (name:str, mode:dict)"""
        name = "All"
        mode = {"all": False, "category": False, "custom": False}

        if len(args) == 0:
            mode["all"] = True
        else:
            _categories = self.categories()
            if isinstance(args[0], str):
                if args[0].lower() == "all":
                    name, mode["all"] = name, True
                if args[0].lower() in _categories:
                    name, mode["category"] = args[0], True

            if isinstance(args[0], Study):
                study_ = args[0]
                if study_.ta is None or study_.name.lower() == "all":
                    name, mode["all"] = name, True
                elif study_.name.lower() in _categories:
                    name, mode["category"] = study_.name, True
                else:
                    name, mode["custom"] = study_.name, True

        return name, mode

    # Public DataFrame Methods
    def categories(self) -> ListStr:
        """Returns the categories."""
        return list(Category.keys())

    def constants(self, append: bool, values: List):
        """Constants

        Add or remove constants to the DataFrame easily with Numpy's arrays or
        lists. Useful when you need easily accessible horizontal lines for
        charting.

        Add constant '1' to the DataFrame
        >>> df.ta.constants(True, [1])
        Remove constant '1' to the DataFrame
        >>> df.ta.constants(False, [1])

        Adding constants for charting
        >>> import numpy as np
        >>> chart_lines = np.append(np.arange(-4, 5, 1), np.arange(-100, 110, 10))
        >>> df.ta.constants(True, chart_lines)
        Removing some constants from the DataFrame
        >>> df.ta.constants(False, np.array([-60, -40, 40, 60]))

        Args:
            append (bool): If True, appends a Numpy range of constants to the
                working DataFrame. If False, it removes the constant range from
                the working DataFrame. Default: None.

        Returns:
            Returns the appended constants
            Returns nothing to the user.  Either adds or removes constant ranges
            from the working DataFrame.
        """
        if isinstance(values, ndarray) or isinstance(values, list):
            if append:
                for x in values:
                    self._df[f"{x}"] = x
                return self._df[self._df.columns[-len(values):]]
            else:
                for x in values:
                    del self._df[f"{x}"]

    def datetime_ordered(self) -> bool:
        """Returns True if the index is a datetime and ordered."""
        if hasattr(self, "_df"):
            return v_datetime_ordered(self._df)
        return False

    def indicators(self,
        as_list: bool = None, exclude: ListStr = None
    ) -> List:
        """List of Indicators

        Args:
            as_list (bool): When True, it returns a list of the
                indicators. Default: False.
            exclude (List): The passed in list will be excluded
                from the indicators list. Default: None.

        Returns:
            Prints the list of indicators. If as_list=True, then a list.
        """
        as_list = bool(as_list) if isinstance(as_list, bool) else False
        user_excluded = []
        if isinstance(exclude, list) and len(exclude):
            user_excluded = exclude

        # Public DataFrame Extension methods
        df_ext_methods = [
            "categories",
            "constants",
            "datetime_ordered",
            "indicators",
            "reverse",
            "strategy",
            "study",
            "to_utc",
        ]
        # Public df.ta.properties
        ta_properties = [
            "adjusted",
            "config",
            "cores",
            # "custom",
            # "ds",
            "exchange",
            "last_run",
            "sample",
            "ticker",
            "time_range",
            "version"
        ]

        # Public non-indicator methods
        ta_indicators = list((x for x in dir(DataFrame().ta) if not x.startswith("_") and not x.endswith("_")))

        # Add Pandas TA methods and properties to be removed
        removed = df_ext_methods + ta_properties

        # Add user excluded methods to be removed
        if isinstance(user_excluded, list) and len(user_excluded) > 0:
            removed += user_excluded

        # Remove the unwanted indicators
        [ta_indicators.remove(x) for x in removed]

        # If as a list, immediately return
        if as_list:
            return ta_indicators

        indicator_count = len(ta_indicators)
        header = f"Pandas TA - Technical Analysis Indicators - v{self.version}"

        s, _count = f"{header}\n", 0
        if indicator_count > 0:
            from pandas_ta.candles.cdl_pattern import ALL_PATTERNS
            s += f"\nIndicators and Utilities [{indicator_count}]:\n    {', '.join(ta_indicators)}\n"
            _count += indicator_count
            if Imports["talib"]:
                s += f"\nCandle Patterns [{len(ALL_PATTERNS)}]:\n    {', '.join(ALL_PATTERNS)}\n"
                _count += len(ALL_PATTERNS)
        s += f"\nTotal Candles, Indicators and Utilities: {_count}"
        print(s)

    def reverse(self) -> DataFrame:
        """Reverses the DataFrame. Simply: df.iloc[::-1]"""
        return self._df.iloc[::-1]

    def sample(self, **kwargs: DictLike):
        """sample
        See help(ta.sample) for parameters.
        """
        return sample(**kwargs)

    def strategy(self, *args: Args, **kwargs: DictLike):
        """Strategy Method

        An experimental method that by default runs all applicable indicators.
        Future implementations will allow more specific indicator generation
        with possibly as json, yaml config file or an sqlite3 table.

        Kwargs:
            chunksize (bool): Adjust the chunksize for the Multiprocessing
                Pool. Default: Number of cores of the OS
            exclude (list): List of indicator names to exclude.
            name (str): Select all indicators or indicators by
                Category such as: "candles", "cycles", "momentum",
                "overlap", "performance", "statistics", "trend", "volatility",
                "volume", or "all". Default: "all"
            ordered (bool): Whether to run "all" in order. Default: True
            timed (bool): Show the process time of the study().
                Default: False
            verbose (bool): Provide some additional insight on the progress
                of the study() execution. Default: False
            warning (bool): Disables depreciation message. Automatically
                disabled when using it's replacement method: df.ta.study().
                Default: True
        """
        kwargs.update({"warning": True})
        return self.study(*args, **kwargs)

    def study(self, *args: Args, **kwargs: DictLike) -> dataclass:
        """Study Method

        An experimental method that by default runs all applicable indicators.

        Kwargs:
            chunksize (int): Adjust the chunksize for the Multiprocessing Pool.
                Default: Number of cores of the OS
            exclude (list): List of indicator names to exclude. Some are
                excluded by default for various reasons; they require additional
                sources (tsignals, xsignals), performance, , not a time series
                chart (vp) etc.
            name (str): Select all indicators or indicators by
                Category such as: "candles", "cycles", "momentum", "overlap",
                "performance", "statistics", "trend", "volatility", "volume", or
                "all". Default: "all"
            ordered (bool): Whether to run "all" in order. Default: True
            timed (bool): Show the process time of the study().
                Default: False
            verbose (bool): Provide some additional insight on the progress of
                the study() execution. Default: False
        """
        _dep_warning = kwargs.pop("warning", False)
        all_ordered = kwargs.pop("ordered", True)
        # Append indicators to the DataFrame by default
        kwargs.setdefault("append", True)
        # If True, it returns the resultant DataFrame. Default: False
        returns = kwargs.pop("returns", False)

        mp_chunksize = kwargs.pop("chunksize", self.cores)
        cores = kwargs.pop("cores", self.cores)
        self.cores = cores

        if _dep_warning:
            print(f"\n[!] DEPRECIATION WARNING:\n    Use study() instead of strategy().\n")

        # Initialize
        initial_column_count = self._df.shape[1]
        excluded = ["long_run", "short_run", "tsignals", "xsignals"]

        # Get the Study Name and mode
        name, mode = self._study_mode(*args)

        # If All or a Category, exclude user list if any
        user_excluded = kwargs.pop("exclude", [])
        if isinstance(user_excluded, str) and len(user_excluded) > 1:
            user_excluded = [user_excluded]
        if mode["all"] or mode["category"]:
            excluded += user_excluded

        # Collect the indicators, remove excluded or include kwarg["append"]
        if mode["category"]:
            ta = self._indicators_by_category(name.lower())
            [ta.remove(x) for x in excluded if x in ta]
        elif mode["custom"]:
            if hasattr(args[0], "cores") and isinstance(args[0].cores, int):
                self.cores = min(self.cores, args[0].cores)
            ta = args[0].ta
            for kwds in ta:
                kwds["append"] = True
        elif mode["all"]:
            ta = self.indicators(as_list=True, exclude=excluded)
        else:
            print(f"[X] Study not available.")
            return None

        verbose = kwargs.pop("verbose", False)
        if verbose:
            print(f"[+] Study: {name}\n[i] Indicator arguments: {kwargs}")
            if mode["all"] or mode["category"]:
                excluded_str = ", ".join(excluded)
                print(f"[i] Excluded[{len(excluded)}]: {excluded_str}")

        timed = kwargs.pop("timed", False)
        results = []
        use_multiprocessing = True if self.cores > 0 else False
        has_col_names = False

        if timed:
            stime = perf_counter()

        if use_multiprocessing and mode["custom"]:
            # Determine if the Custom Model has 'col_names' parameter
            has_col_names = (True if len([
                True for x in ta
                if "col_names" in x and isinstance(x["col_names"], tuple)
            ]) else False)

            if has_col_names:
                use_multiprocessing = False

        if Imports["tqdm"]:
            from tqdm import tqdm

        if use_multiprocessing:
            _total_ta = len(ta)
            with Pool(self.cores) as pool:
                # Some magic to optimize chunksize for speed
                # based on total ta indicators
                if mp_chunksize > _total_ta:
                    _chunksize = mp_chunksize - 1
                elif mp_chunksize > 0:
                    _chunksize = mp_chunksize
                else:
                    _chunksize = int(log10(_total_ta)) + 1
                if verbose:
                    print(f"[i] Multiprocessing {_total_ta} indicators with chunksize {_chunksize} and {self.cores}/{cpu_count()} cpus.")

                results = None
                if mode["custom"]:
                    # Create a list of all the custom indicators into a list
                    custom_ta = [(
                        ind["kind"],
                        ind["params"] if "params" in ind and isinstance(ind["params"], tuple) else (),
                        {**ind, **kwargs},
                    ) for ind in ta]
                    # Custom multiprocessing pool. Must be ordered for Chained Strategies
                    # May fix this to cpus if Chaining/Composition if it remains
                    if Imports["tqdm"] and verbose:
                        results = tqdm(pool.map(self._mp_worker, custom_ta, _chunksize), total=len(custom_ta) // _chunksize)
                    else:
                        results = pool.map(self._mp_worker, custom_ta, _chunksize)
                else:
                    default_ta = [(ind, tuple(), kwargs) for ind in ta]
                    tqdm_total = len(default_ta) // _chunksize
                    # All and Categorical multiprocessing pool.
                    if all_ordered:
                        if Imports["tqdm"] and verbose:
                            results = tqdm(pool.imap(self._mp_worker, default_ta, _chunksize), total=tqdm_total) # Order over Speed
                        else:
                            results = pool.imap(self._mp_worker, default_ta, _chunksize) # Order over Speed
                    else:
                        if Imports["tqdm"] and verbose:
                            results = tqdm(pool.imap_unordered(self._mp_worker, default_ta, _chunksize), total=tqdm_total) # Speed over Order
                        else:
                            results = pool.imap_unordered(self._mp_worker, default_ta, _chunksize) # Speed over Order
                if results is None:
                    print(f"[X] ta.study('{name}') has no results.")
                    return

                pool.close()
                pool.join()
                self._last_run = get_time(self.exchange, to_string=True)

        else:
            # Without multiprocessing:
            if verbose:
                _col_msg = f"[i] No multiprocessing (cores = 0)."
                if has_col_names:
                    _col_msg = f"[i] No multiprocessing support for 'col_names' option."
                print(_col_msg)

            if mode["custom"]:
                if Imports["tqdm"] and verbose:
                    pbar = tqdm(ta, f"[i] Progress")
                    for ind in pbar:
                        params = ind["params"] if "params" in ind and isinstance(ind["params"], tuple) else tuple()
                        getattr(self, ind["kind"])(*params, **{**ind, **kwargs})
                else:
                    for ind in ta:
                        params = ind["params"] if "params" in ind and isinstance(ind["params"], tuple) else tuple()
                        getattr(self, ind["kind"])(*params, **{**ind, **kwargs})
            else:
                if Imports["tqdm"] and verbose:
                    pbar = tqdm(ta, f"[i] Progress")
                    for ind in pbar:
                        getattr(self, ind)(*tuple(), **kwargs)
                else:
                    for ind in ta:
                        getattr(self, ind)(*tuple(), **kwargs)
                self._last_run = get_time(self.exchange, to_string=True)

        # Apply prefixes/suffixes and appends indicator results to the DataFrame
        [self._post_process(r, **kwargs) for r in results]

        final_column_count = self._df.shape[1]
        _added_columns = final_column_count - initial_column_count

        if verbose:
            print(f"[i] Total indicators: {len(ta)}")
            print(f"[i] Columns added: {_added_columns}")
            print(f"[i] Last Run: {self._last_run}")
        if timed:
            ft = final_time(stime)
            if _added_columns > 0:
                avgtd = (perf_counter() - stime) / _added_columns
            else:
                avgtd = perf_counter() - stime
            print(f"[i] Analysis Time: {ft} for {_added_columns} columns (avg {avgtd * 1000:2.4f} ms / col)")

        if returns:
            return self._df

    def ticker(self, ticker: str = None, period: str = None, **kwargs: DictLike):
        """ticker

        This method downloads Historical Data using the yfinance package if it
        is installed. Additionally it can run a ta.Study after download.

        Historical Data
        >>> df = df.ta.ticker("aapl")
        More specifically (for Yahoo Finance)
        >>> df = df.ta.ticker("aapl", period="max", interval="1d")

        Changing the period of Historical Data
        Period is used instead of start/end
        >>> df = df.ta.ticker("aapl", period="1y")

        Changing the period and interval of Historical Data
        Retrieves the past year in weeks
        >>> df = df.ta.ticker("aapl", period="1y", interval="1wk")
        Retrieves the past month in hours
        >>> df = df.ta.ticker("aapl", period="1mo", interval="1h")

        Args:
            ticker (str): Any string for a ticker you would use with yfinance.
                Default: "SPY"
            period (str): See the yfinance history() method for more options.
                Default: "max"

        Kwargs:
            study (str | ta.Study): Which study to apply after downloading.
                Default: None
            timed (bool): Print download time to stdout. Default: False

            For additional yfinance history() keyword arguments:
                https://github.com/ranaroussi/yfinance/blob/main/yfinance/base.py

        Returns:
            DataFrame or None
        """
        if not Imports["yfinance"]:
            print(f"[X] Please install yfinance to use this method. (pip install yfinance)")
            return

        # Pandas TA keywords to remove from **kwargs
        strategy = kwargs.pop("strategy", None)
        study = kwargs.pop("study", strategy)
        timed = kwargs.pop("timed", False)

        # yfinance keywords to filter from **kwargs
        ticker = v_str(ticker, "SPY")
        period = v_str(period, "max")
        interval = kwargs.pop("interval", "1d")
        proxy = kwargs.pop("proxy", {})

        df, stime = DataFrame(), None
        if ticker is not None:
            import yfinance as yf

            yfd = yf.Ticker(ticker)

            if timed: stime = perf_counter()
            df = yfd.history(
                period=period, interval=interval,
                proxy=proxy, **kwargs
            )
        else:
            return None

        if timed:
            df.timed = final_time(stime)
            print(f"[+] yf | {ticker}{df.shape}: {df.timed}")

        self._df = df

        if study is not None:
            self.study(study, **kwargs)

        return self._df

    def to_utc(self) -> None:
        """Sets the DataFrame index to UTC format"""
        self._df = to_utc(self._df)

    # Public DataFrame Methods: Indicators and Utilities
    # Candles
    def cdl_pattern(self, name: str = "all", offset=None, **kwargs: DictLike):
        open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = cdl_pattern(open_=open_, high=high, low=low, close=close, name=name, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def cdl_z(self, full=None, offset=None, **kwargs: DictLike):
        open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = cdl_z(open_=open_, high=high, low=low, close=close, full=full, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def ha(self, offset=None, **kwargs: DictLike):
        open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = ha(open_=open_, high=high, low=low, close=close, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    # Cycles
    def ebsw(self, close=None, length=None, bars=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = ebsw(close=close, length=length, bars=bars, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def reflex(self, close=None, length=None, smooth=None, alpha=None, pi=None, sqrt2=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = reflex(close=close, length=length, smooth=smooth, alpha=alpha, pi=pi, sqrt2=sqrt2, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    # Momentum
    def ao(self, fast=None, slow=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        result = ao(high=high, low=low, fast=fast, slow=slow, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def apo(self, fast=None, slow=None, mamode=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = apo(close=close, fast=fast, slow=slow, mamode=mamode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def bias(self, length=None, mamode=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = bias(close=close, length=length, mamode=mamode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def bop(self, percentage=False, offset=None, **kwargs: DictLike):
        open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = bop(open_=open_, high=high, low=low, close=close, percentage=percentage, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def brar(self, length=None, scalar=None, drift=None, offset=None, **kwargs: DictLike):
        open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = brar(open_=open_, high=high, low=low, close=close, length=length, scalar=scalar, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def cci(self, length=None, c=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = cci(high=high, low=low, close=close, length=length, c=c, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def cfo(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = cfo(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def cg(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = cg(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def cmo(self, length=None, scalar=None, drift=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = cmo(close=close, length=length, scalar=scalar, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def coppock(self, length=None, fast=None, slow=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = coppock(close=close, length=length, fast=fast, slow=slow, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def crsi(self, length_rsi=None, length_streak=None, length_rank=None,
    drift=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = crsi(close=close, length_rsi=length_rsi, length_streak=length_streak, length_rank=length_rank, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def cti(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = cti(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def dm(self, drift=None, offset=None, mamode=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        result = dm(high=high, low=low, drift=drift, mamode=mamode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def er(self, length=None, drift=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = er(close=close, length=length, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def eri(self, length=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = eri(high=high, low=low, close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def exhc(self, length=None, cap=None, asint=None, show_all=None, nozeros=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = exhc(close=close, length=length, cap=cap, asint=asint, show_all=show_all, nozeros=nozeros, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def fisher(self, length=None, signal=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        result = fisher(high=high, low=low, length=length, signal=signal, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def inertia(self, length=None, rvi_length=None, scalar=None, refined=None, thirds=None, mamode=None, drift=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        if refined is not None or thirds is not None:
            high = self._get_column(kwargs.pop("high", "high"))
            low = self._get_column(kwargs.pop("low", "low"))
            result = inertia(close=close, high=high, low=low, length=length, rvi_length=rvi_length, scalar=scalar,
                             refined=refined, thirds=thirds, mamode=mamode, drift=drift, offset=offset, **kwargs)
        else:
            result = inertia(close=close, length=length, rvi_length=rvi_length, scalar=scalar, refined=refined,
                             thirds=thirds, mamode=mamode, drift=drift, offset=offset, **kwargs)

        return self._post_process(result, **kwargs)

    def kdj(self, length=None, signal=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = kdj(high=high, low=low, close=close, length=length, signal=signal, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def kst(self, roc1=None, roc2=None, roc3=None, roc4=None, sma1=None, sma2=None, sma3=None, sma4=None, signal=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = kst(close=close, roc1=roc1, roc2=roc2, roc3=roc3, roc4=roc4, sma1=sma1, sma2=sma2, sma3=sma3,
                     sma4=sma4, signal=signal, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def macd(self, fast=None, slow=None, signal=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = macd(close=close, fast=fast, slow=slow, signal=signal, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def mom(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = mom(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def pgo(self, length=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = pgo(high=high, low=low, close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def ppo(self, fast=None, slow=None, scalar=None, mamode=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = ppo(close=close, fast=fast, slow=slow, scalar=scalar, mamode=mamode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def psl(self, open_=None, length=None, scalar=None, drift=None, offset=None, **kwargs: DictLike):
        if open_ is not None:
            open_ = self._get_column(kwargs.pop("open", "open"))

        close = self._get_column(kwargs.pop("close", "close"))
        result = psl(close=close, open_=open_, length=length, scalar=scalar, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def qqe(self, length=None, smooth=None, factor=None, mamode=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = qqe(close=close, length=length, smooth=smooth, factor=factor, mamode=mamode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def roc(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = roc(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def rsi(self, length=None, scalar=None, drift=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = rsi(close=close, length=length, scalar=scalar, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def rsx(self, length=None, drift=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = rsx(close=close, length=length, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def rvgi(self, length=None, swma_length=None, offset=None, **kwargs: DictLike):
        open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = rvgi(open_=open_, high=high, low=low, close=close, length=length, swma_length=swma_length,
                      offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def slope(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = slope(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def smc(self, abr_length=None, close_length=None, vol_length=None, percent=None, vol_ratio=None, asint=None, mamode=None, talib=None, offset=None, **kwargs):
        open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = smc(
            open_=open_, high=high, low=low, close=close,
            abr_length=abr_length, close_length=close_length, vol_length=vol_length, percent=percent,
            vol_ratio=vol_ratio, asint=asint, mamode=mamode, talib=talib, offset=offset, **kwargs
        )
        return self._post_process(result, **kwargs)

    def smi(self, fast=None, slow=None, signal=None, scalar=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = smi(close=close, fast=fast, slow=slow, signal=signal, scalar=scalar, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def squeeze(self, bb_length=None, bb_std=None, kc_length=None, kc_scalar=None, mom_length=None, mom_smooth=None, use_tr=None, mamode=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = squeeze(high=high, low=low, close=close, bb_length=bb_length, bb_std=bb_std, kc_length=kc_length,
                         kc_scalar=kc_scalar, mom_length=mom_length, mom_smooth=mom_smooth, use_tr=use_tr,
                         mamode=mamode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def squeeze_pro(self, bb_length=None, bb_std=None, kc_length=None, kc_scalar_wide=None, kc_scalar_normal=None, kc_scalar_narrow=None, mom_length=None, mom_smooth=None, use_tr=None, mamode=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = squeeze_pro(high=high, low=low, close=close, bb_length=bb_length, bb_std=bb_std, kc_length=kc_length,
                             kc_scalar_wide=kc_scalar_wide, kc_scalar_normal=kc_scalar_normal,
                             kc_scalar_narrow=kc_scalar_narrow, mom_length=mom_length, mom_smooth=mom_smooth,
                             use_tr=use_tr, mamode=mamode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def stc(self, tclength=None, ma1=None, ma2=None, osc=None, fast=None, slow=None, factor=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = stc(close=close, tclength=tclength, ma1=ma1, ma2=ma2, osc=osc, fast=fast, slow=slow, factor=factor,
                     offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def stoch(self, k=None, d=None, smooth_k=None, mamode=None, talib=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = stoch(high=high, low=low, close=close, k=k, d=d, smooth_k=smooth_k, mamode=mamode, talib=talib, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def stochf(self, k=None, d=None, mamode=None, talib=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = stochf(high=high, low=low, close=close, k=k, d=d, mamode=mamode, talib=talib, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def stochrsi(self, length=None, rsi_length=None, k=None, d=None, mamode=None, talib=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = stochrsi(high=high, low=low, close=close, length=length, rsi_length=rsi_length, k=k, d=d,
                          mamode=mamode, talib=talib, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def tmo(self, tmo_length=None, calc_length=None, smooth_length=None, mamode=None, compute_momentum=False, normalize_signal=False, offset=None, **kwargs: DictLike):
        open_ = self._get_column(kwargs.pop("open", "open"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = tmo(open_=open_, close=close, tmo_length=tmo_length, calc_length=calc_length, smooth_length=smooth_length, mamode=mamode, compute_momentum=compute_momentum, normalize_signal=normalize_signal, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def trix(self, length=None, signal=None, scalar=None, drift=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = trix(close=close, length=length, signal=signal, scalar=scalar, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def tsi(self, fast=None, slow=None, drift=None, mamode=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = tsi(close=close, fast=fast, slow=slow, drift=drift, mamode=mamode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def uo(self, fast=None, medium=None, slow=None, fast_w=None, medium_w=None, slow_w=None, drift=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = uo(high=high, low=low, close=close, fast=fast, medium=medium, slow=slow, fast_w=fast_w,
                    medium_w=medium_w, slow_w=slow_w, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def willr(self, length=None, percentage=True, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = willr(high=high, low=low, close=close, length=length, percentage=percentage, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    # Overlap
    def alligator(self, jaw=None, teeth=None, lips=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = alligator(close=close, jaw=jaw, teeth=teeth, lips=lips, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def alma(self, length=None, sigma=None, distribution_offset=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = alma(close=close, length=length, sigma=sigma, distribution_offset=distribution_offset, offset=offset,
                      **kwargs)
        return self._post_process(result, **kwargs)

    def dema(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = dema(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def ema(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = ema(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def fwma(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = fwma(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def hilo(self, high_length=None, low_length=None, mamode=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = hilo(high=high, low=low, close=close, high_length=high_length, low_length=low_length, mamode=mamode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def hl2(self, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        result = hl2(high=high, low=low, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def hlc3(self, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = hlc3(high=high, low=low, close=close, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def hma(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = hma(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def hwma(self, na=None, nb=None, nc=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = hwma(close=close, na=na, nb=nb, nc=nc, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def jma(self, length=None, phase=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = jma(close=close, length=length, phase=phase, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def kama(self, length=None, fast=None, slow=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = kama(close=close, length=length, fast=fast, slow=slow, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def ichimoku(self, tenkan=None, kijun=None, senkou=None, include_chikou=True, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result, span = ichimoku(high=high, low=low, close=close, tenkan=tenkan, kijun=kijun, senkou=senkou,
                                include_chikou=include_chikou, offset=offset, **kwargs)
        self._add_prefix_suffix(result, **kwargs)
        self._add_prefix_suffix(span, **kwargs)
        self._append(result, **kwargs)
        # return self._post_process(result, **kwargs), span
        return result, span

    def linreg(self, length=None, offset=None, adjust=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = linreg(close=close, length=length, offset=offset, adjust=adjust, **kwargs)
        return self._post_process(result, **kwargs)

    def mama(self, fastlimit=None, slowlimit=None, prenan=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = mama(close=close, fastlimit=fastlimit, slowlimit=slowlimit, prenan=prenan, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def mcgd(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = mcgd(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def midpoint(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = midpoint(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def midprice(self, length=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        result = midprice(high=high, low=low, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def ohlc4(self, offset=None, **kwargs: DictLike):
        open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = ohlc4(open_=open_, high=high, low=low, close=close, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def pivots(self, method=None, anchor=None, **kwargs: DictLike):
        open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = pivots(open_=open_, high=high, low=low, close=close, method=method, anchor=anchor, **kwargs)
        return self._post_process(result, **kwargs)

    def pwma(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = pwma(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def rma(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = rma(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def rwi(self, length=None, offset=None, **kwargs):
        close = self._get_column(kwargs.pop("close", "close"))
        result = rwi(high=high, low=low, close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def sinwma(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = sinwma(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def sma(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = sma(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def smma(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = smma(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def ssf(self, length=None, everget=None, pi=None, sqrt2=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = ssf(close=close, length=length, everget=everget, pi=pi, sqrt2=sqrt2, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def ssf3(self, length=None, pi=None, sqrt3=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = ssf3(close=close, length=length, pi=pi, sqrt3=sqrt3, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def supertrend(self, length=None, multiplier=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = supertrend(high=high, low=low, close=close, length=length, multiplier=multiplier, offset=offset,
                            **kwargs)
        return self._post_process(result, **kwargs)

    def swma(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = swma(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def t3(self, length=None, a=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = t3(close=close, length=length, a=a, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def tema(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = tema(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def trima(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = trima(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def vidya(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = vidya(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def wcp(self, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = wcp(high=high, low=low, close=close, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def wma(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = wma(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def zlma(self, length=None, mamode=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = zlma(close=close, length=length, mamode=mamode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    # Performance
    def log_return(self, length=None, cumulative=False, percent=False, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = log_return(close=close, length=length, cumulative=cumulative, percent=percent, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def percent_return(self, length=None, cumulative=False, percent=False, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = percent_return(close=close, length=length, cumulative=cumulative, percent=percent, offset=offset,
                                **kwargs)
        return self._post_process(result, **kwargs)

    # Statistics
    def entropy(self, length=None, base=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = entropy(close=close, length=length, base=base, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def kurtosis(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = kurtosis(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def mad(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = mad(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def median(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = median(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def quantile(self, length=None, q=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = quantile(close=close, length=length, q=q, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def skew(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = skew(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def stdev(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = stdev(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def tos_stdevall(self, length=None, stds=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = tos_stdevall(close=close, length=length, stds=stds, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def variance(self, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = variance(close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def zscore(self, length=None, std=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = zscore(close=close, length=length, std=std, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    # Transform
    def cube(self, cubing_exponent=None, signal_offset=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = cube(close=close, cubing_exponent=cubing_exponent, signal_offset=signal_offset, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def ifisher(self, amplifying_factor=None, signal_offset=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = ifisher(close=close, amplifying_factor=amplifying_factor, signal_offset=signal_offset, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def remap(self, from_min=None, from_max=None, to_min=None, to_max=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = remap(close=close, from_min=from_min, from_max=from_max, to_min=to_min, to_max=to_max, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    # Trend
    def adx(self, length=None, lensig=None, mamode=None, scalar=None, drift=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = adx(high=high, low=low, close=close, length=length, lensig=lensig, mamode=mamode, scalar=scalar,
                     drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def alphatrend(self, volume=None, src=None, length=None, multiplier=None, threshold=None, lag=None, mamode=None, talib=None, offset=None, **kwargs: DictLike):
        open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        if volume is not None:
            volume = self._get_column(kwargs.pop("volume", "volume"))
        result = alphatrend(
            open_=open_, high=high, low=low, close=close, volume=volume,
            src=src, length=length, multiplier=multiplier,
            threshold=threshold, lag=lag, mamode=mamode,
            talib=talib, offset=offset, **kwargs
        )
        return self._post_process(result, **kwargs)

    def amat(self, fast=None, slow=None, mamode=None, lookback=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = amat(close=close, fast=fast, slow=slow, mamode=mamode, lookback=lookback, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def aroon(self, length=None, scalar=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        result = aroon(high=high, low=low, length=length, scalar=scalar, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def chop(self, length=None, atr_length=None, ln=None, scalar=None, drift=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = chop(high=high, low=low, close=close, length=length, atr_length=atr_length, ln=ln, scalar=scalar,
                      drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def cksp(self, p=None, x=None, q=None, mamode=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = cksp(high=high, low=low, close=close, p=p, x=x, q=q, mamode=mamode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def decay(self, length=None, mode=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = decay(close=close, length=length, mode=mode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def decreasing(self, length=None, strict=None, asint=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = decreasing(close=close, length=length, strict=strict, asint=asint, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def dpo(self, length=None, centered=True, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = dpo(close=close, length=length, centered=centered, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def ht_trendline(self, talib=None, prenan=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = ht_trendline(close=close, talib=talib, prenan=prenan, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def increasing(self, length=None, strict=None, asint=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = increasing(close=close, length=length, strict=strict, asint=asint, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def long_run(self, fast=None, slow=None, length=None, offset=None, **kwargs: DictLike):
        if fast is None and slow is None:
            return self._df
        else:
            result = long_run(fast=fast, slow=slow, length=length, offset=offset, **kwargs)
            return self._post_process(result, **kwargs)

    def psar(self, af0=None, af=None, max_af=None, tv=False, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", None))
        result = psar(high=high, low=low, close=close, af0=af0, af=af, max_af=max_af, tv=tv, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def qstick(self, length=None, offset=None, **kwargs: DictLike):
        open_ = self._get_column(kwargs.pop("open", "open"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = qstick(open_=open_, close=close, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def rwi(self, length=None, lensig=None, mamode=None, scalar=None, drift=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = rwi(high=high, low=low, close=close, length=length, lensig=lensig, mamode=mamode, scalar=scalar,
                     drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def short_run(self, fast=None, slow=None, length=None, offset=None, **kwargs: DictLike):
        if fast is None and slow is None:
            return self._df
        else:
            result = short_run(fast=fast, slow=slow, length=length, offset=offset, **kwargs)
            return self._post_process(result, **kwargs)

    def supertrend(self, period=None, multiplier=None, mamode=None, drift=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = supertrend(high=high, low=low, close=close, period=period, multiplier=multiplier, mamode=mamode,
                            drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def trendflex(self, close=None, length=None, smooth=None, alpha=None, pi=None, sqrt2=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = trendflex(close=close, length=length, smooth=smooth, alpha=alpha, pi=pi, sqrt2=sqrt2, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def tsignals(self, trend=None, asbool=None, trend_reset=None, trend_offset=None, offset=None, **kwargs: DictLike):
        if trend is None:
            return self._df
        else:
            result = tsignals(trend, asbool=asbool, trend_offset=trend_offset, trend_reset=trend_reset, offset=offset, **kwargs)
            return self._post_process(result, **kwargs)

    def vhf(self, length=None, drift=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = vhf(close=close, length=length, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def vortex(self, drift=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = vortex(high=high, low=low, close=close, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def xsignals(self, signal=None, xa=None, xb=None, above=None, long=None, asbool=None, trend_reset=None, trend_offset=None, offset=None, **kwargs: DictLike):
        if signal is None:
            return self._df
        else:
            result = xsignals(signal=signal, xa=xa, xb=xb, above=above, long=long, asbool=asbool,
                              trend_offset=trend_offset, trend_reset=trend_reset, offset=offset, **kwargs)
            return self._post_process(result, **kwargs)

    def zigzag(self, close=None, legs=None, deviation=None, retrace=None, last_extreme=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        if close is not None:
            close = self._get_column(kwargs.pop("close", "close"))
        result = zigzag(
            high=high, low=low, close=close,
            legs=legs, deviation=deviation,
            retrace=retrace, last_extreme=last_extreme,
            offset=offset, **kwargs
        )
        return self._post_process(result, **kwargs)

    # Volatility
    def aberration(self, length=None, atr_length=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = aberration(high=high, low=low, close=close, length=length, atr_length=atr_length, offset=offset,
                            **kwargs)
        return self._post_process(result, **kwargs)

    def accbands(self, length=None, c=None, mamode=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = accbands(high=high, low=low, close=close, length=length, c=c, mamode=mamode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def atr(self, length=None, mamode=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = atr(high=high, low=low, close=close, length=length, mamode=mamode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def atrts(self, length=None, ma_length=None, multiplier=None, mamode=None, talib=None, drift=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = atrts(high=high, low=low, close=close, length=length, ma_length=ma_length, multiplier=multiplier, mamode=mamode, talib=talib, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def bbands(self, length=None, std=None, mamode=None, offset=None, **kwargs: DictLike):
        close  = self._get_column(kwargs.pop("close", "close"))
        result = bbands(close=close, length=length, std=std, mamode=mamode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def chandelier_exit(self, high_length=None, low_length=None, atr_length=None, multiplier=None, mamode=None, talib=None, use_close=None, drift=None, offset=None, **kwargs):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = chandelier_exit(
            high=high, low=low, close=close,
            high_length=high_length, low_length=low_length, atr_length=atr_length,
            multiplier=multiplier, mamode=mamode, talib=talib,
            use_close=use_close, drift=drift, offset=offset, **kwargs
        )
        return self._post_process(result, **kwargs)

    def donchian(self, lower_length=None, upper_length=None, offset: Int = None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        result = donchian(high=high, low=low, lower_length=lower_length, upper_length=upper_length, offset=offset,
                          **kwargs)
        return self._post_process(result, **kwargs)

    def hwc(self, na=None, nb=None, nc=None, nd=None, scalar=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = hwc(close=close, na=na, nb=nb, nc=nc, nd=nd, scalar=scalar, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def kc(self, length=None, scalar=None, mamode=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = kc(high=high, low=low, close=close, length=length, scalar=scalar, mamode=mamode, offset=offset,
                    **kwargs)
        return self._post_process(result, **kwargs)

    def massi(self, fast=None, slow=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        result = massi(high=high, low=low, fast=fast, slow=slow, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def natr(self, length=None, mamode=None, scalar=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = natr(high=high, low=low, close=close, length=length, mamode=mamode, scalar=scalar, offset=offset,
                      **kwargs)
        return self._post_process(result, **kwargs)

    def pdist(self, drift=None, offset=None, **kwargs: DictLike):
        open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = pdist(open_=open_, high=high, low=low, close=close, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def rvi(self, length=None, scalar=None, refined=None, thirds=None, mamode=None, drift=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = rvi(high=high, low=low, close=close, length=length, scalar=scalar, refined=refined, thirds=thirds,
                     mamode=mamode, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def thermo(self, long=None, short= None, length=None, mamode=None, drift=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        result = thermo(high=high, low=low, long=long, short=short, length=length, mamode=mamode, drift=drift,
                        offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def true_range(self, drift=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = true_range(high=high, low=low, close=close, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def ui(self, length=None, scalar=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        result = ui(close=close, length=length, scalar=scalar, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    # Volume
    def ad(self, open_=None, signed=True, offset=None, **kwargs: DictLike):
        if open_ is not None:
            open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = ad(high=high, low=low, close=close, volume=volume, open_=open_, signed=signed, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def adosc(self, open_=None, fast=None, slow=None, signed=True, offset=None, **kwargs: DictLike):
        if open_ is not None:
            open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = adosc(high=high, low=low, close=close, volume=volume, open_=open_, fast=fast, slow=slow,
                       signed=signed, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def aobv(self, fast=None, slow=None, mamode=None, max_lookback=None, min_lookback=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = aobv(close=close, volume=volume, fast=fast, slow=slow, mamode=mamode, max_lookback=max_lookback,
                      min_lookback=min_lookback, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def cmf(self, open_=None, length=None, offset=None, **kwargs: DictLike):
        if open_ is not None:
            open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = cmf(high=high, low=low, close=close, volume=volume, open_=open_, length=length, offset=offset,
                     **kwargs)
        return self._post_process(result, **kwargs)

    def efi(self, length=None, mamode=None, offset=None, drift=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = efi(close=close, volume=volume, length=length, offset=offset, mamode=mamode, drift=drift, **kwargs)
        return self._post_process(result, **kwargs)

    def eom(self, length=None, divisor=None, offset=None, drift=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = eom(high=high, low=low, close=close, volume=volume, length=length, divisor=divisor, offset=offset,
                     drift=drift, **kwargs)
        return self._post_process(result, **kwargs)

    def kvo(self, fast=None, slow=None, length_sig=None, mamode=None, offset=None, drift=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = kvo(high=high, low=low, close=close, volume=volume, fast=fast, slow=slow, length_sig=length_sig,
                     mamode=mamode, offset=offset, drift=drift, **kwargs)
        return self._post_process(result, **kwargs)

    def mfi(self, length=None, drift=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = mfi(high=high, low=low, close=close, volume=volume, length=length, drift=drift, offset=offset,
                     **kwargs)
        return self._post_process(result, **kwargs)

    def nvi(self, length=None, initial=None, signed=True, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = nvi(close=close, volume=volume, length=length, initial=initial, signed=signed, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def obv(self, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = obv(close=close, volume=volume, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def pvi(self, length=None, initial=None, mamode=None, overlay=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = pvi(close=close, volume=volume, length=length, initial=initial, mamode=mamode, overlay=overlay, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def pvo(self, fast=None, slow=None, signal=None, scalar=None, offset=None, **kwargs: DictLike):
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = pvo(volume=volume, fast=fast, slow=slow, signal=signal, scalar=scalar, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def pvol(self, volume=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = pvol(close=close, volume=volume, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def pvr(self, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = pvr(close=close, volume=volume)
        return self._post_process(result, **kwargs)

    def pvt(self, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = pvt(close=close, volume=volume, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def vhm(self, length=None, slength=None, offset=None, **kwargs: DictLike):
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = vhm(volume=volume, length=length, slength=slength, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def vwap(self, anchor=None, offset=None, **kwargs: DictLike):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))

        if not self.datetime_ordered():
            volume.index = self._df.index

        result = vwap(high=high, low=low, close=close, volume=volume, anchor=anchor, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def vwma(self, volume=None, length=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = vwma(close=close, volume=volume, length=length, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def wb_tsv(self, length=None, signal=None, offset=None, **kwargs: DictLike):
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = wb_tsv(close=close, volume=volume, signal=signal, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)
