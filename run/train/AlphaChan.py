from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc

_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]

class AlphaChan(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs
        )

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        return AlphaChanDL.get_feature_config(conf)

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

class AlphaChanDL(QlibDataLoader):
    """Dataloader to get Alpha158"""

    def __init__(self, config=None, **kwargs):
        _config = {
            "feature": self.get_feature_config(),
        }
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs) # type: ignore

    @staticmethod
    def get_feature_config(
        config={
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
    ):
        """create factors from config

        config = {
            'kbar': {}, # whether to use some hard-code kbar features
            'price': { # whether to use raw price features
                'windows': [0, 1, 2, 3, 4], # use price at n days ago
                'feature': ['OPEN', 'HIGH', 'LOW'] # which price field to use
            },
            'volume': { # whether to use raw volume features
                'windows': [0, 1, 2, 3, 4], # use volume at n days ago
            },
            'rolling': { # whether to use rolling operator based features
                'windows': [5, 10, 20, 30, 60], # rolling windows size
                'include': ['ROC', 'MA', 'STD'], # rolling operator to use
                #if include is None we will use default operators
                'exclude': ['RANK'], # rolling operator not to use
            }
        }
        """
        fields = []
        names = []
        if "kbar" in config:
            fields += [
                "($close-$open)/$open",
                "($high-$low)/$open",
                "($close-$open)/($high-$low+1e-12)",
                "($high-Greater($open, $close))/$open",
                "($high-Greater($open, $close))/($high-$low+1e-12)",
                "(Less($open, $close)-$low)/$open",
                "(Less($open, $close)-$low)/($high-$low+1e-12)",
                "(2*$close-$high-$low)/$open",
                "(2*$close-$high-$low)/($high-$low+1e-12)",
            ]
            names += [
                "KMID",
                "KLEN",
                "KMID2",
                "KUP",
                "KUP2",
                "KLOW",
                "KLOW2",
                "KSFT",
                "KSFT2",
            ]
        if "price" in config:
            windows = config["price"].get("windows", range(5))
            feature = config["price"].get("feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])
            for field in feature:
                field = field.lower()
                fields += ["Ref($%s, %d)/$close" % (field, d) if d != 0 else "$%s/$close" % field for d in windows]
                names += [field.upper() + str(d) for d in windows]
        if "volume" in config:
            windows = config["volume"].get("windows", range(5))
            fields += ["Ref($volume, %d)/($volume+1e-12)" % d if d != 0 else "$volume/($volume+1e-12)" for d in windows]
            names += ["VOLUME" + str(d) for d in windows]
        if "rolling" in config:
            windows = config["rolling"].get("windows", [5, 10, 20, 30, 60])
            include = config["rolling"].get("include", None)
            exclude = config["rolling"].get("exclude", [])
            # `exclude` in dataset config unnecessary filed
            # `include` in dataset config necessary field

            def use(x):
                return x not in exclude and (include is None or x in include)

            # Some factor ref: https://guorn.com/static/upload/file/3/134065454575605.pdf
            if use("ROC"):
                # https://www.investopedia.com/terms/r/rateofchange.asp
                # Rate of change, the price change in the past d days, divided by latest close price to remove unit
                fields += ["Ref($close, %d)/$close" % d for d in windows]
                names += ["ROC%d" % d for d in windows]
            if use("MA"):
                # https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp
                # Simple Moving Average, the simple moving average in the past d days, divided by latest close price to remove unit
                fields += ["Mean($close, %d)/$close" % d for d in windows]
                names += ["MA%d" % d for d in windows]
            if use("STD"):
                # The standard diviation of close price for the past d days, divided by latest close price to remove unit
                fields += ["Std($close, %d)/$close" % d for d in windows]
                names += ["STD%d" % d for d in windows]
            if use("BETA"):
                # The rate of close price change in the past d days, divided by latest close price to remove unit
                # For example, price increase 10 dollar per day in the past d days, then Slope will be 10.
                fields += ["Slope($close, %d)/$close" % d for d in windows]
                names += ["BETA%d" % d for d in windows]
            if use("RSQR"):
                # The R-sqaure value of linear regression for the past d days, represent the trend linear
                fields += ["Rsquare($close, %d)" % d for d in windows]
                names += ["RSQR%d" % d for d in windows]
            if use("RESI"):
                # The redisdual for linear regression for the past d days, represent the trend linearity for past d days.
                fields += ["Resi($close, %d)/$close" % d for d in windows]
                names += ["RESI%d" % d for d in windows]
            if use("MAX"):
                # The max price for past d days, divided by latest close price to remove unit
                fields += ["Max($high, %d)/$close" % d for d in windows]
                names += ["MAX%d" % d for d in windows]
            if use("LOW"):
                # The low price for past d days, divided by latest close price to remove unit
                fields += ["Min($low, %d)/$close" % d for d in windows]
                names += ["MIN%d" % d for d in windows]
            if use("QTLU"):
                # The 80% quantile of past d day's close price, divided by latest close price to remove unit
                # Used with MIN and MAX
                fields += ["Quantile($close, %d, 0.8)/$close" % d for d in windows]
                names += ["QTLU%d" % d for d in windows]
            if use("QTLD"):
                # The 20% quantile of past d day's close price, divided by latest close price to remove unit
                fields += ["Quantile($close, %d, 0.2)/$close" % d for d in windows]
                names += ["QTLD%d" % d for d in windows]
            if use("RANK"):
                # Get the percentile of current close price in past d day's close price.
                # Represent the current price level comparing to past N days, add additional information to moving average.
                fields += ["Rank($close, %d)" % d for d in windows]
                names += ["RANK%d" % d for d in windows]
            if use("RSV"):
                # Represent the price position between upper and lower resistent price for past d days.
                fields += ["($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)" % (d, d, d) for d in windows]
                names += ["RSV%d" % d for d in windows]
            if use("IMAX"):
                # The number of days between current date and previous highest price date.
                # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
                # The indicator measures the time between highs and the time between lows over a time period.
                # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
                fields += ["IdxMax($high, %d)/%d" % (d, d) for d in windows]
                names += ["IMAX%d" % d for d in windows]
            if use("IMIN"):
                # The number of days between current date and previous lowest price date.
                # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
                # The indicator measures the time between highs and the time between lows over a time period.
                # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
                fields += ["IdxMin($low, %d)/%d" % (d, d) for d in windows]
                names += ["IMIN%d" % d for d in windows]
            if use("IMXD"):
                # The time period between previous lowest-price date occur after highest price date.
                # Large value suggest downward momemtum.
                fields += ["(IdxMax($high, %d)-IdxMin($low, %d))/%d" % (d, d, d) for d in windows]
                names += ["IMXD%d" % d for d in windows]
            if use("CORR"):
                # The correlation between absolute close price and log scaled trading volume
                fields += ["Corr($close, Log($volume+1), %d)" % d for d in windows]
                names += ["CORR%d" % d for d in windows]
            if use("CORD"):
                # The correlation between price change ratio and volume change ratio
                fields += ["Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), %d)" % d for d in windows]
                names += ["CORD%d" % d for d in windows]
            if use("CNTP"):
                # The percentage of days in past d days that price go up.
                fields += ["Mean($close>Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTP%d" % d for d in windows]
            if use("CNTN"):
                # The percentage of days in past d days that price go down.
                fields += ["Mean($close<Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTN%d" % d for d in windows]
            if use("CNTD"):
                # The diff between past up day and past down day
                fields += ["Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)" % (d, d) for d in windows]
                names += ["CNTD%d" % d for d in windows]
            if use("SUMP"):
                # The total gain / the absolute total price changed
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMP%d" % d for d in windows]
            if use("SUMN"):
                # The total lose / the absolute total price changed
                # Can be derived from SUMP by SUMN = 1 - SUMP
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMN%d" % d for d in windows]
            if use("SUMD"):
                # The diff ratio between total gain and total lose
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "(Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d))"
                    "/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["SUMD%d" % d for d in windows]
            if use("VMA"):
                # Simple Volume Moving average: https://www.barchart.com/education/technical-indicators/volume_moving_average
                fields += ["Mean($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VMA%d" % d for d in windows]
            if use("VSTD"):
                # The standard deviation for volume in past d days.
                fields += ["Std($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VSTD%d" % d for d in windows]
            if use("WVMA"):
                # The volume weighted price change volatility
                fields += [
                    "Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["WVMA%d" % d for d in windows]
            if use("VSUMP"):
                # The total volume increase / the absolute total volume changed
                fields += [
                    "Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMP%d" % d for d in windows]
            if use("VSUMN"):
                # The total volume increase / the absolute total volume changed
                # Can be derived from VSUMP by VSUMN = 1 - VSUMP
                fields += [
                    "Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMN%d" % d for d in windows]
            if use("VSUMD"):
                # The diff ratio between total volume increase and total volume decrease
                # RSI indicator for volume
                fields += [
                    "(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))"
                    "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["VSUMD%d" % d for d in windows]

        return fields, names
