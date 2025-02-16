import os
from Mining.Data.Data import Data
from Mining.Expression.Operator import *

# OPERANDS (CONSTANT) =========================================================
CONST_TIMEDELTAS = [1, 5, 10, 20, 30, 40, 50, 60, 120, 240]
CONST_RATIOS = [0.01, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 10]
CONST_OSCILLATORS = [10., 20., 30., 40., 50., 60., 70., 80., 90.]
CONST_RATIOS.extend([-const for const in CONST_RATIOS])
CONST_OSCILLATORS.extend([-const for const in CONST_OSCILLATORS])

# OPERATORS ===================================================================
OPERATORS: List[Type[Operator]] = [
    # Unary
    Abs, Sign, Log1p, CS_Rank,
    # Binary
    Add, Sub, Mul, Div, Pow, Max, Min,
    # Rolling
    TS_Ref, TS_Mean, TS_Sum, TS_Std, TS_Var, TS_Skew, TS_Kurt, TS_Max, TS_Min,
    TS_Med, TS_Mad, TS_Rank, TS_Delta, TS_WMA, TS_EMA,
    # Pair rolling
    TS_Cov, TS_Corr
]

# DATA and OPERANDS (FEATURES) ================================================
MAX_PAST = max(CONST_TIMEDELTAS)
MAX_FUTURE = 0
DATAPATH = f"{os.path.dirname(__file__)}/Data/Example/TimeSeries"
DATA = Data(DATAPATH, MAX_PAST, MAX_FUTURE, init=True)
FEATURES = DATA.features

# =============================================================================

MAX_EXPR_LENGTH = 15
MAX_EPISODE_LENGTH = 256

REWARD_PER_STEP = 0.
