import os
# from Mining.Expression.Expression import Operators

DATAPATH = f"{os.path.dirname(__file__)}/Data/Example/TimeSeries"

MAX_EXPR_LENGTH = 15
MAX_EPISODE_LENGTH = 256

# CONSTANT OPERANDS
CONST_TIMEDELTAS = [1, 5, 10, 20, 30, 40, 50, 60, 120, 240]
CONST_RATIOS = [0.01, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 10]
CONST_OSCILLATORS = [10, 20, 30, 40, 50, 60, 70, 80, 90]

CONST_RATIOS.extend([-const for const in CONST_RATIOS])
CONST_OSCILLATORS.extend([-const for const in CONST_OSCILLATORS])

# Data Preparation
MAX_PAST = max(CONST_TIMEDELTAS)
MAX_FUTURE = 10

REWARD_PER_STEP = 0.
