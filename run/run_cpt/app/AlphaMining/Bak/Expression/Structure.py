from enum import Enum
from torch import Tensor
from typing import NamedTuple, List, Sequence, Tuple


class Token(object):
    def __init__(self, s=None):
        self.s = s
        self.name = s


class OperatorToken(object):
    def __init__(self, s=None):
        self.s = s


class UnaryOpToken(OperatorToken):
    def __init__(self, s=None):
        self.s = s


class BinaryOpToken(OperatorToken):
    def __init__(self, s=None):
        self.s = s


class TernaryOpToken(OperatorToken):
    def __init__(self, s=None):
        self.s = s


class ExpressionNode:
    def __init__(self, token, children=[], value=None):
        self.token: Token = token
        self.children: Sequence[ExpressionNode] = children
        self.value = value


class DimensionType(Enum):
    """
    Note that we do not consider dimension here, merely the property of data itself
    """
    # raw data type
    price = 1
    volume = 2
    oscillator = 3
    ratio = 4
    condition = 5
    misc = 6
    
    # intermediate data type
    timedelta = 10
    

class Dimension(Tuple):
    List[DimensionType]


class Value(NamedTuple):
    value: Tensor
    dimension: Dimension
