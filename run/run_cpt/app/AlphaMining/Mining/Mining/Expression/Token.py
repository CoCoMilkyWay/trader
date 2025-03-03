from enum import IntEnum
from typing import List, Type
from Mining.Expression.Operator import Operator
from Mining.Expression.Operand import Operand
from Mining.Expression.Dimension import DimensionType, Dimension

class Token:
    def __repr__(self):
        return str(self)


class ConstantTDToken(Token):
    def __init__(self, value: int) -> None:
        self.value = value
        self.dim = Dimension(['timedelta'])

    def __str__(self): return str(self.value)


class ConstantRTToken(Token):
    def __init__(self, value: float) -> None:
        self.value = value
        self.dim = Dimension(['ratio'])

    def __str__(self): return str(self.value)


class ConstantOSToken(Token):
    def __init__(self, value: float) -> None:
        self.value = value
        self.dim = Dimension(['oscillator'])

    def __str__(self): return str(self.value)


class FeatureToken(Token):
    def __init__(self, feature: str, dim: Dimension) -> None:
        self.feature = feature
        self.dim = dim

    def __str__(self): return '$' + self.feature.lower()


class OperatorToken(Token):
    def __init__(self, operator: Type[Operator]) -> None:
        self.operator = operator

    def __str__(self): return self.operator.__name__


class OperandToken(Token):
    def __init__(self, operand: Type[Operand]) -> None:
        self.operand = operand

    def __str__(self): return str(self.operand)


class SyntaxType(IntEnum):
    BEG = 0
    SEP = 1


class SyntaxToken(Token):
    def __init__(self, syntax: SyntaxType) -> None:
        self.syntax = syntax

    def __str__(self): return self.syntax.name


BEG_TOKEN = SyntaxToken(SyntaxType.BEG)
SEP_TOKEN = SyntaxToken(SyntaxType.SEP)
