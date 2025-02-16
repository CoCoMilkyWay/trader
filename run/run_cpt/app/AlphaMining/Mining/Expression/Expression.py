from abc import ABCMeta, abstractmethod

from torch import Tensor
from Mining.Data.Data import Data

# Expression: Operator -> Unary, Binary, ...
#             Operand  -> scalar, vector, matrix (Feature, Constant)
#             Data Dimensionon -> price, volume, ...


class Expression(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, data: Data,
                 period: slice = slice(0, 1)) -> Tensor: ...

    def __repr__(self) -> str: return str(self)

    @property
    @abstractmethod
    def is_featured(self) -> bool: ...


