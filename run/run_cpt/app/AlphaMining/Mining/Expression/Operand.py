import torch
from torch import Tensor
from enum import IntEnum
from typing import List, Dict, Union, Callable, Optional
from Mining.Expression.Expression import Expression
from Mining.Expression.Dimension import DimensionType, Dimension
from Mining.Data.Data import Data


class OperandType(IntEnum):
    scalar = 0  # timedelta
    vector = 1
    matrix = 2  # feature, intermediate, ratio, oscillator


_operand_output = Union[Tensor, int]


class Operand(Expression):
    def __init__(self, Value, OperandType: OperandType, Dimension: Dimension) -> None:
        self.Value = Value
        self.OperandType = OperandType
        self.Dimension = Dimension
        # Cache for evaluated result
        self.result = None
        # self.IC: Dict[str, float] = {}
        # self.ric: Dict[str, float] = {}
        # self.icir: Dict[str, float] = {}
        # self.ricir: Dict[str, float] = {}

    def final_evaluate(self, data: Data) -> Tensor:
        """
        interface that is called only once, and is guaranteed by top level
        to be a valid operator output, thus must output a tensor
        """
        if self.result is None:
            self.result = self.evaluate(data)
            print(f"Evaluating(Final):{self}, "
                  f"Shape:{list(self.result.size())}")  # type: ignore
        assert type(self.result) == Tensor
        return self.result

    def evaluate(self, data: Data) -> _operand_output:
        start = data.max_past
        stop = data.max_past + data.n_timestamps + data.max_future - 1
        # assert(-data.max_past <= period.start and period.stop <= data.max_future + 1)
        if self.OperandType == OperandType.scalar:
            if type(self.Value) == int:
                return self.Value
            else:
                raise RuntimeError(
                    f"Wrong constant operand/dimension type: {self.OperandType}/{type(self.Value)}")
        elif self.OperandType == OperandType.vector:
            assert (type(self.Value) == Tensor)
            assert (self.Value.dim() == 1)
            return self.Value
        elif self.OperandType == OperandType.matrix:
            if callable(self.Value):
                result = self.Value(data)
                if isinstance(result, Tensor):
                    return result
                else:
                    raise RuntimeError(
                        f"Callable did not return a valid type: {type(result)}")
            if type(self.Value) == Tensor:
                assert (self.Value.dim() == 2)
                return self.Value
            elif type(self.Value) == float:
                return torch.full(size=(data.n_timestamps-1, data.n_codes),
                                  fill_value=self.Value, dtype=data.dtype, device=data.device)
            elif type(self.Value) == str:  # feature
                feature_idx = data.features.index(self.Value)
                return data.features_tensor[start:stop, feature_idx, :]
            else:
                raise RuntimeError(
                    f"Wrong matrix operand/dimension type: {self.OperandType}/{type(self.Value)}")
        else:
            raise RuntimeError(f"Unknown operand type when evaluating")

    def __str__(self) -> str:
        if self.OperandType == OperandType.scalar:
            if DimensionType.timedelta in self.Dimension:
                return f"{int(self.Value)}d"
            else:
                return str(f"{self.Value:.2f}")
        elif self.OperandType == OperandType.vector:
            raise RuntimeError(f"Unknown vector operand when parsing name")
        elif self.OperandType == OperandType.matrix:
            if type(self.Value) == str:
                return '$' + self.Value.lower()
            if type(self.Value) == float:
                return str(self.Value)
            if callable(self.Value):
                return str(self.Value.__self__)
            else:
                raise RuntimeError(
                    f"Try parsing intermediate operand name: {self.Value}")
        else:
            raise RuntimeError(f"Unknown operand type when parsing name")

    @property
    def is_featured(self) -> bool:
        if self.OperandType == OperandType.matrix:
            if type(self.Value) in [Tensor, str]:
                return True
            if callable(self.Value):
                return True
            return False
        else:
            return False


_operand_input = Union[Callable, Operand, Tensor, str, float, int]

# def into_tensor(input: Union[float, int]) -> Tensor:
#     return Tensor([]).new_empty(()).fill_(input)


def into_operand(value: _operand_input, Dim: Optional[Dimension]) -> Operand:
    """
    cast into operand or remain expression
    """
    if type(value) == Operand:
        if isinstance(value, Operand):
            return value
        else:
            raise RuntimeError(f"not an instanced operand: {value}")
    elif Dim:
        assert type(
            Dim) == Dimension, f"Non-Dimension received: {Dim}/{type(Dim)}"
        # this is for building expression stack without evaluating first
        # e.g.  output of the lower level operators
        #       as new operand(link only) to upper level operators
        if callable(value):
            return Operand(value, OperandType.matrix, Dim)
        if type(value) == Tensor:
            return Operand(value, OperandType.matrix, Dim)
        elif type(value) == str:
            return Operand(value, OperandType.matrix, Dim)
        elif type(value) == float:
            return Operand(value, OperandType.matrix, Dim)
        elif type(value) == int:
            return Operand(value, OperandType.scalar, Dim)
        else:
            raise RuntimeError(
                f"Unknown operand value type: {value}{type(value)}")
    else:
        raise RuntimeError(
            f"Non-Operand input require dimension: {value}{type(value)}")
