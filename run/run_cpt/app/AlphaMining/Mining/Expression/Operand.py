import torch
from torch import Tensor
from enum import IntEnum
from typing import Union, Optional
from Mining.Expression.Expression import Expression
from Mining.Expression.Dimension import DimensionType, Dimension
from Mining.Data.Data import Data

class OperandType(IntEnum):
    scalar = 0 # timedelta
    vector = 1
    matrix = 2 # feature, intermediate, ratio, oscillator

_operand_output = Union[Tensor, int]

class Operand(Expression):
    def __init__(self, Value, OperandType:OperandType, Dimension:Dimension) -> None:
        self.Value = Value
        self.OperandType = OperandType
        self.Dimension = Dimension
    
    def evaluate(self, data: Data) -> _operand_output:
        start = data.max_past
        stop = data.max_past + data.n_timestamps + data.max_future  - 1
        #assert(-data.max_past <= period.start and period.stop <= data.max_future + 1)
        if self.OperandType == OperandType.scalar:
            if type(self.Value) == int:
                return self.Value
            else:
                raise RuntimeError(f"Wrong constant operand/dimension type: {self.OperandType}/{type(self.Value)}")
        elif self.OperandType == OperandType.vector:
            assert(type(self.Value) == Tensor)
            assert(self.Value.dim()==1)
            return self.Value
        elif self.OperandType == OperandType.matrix:
            if type(self.Value) == Tensor:
                assert(self.Value.dim()==2)
                return self.Value
            elif type(self.Value) == float:
                return torch.full(size=(data.n_timestamps, data.n_codes),
                       fill_value=self.Value, dtype=data.dtype, device=data.device)
            elif type(self.Value) == str: # feature
                feature_idx = data.features.index(self.Value)
                return data.tensor[start:stop, feature_idx, :]
            else:
                raise RuntimeError(f"Wrong matrix operand/dimension type: {self.OperandType}/{type(self.Value)}")
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
            else:
                raise RuntimeError(f"Try parsing intermediate operand name")
        else:
            raise RuntimeError(f"Unknown operand type when parsing name")
        
    @property
    def is_featured(self) -> bool:
        if self.OperandType == OperandType.matrix \
            and type(self.Value) in [Tensor, str]:
            return True
        else:
            return False

_operand_input = Union[Operand, Tensor, str, float, int]

# def into_tensor(input: Union[float, int]) -> Tensor:
#     return Tensor([]).new_empty(()).fill_(input)

def into_operand(value:_operand_input, Dimension:Dimension) -> Operand:
    """
    cast into operand or remain expression
    """
    if type(value) == Operand:
        return value
    elif type(value) == Tensor:
        return Operand(value, OperandType.matrix, Dimension)
    elif type(value) == str:
        return Operand(value, OperandType.matrix, Dimension)
    elif type(value) == float:
        return Operand(value, OperandType.matrix, Dimension)
    elif type(value) == int:
        return Operand(value, OperandType.scalar, Dimension)
    else:
        raise RuntimeError(f"Unknown operand value type: {value}{type(value)}")
