import torch
from torch import Tensor
from enum import IntEnum
from typing import Union, Optional
from Mining.Expression.Expression import Expression
from AlphaMining.Mining.Expression.Content import ContentType, Content
from Mining.Data.Data import Data

class OperandType(IntEnum):
    scalar = 0 # timedelta, ratio, oscillator
    vector = 1
    matrix = 2 # feature, intermediate

_operand_output = Union[Tensor, float, int, bool]

class Operand(Expression):
    def __init__(self, Value, OperandType:OperandType, Content:Content) -> None:
        self.Value = Value
        self.OperandType = OperandType
        self.Content = Content
    
    def evaluate(self, data: Data, period: slice = slice(0, 1)) -> _operand_output:
        assert period.step == 1 or period.step is None
        if self.OperandType == OperandType.scalar:
            if any([t in self.Content for t in[
                ContentType.price,
                ContentType.volume,
                ContentType.misc,
                ContentType.ratio,
                ContentType.oscillator,
            ]]):
                assert(type(self.Value) == float)
            if ContentType.timedelta in self.Content:
                assert(type(self.Value) == int)
            if ContentType.condition in self.Content:
                assert(type(self.Value) == bool)
            else:
                raise RuntimeError(f"Wrong constant operand/content type: {self.OperandType}/{self.Content}")
            return self.Value
        
        elif self.OperandType == OperandType.vector:
            assert(type(self.Value) == Tensor)
            assert(self.Value.dim()==1)
            return self.Value
        elif self.OperandType == OperandType.matrix:
            if type(self.Value) == str: # feature
                feature_idx = data.features.index(self.Value)
                return data.tensor[period.start:period.stop, feature_idx, :]
            else:
                assert(type(self.Value) == Tensor)
                assert(self.Value.dim()==2)
                return self.Value
        else:
            raise RuntimeError(f"Unknown operand type when evaluating")
        
    def __str__(self) -> str:
        if self.OperandType == OperandType.scalar:
            if ContentType.timedelta in self.Content:
                return f"{self.Value}d"
            else:
                return str(self.Value)
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
        if self.OperandType == OperandType.matrix:
            return True
        else:
            return False

_operand_input = Union[Operand, Tensor, str, float, int, bool]

def into_tensor(input: Union[float, int]) -> Tensor:
    return Tensor([]).new_empty(()).fill_(input)

def into_operand(value: _operand_input, Content:Content) -> Operand:
    """
    cast into operand or remain expression
    """
    if type(value) == Operand:
        return value
    elif type(value) == Tensor:
        return Operand(value, OperandType.matrix, Content)
    elif type(value) == int:
        return Operand(value, OperandType.scalar, Content)
    elif type(value) == float:
        return Operand(value, OperandType.scalar, Content)
    elif type(value) == str:
        return Operand(value, OperandType.scalar, Content)
    else:
        raise RuntimeError(f"Unknown operand value type: {value}{type(value)}")
