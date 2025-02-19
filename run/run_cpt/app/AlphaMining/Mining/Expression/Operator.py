import torch
from torch import Tensor
from abc import abstractmethod
from enum import IntEnum
from typing import List, Type, Union, Tuple, Optional, Callable

from Mining.Expression.Expression import Expression
from Mining.Expression.Dimension import DimensionType, Dimension
from Mining.Expression.Operand import OperandType, Operand, into_operand, _operand_input, _operand_output
from Mining.Expression.Dimension import DimensionType as T
from Mining.Data.Data import Data

DEBUG_PRINT = True  # True/False

# Operator base classes


class OperatorType(IntEnum):
    unary = 0
    binary = 1
    ternary = 2


_operator_output = Tensor


class Operator(Expression):
    @classmethod
    @abstractmethod
    def n_args(cls) -> int: ...

    @classmethod
    @abstractmethod
    def category_type(cls) -> OperatorType: ...
    
    @abstractmethod
    def validate_parameters(cls) -> bool: ...
    
    @property
    @abstractmethod
    def output(self) -> Operand: ...
    
    @property
    @abstractmethod
    def valid(self) -> bool: ...
    
    def _check_exprs_featured(self, args: list[Operand]) -> bool:
        any_is_featured: bool = False
        for i, arg in enumerate(args):
            if arg.OperandType != OperandType.matrix:
                return False
            if not isinstance(arg, Operand):
                # raise RuntimeError(f"{arg} is not a valid expression")
                return False
            if DimensionType.timedelta in arg.Dimension:
                # raise RuntimeError(f"{self.__name__} expects a normal expression for operand {i + 1}, "
                #                    f"but got {arg} (a DeltaTime)")
                return False
            any_is_featured = any_is_featured or arg.is_featured
        if not any_is_featured:
            return False
            # if len(args) == 1:
            #     raise RuntimeError(f"{self.__name__} expects a featured expression for its operand, "
            #                        f"but {args[0]} is not featured")
            # else:
            #     raise RuntimeError(f"{self.__name__} expects at least one featured expression for its operands, "
            #                        f"but none of {args} is featured")
        return True

    def _check_delta_time(self, arg) -> bool:
        if DimensionType.timedelta not in arg.Dimension:
            return False
            # raise RuntimeError(
            #     f"{self.__name__} expects a DeltaTime as its last operand, but {arg} is not")
        return True

    @property
    @abstractmethod
    def operands(self) -> Tuple[Expression, ...]: ...

    def __str__(self) -> str:
        # Prepare the operands string
        operands_str = ','.join(str(op) for op in self.operands)

        # Prepare the dimensions string
        dimensions_str = ','.join(
            f"{op.Dimension}" for op in self.operands)  # type: ignore

        # ANSI escape codes for colors
        Default = "\033[0m"
        Red = "\033[91m"
        Green = "\033[92m"
        Yellow = "\033[93m"
        Blue = "\033[94m"

        # Construct the final string representation
        if DEBUG_PRINT:
            return (
                f"{type(self).__name__}({operands_str})"
                f"{Yellow}"
                f"({dimensions_str})->{self.dimension}"  # type: ignore
                f"{Default}"
            )
        else:
            return f"{type(self).__name__}({operands_str})"


def get_subtensor(X: Tensor, i: slice, axis: int) -> Tensor:
    # Create a list of slices (slice(None) for all dimensions initially)
    slices = [slice(None)] * X.ndim  # Initialize with selecting all elements
    # Set the index for the chosen axis (i is not a slice anymore)
    slices[axis] = i
    # Return the subtensor at index i along the given axis
    return X[tuple(slices)]

# (dim0, dim1, dim2, ...)
# dim0  = last dimension = column direction(2d) = depth_slices direction(3d) = batches direction(4d)
#       = number of rows(2d) = number of depth_slices(3d) = number of batches(4d) = ...
# padding: [left, right, top, bottom, up, down, ...]
#          [1st dim ...                    last dim] (row major)


def RollingOp_1D(Op: Callable, X: Tensor, window: int, axis: int):
    '''Return the result of applying Op to a rolling window over a specified axis'''
    # Ensure axis is positive
    axis = axis % X.ndim
    # Repeat the first row (window - 1) times
    pad_top = X[:1, ...].repeat(window - 1, *[1] * (X.ndimension() - 1))
    # Concatenate the padded top with the original tensor
    padded_a = torch.cat([pad_top, X], dim=0)
    # Create a rolling window view along the specified axis
    # this expand the dimension by 1, but won't copy data
    # so don't worry about efficiency
    XView = padded_a.unfold(axis, window, step=1)
    # Apply the operation (e.g., torch.sum, torch.mean) over the rolling window dimension
    Xroll = Op(XView, dim=-1)
    return Xroll


def RollingOp_2D(Op: Callable, X: Tensor, Y: Tensor, window: int, axis: int):
    '''
    Apply an operation to rolling windows over a specified axis in X and Y.

    Parameters:
    - Op: A function that accepts two tensors (X_view, Y_view) and performs the desired computation.
    - X: The first input tensor.
    - Y: The second input tensor (same shape as X).
    - window: An integer specifying the window size.
    - axis: The axis over which to apply the rolling window.

    Returns:
    - The result of applying Op to the rolling windows of X and Y.
    '''
    assert (type(window) == int)
    # Ensure axis is positive
    axis = axis % X.ndim

    # Repeat the first row (window - 1) times
    pad_top_X = X[:1, ...].repeat(window - 1, *[1] * (X.ndimension() - 1))
    pad_top_Y = Y[:1, ...].repeat(window - 1, *[1] * (Y.ndimension() - 1))
    # Concatenate the padded top with the original tensor
    padded_X = torch.cat([pad_top_X, X], dim=0)
    padded_Y = torch.cat([pad_top_Y, Y], dim=0)

    # Unfold X and Y along the specified axis
    X_view = padded_X.unfold(axis, window, step=1)
    Y_view = padded_Y.unfold(axis, window, step=1)

    # Apply the operation Op to the unfolded views
    result = Op(X_view, Y_view, dim=-1)

    return result


class UnaryOperator(Operator):
    def __init__(self, _operand0: Operand) -> None:
        self._operand0 = _operand0
        self.TS = False
        self.CS = False
        self.init()
        self._valid: bool = self.validate_parameters()
        self._output: Operand = into_operand(self.evaluate, Dim=self.dimension)

    @abstractmethod
    def init(self) -> None: ...

    @classmethod
    def n_args(cls) -> int: return 1

    @classmethod
    def category_type(cls): return OperatorType.unary

    def validate_parameters(self) -> bool:
        check = True
        if self.TS:
            check = check and self._check_delta_time(self._operand0)
        else:
            check = check and self._check_exprs_featured([self._operand0])
        check_dim, dimension = self._check_dimension()
        self.dimension = Dimension([dimension])
        check = check and check_dim
        return check

    def evaluate(self, data: Data, period: slice = slice(0, 1)) -> _operator_output:
        return self._apply(self._operand0.evaluate(data))

    @abstractmethod
    def _check_dimension(self) -> Tuple[bool, DimensionType]: ...

    @abstractmethod
    def _apply(self, _operand0: _operand_output) -> _operator_output: ...

    @property
    def operands(self): return self._operand0,

    @property
    def is_featured(self): return self._operand0.is_featured

    @property
    def output(self) -> Operand:
        return self._output

    @property
    def valid(self) -> bool:
        return self._valid


class BinaryOperator(Operator):
    def __init__(self, operand0: Operand, operand1: Operand) -> None:
        self._operand0 = operand0
        self._operand1 = operand1
        self.TS = False
        self.CS = False
        self.init()
        self._valid: bool = self.validate_parameters()
        self._output: Operand = into_operand(self.evaluate, Dim=self.dimension)

    @abstractmethod
    def init(self) -> None: ...

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls):
        return OperatorType.binary

    def validate_parameters(self) -> bool:
        check = True
        check = check and self._check_exprs_featured([self._operand0])
        if self.TS:
            check = check and self._check_delta_time(self._operand1)
        else:
            check = check and self._check_exprs_featured([self._operand1])
        check_dim, dimension = self._check_dimension()
        self.dimension = Dimension([dimension])
        check = check and check_dim
        return check

    def evaluate(self, data: Data, period: slice = slice(0, 1)) -> _operator_output:
        return self._apply(
            self._operand0.evaluate(data),
            self._operand1.evaluate(data),
        )

    @abstractmethod
    def _check_dimension(self) -> Tuple[bool, DimensionType]: ...

    @abstractmethod
    def _apply(self,
               _operand0: _operand_output,
               _operand1: _operand_output) -> _operator_output: ...

    @property
    def operands(self): return self._operand0, self._operand1

    @property
    def is_featured(
        self): return self._operand0.is_featured or self._operand1.is_featured

    @property
    def output(self) -> Operand:
        return self._output

    @property
    def valid(self) -> bool:
        return self._valid

    # def evaluate(self, data: Data, period: slice = slice(0, 1)) -> Tensor:
    #     start = period.start - self._delta_time + 1
    #     stop = period.stop
    #     # L: period length (requested time window length)
    #     # W: window length (dt for rolling)
    #     # S: stock count
    #     values = self._operand.evaluate(
    #         data, slice(start, stop))   # (L+W-1, S)
    #     values = values.unfold(0, self._delta_time, 1)              # (L, S, W)
    #     return self._apply(values)                                  # (L, S)


class TernaryOperator(Operator):
    def __init__(self, operand0: Operand, operand1: Operand, operand2: Operand) -> None:
        self._operand0 = operand0
        self._operand1 = operand1
        self._operand2 = operand2
        self.TS = False
        self.CS = False
        self.init()
        self._valid: bool = self.validate_parameters()
        self._output: Operand = into_operand(self.evaluate, Dim=self.dimension)

    @abstractmethod
    def init(self) -> None: ...

    @classmethod
    def n_args(cls) -> int: return 3

    @classmethod
    def category_type(cls):
        return OperatorType.ternary

    def validate_parameters(self) -> bool:
        check = True
        check = check and self._check_exprs_featured([self._operand0])
        check = check and self._check_exprs_featured([self._operand1])
        if self.TS:
            check = check and self._check_delta_time(self._operand2)
        else:
            check = check and self._check_exprs_featured([self._operand2])
        check_dim, dimension = self._check_dimension()
        self.dimension = Dimension([dimension])
        check = check and check_dim
        return check

    def evaluate(self, data: Data, period: slice = slice(0, 1)) -> _operator_output:
        return self._apply(
            self._operand0.evaluate(data),
            self._operand1.evaluate(data),
            self._operand2.evaluate(data),
        )

    @abstractmethod
    def _check_dimension(self) -> Tuple[bool, DimensionType]: ...

    @abstractmethod
    def _apply(self,
               _operand0: _operand_output,
               _operand1: _operand_output,
               _operand2: _operand_output) -> _operator_output: ...

    @property
    def operands(self): return self._operand0, self._operand1, self._operand2

    @property
    def is_featured(self):
        return self._operand0.is_featured or self._operand1.is_featured or self._operand2.is_featured

    @property
    def output(self) -> Operand:
        return self._output

    @property
    def valid(self) -> bool:
        return self._valid


def check_dimension_policy(
    dimension_map: list[list[list[T]]],
    operand: Union[UnaryOperator, BinaryOperator, TernaryOperator],
) -> Tuple[bool, DimensionType]:
    """"
    'price', 'volume', 'ratio', 'misc', 'oscillator', 'timedelta', 'condition',
    map = [
        [[T.price], [T.price,T.misc], [T.timedelta], T.ratio],
        [[T.volume], [T.volume,T.misc], [T.timedelta], T.ratio],
        [[T.ratio], [T.ratio,T.misc], [T.timedelta], T.ratio],
        [[T.misc], [T.price,T.volume,T.ratio,T.misc,T.oscillator], [T.timedelta], T.ratio],
        [[T.oscillator], [T.oscillator,T.misc], [T.timedelta], T.ratio],
    ]
    return check_dimension_policy(map, self)
    """
    num_rules = len(dimension_map)
    num_operands = len(dimension_map[0]) - 1
    for rule_idx in range(num_rules):
        check = True
        for oprd_idx in range(num_operands):
            dimension: Dimension = getattr(
                operand, f'_operand{oprd_idx}').Dimension
            check = check and dimension.are_in(
                Dimension(dimension_map[rule_idx][oprd_idx]))
        if check:
            return True, dimension_map[rule_idx][-1]  # type: ignore
    return False, T.misc


# Operator implementations

class Abs(UnaryOperator):
    def init(self):
        self.TS = False
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], T.price],
            [[T.volume], T.volume],
            [[T.ratio], T.ratio],
            [[T.misc], T.misc],
            [[T.oscillator], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand: Tensor) -> Tensor: return operand.abs()


class Sign(UnaryOperator):
    def init(self):
        self.TS = False
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.ratio], T.condition],
            [[T.misc], T.condition],
            [[T.oscillator], T.condition],
        ]
        return check_dimension_policy(map, self)

    # -1. 0. 1.
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign()


class Log1p(UnaryOperator):
    def init(self):
        self.TS = False
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.volume], T.misc],
            [[T.ratio], T.ratio],
            [[T.misc], T.misc],
            [[T.oscillator], T.misc],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand: Tensor) -> Tensor: return operand.abs().log1p()


class CS_Rank(UnaryOperator):
    def init(self):
        self.TS = False
        self.CS = True

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], T.ratio],
            [[T.volume], T.ratio],
            [[T.ratio], T.ratio],
            [[T.misc], T.ratio],
            [[T.oscillator], T.ratio],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand: Tensor) -> Tensor:
        nan_mask = operand.isnan()
        n = (~nan_mask).sum(dim=1, keepdim=True)
        # argsort(): small -> large -> NaN(masked off again)
        rank = operand.argsort().argsort() / n
        rank[nan_mask] = torch.nan
        return rank


class Add(BinaryOperator):
    def init(self):
        self.TS = False
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.price], T.price],
            [[T.volume], [T.volume], T.volume],
            [[T.ratio], [T.ratio], T.ratio],
            [[T.misc], [T.misc], T.misc],
            [[T.oscillator], [T.oscillator], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs + rhs


class Sub(BinaryOperator):
    def init(self):
        self.TS = False
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.price], T.price],
            [[T.volume], [T.volume], T.volume],
            [[T.ratio], [T.ratio], T.ratio],
            [[T.misc], [T.misc], T.misc],
            [[T.oscillator], [T.oscillator], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs - rhs


class Mul(BinaryOperator):
    def init(self):
        self.TS = False
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.ratio], T.price],
            [[T.volume], [T.ratio], T.volume],
            [[T.ratio], [T.ratio], T.ratio],
            [[T.misc], [T.ratio], T.misc],
            [[T.oscillator], [T.ratio], T.oscillator],

            [[T.ratio], [T.price], T.price],
            [[T.ratio], [T.volume], T.volume],
            [[T.ratio], [T.ratio], T.ratio],
            [[T.ratio], [T.misc], T.misc],
            [[T.ratio], [T.oscillator], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs * rhs


class Div(BinaryOperator):
    def init(self):
        self.TS = False
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.price], T.ratio],
            [[T.volume], [T.volume], T.ratio],
            [[T.ratio], [T.ratio], T.ratio],
            [[T.misc], [T.misc], T.ratio],
            [[T.oscillator], [T.oscillator], T.ratio],

            [[T.price], [T.ratio], T.price],
            [[T.volume], [T.ratio], T.volume],
            [[T.ratio], [T.ratio], T.ratio],
            [[T.misc], [T.ratio], T.misc],
            [[T.oscillator], [T.ratio], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs / rhs


class Pow(BinaryOperator):
    def init(self):
        self.TS = False
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.ratio], T.price],
            [[T.volume], [T.ratio], T.volume],
            [[T.ratio], [T.ratio], T.ratio],
            [[T.misc], [T.ratio], T.misc],
            [[T.oscillator], [T.ratio], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs ** rhs


class Max(BinaryOperator):
    def init(self):
        self.TS = False
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.price], T.price],
            [[T.volume], [T.volume], T.volume],
            [[T.ratio], [T.ratio], T.ratio],
            [[T.misc], [T.misc], T.misc],
            [[T.oscillator], [T.oscillator], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.max(rhs)


class Min(BinaryOperator):
    def init(self):
        self.TS = False
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.price], T.price],
            [[T.volume], [T.volume], T.volume],
            [[T.ratio], [T.ratio], T.ratio],
            [[T.misc], [T.misc], T.misc],
            [[T.oscillator], [T.oscillator], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.min(rhs)


class TS_Ref(BinaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.timedelta], T.price],
            [[T.volume], [T.timedelta], T.volume],
            [[T.ratio], [T.timedelta], T.ratio],
            [[T.misc], [T.timedelta], T.misc],
            [[T.oscillator], [T.timedelta], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        def _DelT(X: Tensor, dim: int):
            result = get_subtensor(
                X=X, i=slice(-1, None), axis=dim).squeeze(dim=dim)
            assert result.dim() == 2
            return result
        return RollingOp_1D(_DelT, operand0, operand1, 0)


class TS_Delta(BinaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.timedelta], T.price],
            [[T.volume], [T.timedelta], T.volume],
            [[T.ratio], [T.timedelta], T.ratio],
            [[T.misc], [T.timedelta], T.misc],
            [[T.oscillator], [T.timedelta], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        def _DelT(X: Tensor, dim: int):
            result = \
                get_subtensor(X=X, i=slice(0, 1), axis=dim).squeeze(dim=dim) - \
                get_subtensor(X=X, i=slice(-1, None),
                              axis=dim).squeeze(dim=dim)
            assert result.dim() == 2
            return result
        return RollingOp_1D(_DelT, operand0, operand1, 0)


class TS_Mean(BinaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.timedelta], T.price],
            [[T.volume], [T.timedelta], T.volume],
            [[T.ratio], [T.timedelta], T.ratio],
            [[T.misc], [T.timedelta], T.misc],
            [[T.oscillator], [T.timedelta], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        return RollingOp_1D(torch.mean, operand0, operand1, 0)


class TS_Sum(BinaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.timedelta], T.price],
            [[T.volume], [T.timedelta], T.volume],
            [[T.ratio], [T.timedelta], T.ratio],
            [[T.misc], [T.timedelta], T.misc],
            [[T.oscillator], [T.timedelta], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        return RollingOp_1D(torch.sum, operand0, operand1, 0)


class TS_Std(BinaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.timedelta], T.price],
            [[T.volume], [T.timedelta], T.volume],
            [[T.ratio], [T.timedelta], T.ratio],
            [[T.misc], [T.timedelta], T.misc],
            [[T.oscillator], [T.timedelta], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        return RollingOp_1D(torch.std, operand0, operand1, 0)


class TS_Var(BinaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.timedelta], T.price],
            [[T.volume], [T.timedelta], T.volume],
            [[T.ratio], [T.timedelta], T.ratio],
            [[T.misc], [T.timedelta], T.misc],
            [[T.oscillator], [T.timedelta], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        return RollingOp_1D(torch.var, operand0, operand1, 0)


class TS_Skew(BinaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.timedelta], T.price],
            [[T.volume], [T.timedelta], T.volume],
            [[T.ratio], [T.timedelta], T.ratio],
            [[T.misc], [T.timedelta], T.misc],
            [[T.oscillator], [T.timedelta], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        def _Skew(X: Tensor, dim: int):
            # skew = m3 / m2^(3/2)
            central = X - X.mean(dim=dim, keepdim=True)
            m3 = (central ** 3).mean(dim=dim)
            m2 = (central ** 2).mean(dim=dim)
            return m3 / m2 ** 1.5
        return RollingOp_1D(_Skew, operand0, operand1, 0)


class TS_Kurt(BinaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.timedelta], T.price],
            [[T.volume], [T.timedelta], T.volume],
            [[T.ratio], [T.timedelta], T.ratio],
            [[T.misc], [T.timedelta], T.misc],
            [[T.oscillator], [T.timedelta], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        def _Kurt(X: Tensor, dim: int):
            # kurt = m4 / var^2 - 3
            central = X - X.mean(dim=dim, keepdim=True)
            m4 = (central ** 4).mean(dim=dim)
            var = X.var(dim=dim)
            return m4 / var ** 2 - 3
        return RollingOp_1D(_Kurt, operand0, operand1, 0)


class TS_Max(BinaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.timedelta], T.price],
            [[T.volume], [T.timedelta], T.volume],
            [[T.ratio], [T.timedelta], T.ratio],
            [[T.misc], [T.timedelta], T.misc],
            [[T.oscillator], [T.timedelta], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        return RollingOp_1D(torch.max, operand0, operand1, 0)[0]


class TS_Min(BinaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.timedelta], T.price],
            [[T.volume], [T.timedelta], T.volume],
            [[T.ratio], [T.timedelta], T.ratio],
            [[T.misc], [T.timedelta], T.misc],
            [[T.oscillator], [T.timedelta], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        return RollingOp_1D(torch.min, operand0, operand1, 0)[0]


class TS_Med(BinaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.timedelta], T.price],
            [[T.volume], [T.timedelta], T.volume],
            [[T.ratio], [T.timedelta], T.ratio],
            [[T.misc], [T.timedelta], T.misc],
            [[T.oscillator], [T.timedelta], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        return RollingOp_1D(torch.median, operand0, operand1, 0)[0]


class TS_Mad(BinaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.timedelta], T.price],
            [[T.volume], [T.timedelta], T.volume],
            [[T.ratio], [T.timedelta], T.ratio],
            [[T.misc], [T.timedelta], T.misc],
            [[T.oscillator], [T.timedelta], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        def _Mad(X: Tensor, dim: int):
            central = X - X.mean(dim=dim, keepdim=True)
            return central.abs().mean(dim=dim)
        return RollingOp_1D(_Mad, operand0, operand1, 0)


class TS_Rank(BinaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.timedelta], T.ratio],
            [[T.volume], [T.timedelta], T.ratio],
            [[T.ratio], [T.timedelta], T.ratio],
            [[T.misc], [T.timedelta], T.ratio],
            [[T.oscillator], [T.timedelta], T.ratio],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        def _Rank(X: Tensor, dim: int):
            n = X.shape[dim]
            # Extract the last value along the specified axis
            last = X.index_select(dim, torch.tensor([-1])).unsqueeze(dim)
            left = (last < X).count_nonzero(dim=dim)
            right = (last <= X).count_nonzero(dim=dim)
            return (right + left + (right > left)) / (2 * n)
        return RollingOp_1D(_Rank, operand0, operand1, 0)


class TS_WMA(BinaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.timedelta], T.price],
            [[T.volume], [T.timedelta], T.volume],
            [[T.ratio], [T.timedelta], T.ratio],
            [[T.misc], [T.timedelta], T.misc],
            [[T.oscillator], [T.timedelta], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        def _WMA(X: Tensor, dim: int):
            n = X.shape[dim]
            weights = torch.arange(n, dtype=X.dtype, device=X.device)
            weights /= weights.sum()
            return (weights * X).sum(dim=dim)
        return RollingOp_1D(_WMA, operand0, operand1, 0)


class TS_EMA(BinaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.timedelta], T.price],
            [[T.volume], [T.timedelta], T.volume],
            [[T.ratio], [T.timedelta], T.ratio],
            [[T.misc], [T.timedelta], T.misc],
            [[T.oscillator], [T.timedelta], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        def _EMA(X: Tensor, dim: int):
            n = X.shape[dim]
            alpha = 1 - 2 / (1 + n)
            power = torch.arange(n, 0, -1, dtype=X.dtype, device=X.device)
            weights = alpha ** power
            weights /= weights.sum()
            return (weights * X).sum(dim=dim)
        return RollingOp_1D(_EMA, operand0, operand1, 0)


class TS_Cov(TernaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.price], [T.timedelta], T.price],
            [[T.volume], [T.volume], [T.timedelta], T.volume],
            [[T.ratio], [T.ratio], [T.timedelta], T.ratio],
            [[T.misc], [T.misc], [T.timedelta], T.misc],
            [[T.oscillator], [T.oscillator], [T.timedelta], T.oscillator],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: Tensor, operand2: int) -> Tensor:
        def _Cov(X: Tensor, Y: Tensor, dim: int):
            n = X.shape[dim]
            clhs = X - X.mean(dim=dim, keepdim=True)
            crhs = Y - Y.mean(dim=dim, keepdim=True)
            return (clhs * crhs).sum(dim=dim) / (n - 1)
        return RollingOp_2D(_Cov, operand0, operand1, operand2, 0)


class TS_Corr(TernaryOperator):
    def init(self):
        self.TS = True
        self.CS = False

    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price], [T.price, T.misc], [T.timedelta], T.ratio],
            [[T.volume], [T.volume, T.misc], [T.timedelta], T.ratio],
            [[T.ratio], [T.ratio, T.misc], [T.timedelta], T.ratio],
            [[T.misc], [T.price, T.volume, T.ratio, T.misc,
                        T.oscillator], [T.timedelta], T.ratio],
            [[T.oscillator], [T.oscillator, T.misc], [T.timedelta], T.ratio],
        ]
        return check_dimension_policy(map, self)

    def _apply(self, operand0: Tensor, operand1: Tensor, operand2: int) -> Tensor:
        def _Corr(X: Tensor, Y: Tensor, dim: int):
            clhs = X - X.mean(dim=dim, keepdim=True)
            crhs = Y - Y.mean(dim=dim, keepdim=True)
            ncov = (clhs * crhs).sum(dim=dim)
            nlvar = (clhs ** 2).sum(dim=dim)
            nrvar = (crhs ** 2).sum(dim=dim)
            stdmul = (nlvar * nrvar).sqrt()
            stdmul[(nlvar < 1e-6) | (nrvar < 1e-6)] = 1
            return ncov / stdmul
        return RollingOp_2D(_Corr, operand0, operand1, operand2, 0)
