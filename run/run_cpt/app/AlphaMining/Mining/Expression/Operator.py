import torch
from torch import Tensor
from abc import abstractmethod
from enum import IntEnum
from typing import List, Type, Union, Tuple

from Mining.Expression.Expression import Expression
from Mining.Expression.Dimension import DimensionType, Dimension
from Mining.Expression.Operand import OperandType, Operand, into_operand, _operand_input, _operand_output
from Mining.Expression.Dimension import DimensionType as T
from Mining.Data.Data import Data

# Operator base classes

class OperatorType(IntEnum):
    unary = 0
    binary = 1
    ternary = 2

class Operator(Expression):
    @abstractmethod
    def n_args(self) -> int: ...

    @abstractmethod
    def category_type(self) -> OperatorType: ...

    @abstractmethod
    def validate_parameters(self, *args) -> bool: ...

    def _check_arity(self, *args) -> bool:
        arity = self.n_args()
        if len(args) == arity:
            return True
        else:
            raise RuntimeError(f"{self.__name__} expects {arity} operand(s), but received {len(args)}")

    def _check_exprs_featured(self, args: list[Operand]) -> bool:
        any_is_featured: bool = False
        for i, arg in enumerate(args):
            if not isinstance(arg, Operand):
                raise RuntimeError(f"{arg} is not a valid expression")
            if DimensionType.timedelta in arg.Dimension:
                raise RuntimeError(f"{self.__name__} expects a normal expression for operand {i + 1}, "
                            f"but got {arg} (a DeltaTime)")
            any_is_featured = any_is_featured or arg.is_featured
        if not any_is_featured:
            if len(args) == 1:
                raise RuntimeError(f"{self.__name__} expects a featured expression for its operand, "
                            f"but {args[0]} is not featured")
            else:
                raise RuntimeError(f"{self.__name__} expects at least one featured expression for its operands, "
                            f"but none of {args} is featured")
        return True

    def _check_delta_time(self, arg) -> bool:
        if DimensionType.timedelta not in arg.Dimension:
            raise RuntimeError(f"{self.__name__} expects a DeltaTime as its last operand, but {arg} is not")
        return True

    @property
    @abstractmethod
    def operands(self) -> Tuple[Expression, ...]: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({','.join(str(op) for op in self.operands)})"


def get_subtensor(X: Tensor, i: slice, axis: int) -> Tensor:
    # Create a list of slices (slice(None) for all dimensions initially)
    slices = [slice(None)] * X.ndim  # Initialize with selecting all elements
    # Set the index for the chosen axis (i is not a slice anymore)
    slices[axis] = i

    # Return the subtensor at index i along the given axis
    return X[tuple(slices)]


def RollingOp_1D(Op, X, window, axis):
    '''Return the result of applying Op to a rolling window over a specified axis'''
    # Ensure axis is positive
    axis = axis % X.ndim
    # Compute the padding: PyTorch requires padding in reverse order
    # [..., dim2_right, dim2_left, dim1_bottom, dim1_top, dim0_right, dim0_left]
    pad = [0, 0] * (X.ndim - axis - 1) + [0, window - 1]
    # Apply padding along the chosen axis
    PaddedX = torch.nn.functional.pad(X, pad, mode='replicate')
    # Create a rolling window view along the specified axis
    # this expand the dimension by 1, but won't copy data
    # so don't worry about efficiency
    XView = PaddedX.unfold(axis, window, step=1)
    # Apply the operation (e.g., torch.sum, torch.mean) over the rolling window dimension
    Xroll = Op(XView, dim=-1)
    return Xroll


def RollingOp_2D(Op, X, Y, window, axis):
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
    import torch

    # Ensure axis is positive
    axis = axis % X.ndim

    # Compute the padding: pad the beginning of the axis with (window - 1) elements
    pad = [0] * (2 * X.ndim)
    # PyTorch's pad expects padding for each dimension in the order:
    # (dim_n_pad_right, dim_n_pad_left, ..., dim_0_pad_right, dim_0_pad_left)
    # So we need to calculate the correct indices
    pad_index = (X.ndim - axis - 1) * 2
    pad[pad_index + 1] = window - 1  # Left padding for the specified axis

    # Apply padding to both X and Y
    PaddedX = torch.nn.functional.pad(X, pad, mode='replicate')
    PaddedY = torch.nn.functional.pad(Y, pad, mode='replicate')

    # Unfold X and Y along the specified axis
    X_view = PaddedX.unfold(axis, window, step=1)
    Y_view = PaddedY.unfold(axis, window, step=1)

    # Apply the operation Op to the unfolded views
    result = Op(X_view, Y_view)

    return result

class UnaryOperator(Operator):
    def __init__(self, operand0:_operand_input, dimension:Dimension) -> None:
        self._operand0 = into_operand(operand0, dimension)
        assert self._operand0.OperandType == OperandType.matrix
        self.init()

    @abstractmethod
    def init(self) -> None: ...

    def n_args(self) -> int: return 1

    def category_type(self): return OperatorType.unary

    def validate_parameters(self, *args) -> bool:
        check = True
        check = check and self._check_arity(*args)
        check = check and self._check_exprs_featured([args[0]])
        check_dim, dimension = self._check_dimension()
        self.dimension = Dimension([dimension])
        check = check and check_dim
        return check

    def evaluate(self, data: Data, period: slice = slice(0, 1)) -> _operand_output:
        return self._apply(self._operand0.evaluate(data))

    @abstractmethod
    def _check_dimension(self) -> Tuple[bool, DimensionType]: ...

    @abstractmethod
    def _apply(self, _operand0: _operand_output) -> _operand_output: ...

    @property
    def operands(self): return self._operand0,

    @property
    def is_featured(self): return self._operand0.is_featured


class BinaryOperator(Operator):
    def __init__(self, operand0:_operand_input, operand1:_operand_input, dimension:Dimension) -> None:
        self._operand0 = into_operand(operand0, dimension)
        self._operand1 = into_operand(operand1, dimension)
        assert self._operand0.OperandType == OperandType.matrix
        self.rolling = DimensionType.timedelta in self._operand1.Dimension
        self.init()

    @abstractmethod
    def init(self) -> None: ...

    def n_args(self) -> int: return 2

    def category_type(self):
        return OperatorType.binary

    def validate_parameters(self, *args) -> bool:
        check = True
        check = check and self._check_arity(*args)
        check = check and self._check_exprs_featured([args[0]])
        if self.rolling:
            check = check and self._check_delta_time([args[1]])
        else:
            check = check and self._check_exprs_featured([args[1]])
        check_dim, dimension = self._check_dimension()
        self.dimension = Dimension([dimension])
        check = check and check_dim
        return check

    def evaluate(self, data: Data, period: slice = slice(0, 1)) -> _operand_output:
        return self._apply(
            self._operand0.evaluate(data),
            self._operand1.evaluate(data),
        )

    @abstractmethod
    def _check_dimension(self) -> Tuple[bool, DimensionType]: ...

    @abstractmethod
    def _apply(self, _operand0: _operand_output, _operand1: _operand_output) -> _operand_output: ...

    def __str__(self) -> str: return f"{type(self).__name__}({self._operand0},{self._operand1})"

    @property
    def operands(self): return self._operand0, self._operand1

    @property
    def is_featured(self): return self._operand0.is_featured or self._operand1.is_featured

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
    def __init__(self, operand0:_operand_input, operand1:_operand_input, operand2:_operand_input, dimension:Dimension) -> None:
        self._operand0 = into_operand(operand0, dimension)
        self._operand1 = into_operand(operand1, dimension)
        self._operand2 = into_operand(operand2, dimension)
        assert self._operand0.OperandType == OperandType.matrix
        assert self._operand1.OperandType == OperandType.matrix
        self.rolling = DimensionType.timedelta in self._operand2.Dimension
        self.init()

    @abstractmethod
    def init(self) -> None: ...

    def n_args(self) -> int: return 3

    def category_type(self):
        return OperatorType.ternary

    def validate_parameters(self, *args) -> bool:
        check = True
        check = check and self._check_arity(*args)
        check = check and self._check_exprs_featured([args[0]])
        check = check and self._check_exprs_featured([args[1]])
        if self.rolling:
            check = check and self._check_delta_time([args[2]])
        else:
            check = check and self._check_exprs_featured([args[2]])
        check_dim, dimension = self._check_dimension()
        self.dimension = Dimension([dimension])
        check = check and check_dim
        return check

    def evaluate(self, data: Data, period: slice = slice(0, 1)) -> _operand_output:
        return self._apply(
            self._operand0.evaluate(data),
            self._operand1.evaluate(data),
            self._operand2.evaluate(data),
            )

    @abstractmethod
    def _check_dimension(self) -> Tuple[bool, DimensionType]: ...

    @abstractmethod
    def _apply(self, _operand0: _operand_output, _operand1: _operand_output, _operand2: _operand_output) -> _operand_output: ...

    def __str__(self) -> str: return f"{type(self).__name__}({self._operand0},{self._operand1})"

    @property
    def operands(self): return self._operand0, self._operand1, self._operand2

    @property
    def is_featured(self):
        return self._operand0.is_featured or self._operand1.is_featured or self._operand2.is_featured

def __check_dimension(
    dimension_map:list[list[list[T]]],
    operand:Union[UnaryOperator, BinaryOperator, TernaryOperator],
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
    return __check_dimension(map, self)
    """
    num_rules = len(dimension_map)
    num_operands = len(dimension_map[0]) - 1
    for rule_idx in range(num_rules):
        check = True
        for oprd_idx in range(num_operands):
            dimension:Dimension = getattr(operand, f'_operand{oprd_idx}').Dimension
            check = check and dimension.are_in(Dimension(dimension_map[rule_idx][oprd_idx]))
        if check: return True, dimension_map[rule_idx][-1] # type: ignore
    return False, T.misc


# Operator implementations

class Abs(UnaryOperator):
    def init(self):
        self.TS = False
        self.CS = False
    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price],T.price],
            [[T.volume],T.volume],
            [[T.ratio],T.ratio],
            [[T.misc],T.misc],
            [[T.oscillator],T.oscillator],
        ]
        return __check_dimension(map, self)
    def _apply(self, operand: Tensor) -> Tensor: return operand.abs()


class Sign(UnaryOperator):
    def init(self):
        self.TS = False
        self.CS = False
    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.ratio],T.condition],
            [[T.misc],T.condition],
            [[T.oscillator],T.condition],
        ]
        return __check_dimension(map, self)
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign() # -1. 0. 1.


class Log1p(UnaryOperator):
    def init(self):
        self.TS = False
        self.CS = False
    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.volume],T.misc],
            [[T.ratio],T.ratio],
            [[T.misc],T.misc],
            [[T.oscillator],T.misc],
        ]
        return __check_dimension(map, self)
    def _apply(self, operand: Tensor) -> Tensor: return operand.abs().log1p()


class CS_Rank(UnaryOperator):
    def init(self):
        self.TS = False
        self.CS = True
    def _check_dimension(self) -> Tuple[bool, DimensionType]:
        map = [
            [[T.price],T.ratio],
            [[T.volume],T.ratio],
            [[T.ratio],T.ratio],
            [[T.misc],T.ratio],
            [[T.oscillator],T.ratio],
        ]
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        ]
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)

    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        def _DelT(X: Tensor, dim: int):
            return get_subtensor(X=X, i=slice(-1, None), axis=dim)
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
        return __check_dimension(map, self)
    
    def _apply(self, operand0: Tensor, operand1: int) -> Tensor:
        def _DelT(X: Tensor, dim: int):
            return get_subtensor(X=X, i=slice(0, None), axis=dim) \
                - get_subtensor(X=X, i=slice(-1, None), axis=dim)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
        return __check_dimension(map, self)
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
            [[T.price], [T.price,T.misc], [T.timedelta], T.ratio],
            [[T.volume], [T.volume,T.misc], [T.timedelta], T.ratio],
            [[T.ratio], [T.ratio,T.misc], [T.timedelta], T.ratio],
            [[T.misc], [T.price,T.volume,T.ratio,T.misc,T.oscillator], [T.timedelta], T.ratio],
            [[T.oscillator], [T.oscillator,T.misc], [T.timedelta], T.ratio],
        ]
        return __check_dimension(map, self)
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

