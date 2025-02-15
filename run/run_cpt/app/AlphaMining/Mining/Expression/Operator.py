import torch
from torch import Tensor
from abc import abstractmethod
from enum import IntEnum
from typing import List, Type, Union, Tuple

from Mining.Expression.Expression import Expression
from Mining.Expression.Content import ContentType, Content
from Mining.Expression.Operand import OperandType, Operand, into_operand, _operand_input, _operand_output
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
            if ContentType.timedelta in arg.Content:
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
        if ContentType.timedelta not in arg.Content:
            raise RuntimeError(f"{self.__name__} expects a DeltaTime as its last operand, but {arg} is not")
        return True

    @property
    @abstractmethod
    def operands(self) -> Tuple[Expression, ...]: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({','.join(str(op) for op in self.operands)})"

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
    def __init__(self, operand0:_operand_input, content:Content) -> None:
        self._operand0 = into_operand(operand0, content)
        assert self._operand0.OperandType == OperandType.matrix
        self.content = self._output_content()

    def n_args(self) -> int: return 1

    def category_type(self): return OperatorType.unary

    def validate_parameters(self, *args) -> bool:
        check = True
        check = check and self._check_arity(*args)
        check = check and self._check_exprs_featured([args[0]])
        check = check and self._check_content()
        return check

    def evaluate(self, data: Data, period: slice = slice(0, 1)) -> _operand_output:
        return self._apply(self._operand0.evaluate(data, period))

    @abstractmethod
    def _output_content(self) -> bool: ...

    @abstractmethod
    def _check_content(self) -> bool: ...

    @abstractmethod
    def _apply(self, _operand0: _operand_output) -> _operand_output: ...

    @property
    def operands(self): return self._operand0,

    @property
    def is_featured(self): return self._operand0.is_featured


class BinaryOperator(Operator):
    def __init__(self, operand0:_operand_input, operand1:_operand_input, content:Content) -> None:
        self._operand0 = into_operand(operand0, content)
        self._operand1 = into_operand(operand1, content)
        assert self._operand0.OperandType == OperandType.matrix
        self.rolling = ContentType.timedelta in self._operand1.Content
        self.content = self._output_content()

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
        check = check and self._check_content()
        return check

    def evaluate(self, data: Data, period: slice = slice(0, 1)) -> _operand_output:
        return self._apply(
            self._operand0.evaluate(data, period),
            self._operand1.evaluate(data, period),
            )

    @abstractmethod
    def _output_content(self) -> bool: ...

    @abstractmethod
    def _check_content(self) -> bool: ...

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
    def __init__(self, operand0:_operand_input, operand1:_operand_input, operand2:_operand_input, content:Content) -> None:
        self._operand0 = into_operand(operand0, content)
        self._operand1 = into_operand(operand1, content)
        self._operand2 = into_operand(operand2, content)
        assert self._operand0.OperandType == OperandType.matrix
        assert self._operand1.OperandType == OperandType.matrix
        self.rolling = ContentType.timedelta in self._operand2.Content
        self.content = self._output_content()

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
        check = check and self._check_content()
        return check

    def evaluate(self, data: Data, period: slice = slice(0, 1)) -> _operand_output:
        return self._apply(
            self._operand0.evaluate(data, period),
            self._operand1.evaluate(data, period),
            self._operand2.evaluate(data, period),
            )

    @abstractmethod
    def _output_content(self) -> bool: ...

    @abstractmethod
    def _check_content(self) -> bool: ...

    @abstractmethod
    def _apply(self, _operand0: _operand_output, _operand1: _operand_output, _operand2: _operand_output) -> _operand_output: ...

    def __str__(self) -> str: return f"{type(self).__name__}({self._operand0},{self._operand1})"

    @property
    def operands(self): return self._operand0, self._operand1, self._operand2

    @property
    def is_featured(self):
        return self._operand0.is_featured or self._operand1.is_featured or self._operand2.is_featured

def __check_content(
    content_array:list[list[str]],
    operand:Union[UnaryOperator, BinaryOperator, TernaryOperator],
    identical_operands:List[int] = [], # ascending
    ) -> bool:
    """
    'price', 'volume', 'ratio', 'misc', 'timedelta', 'oscillator', 'condition',
    [[allow_list_operand_0],[allow_list_operand_1],...]
    """
    num_operands = len(content_array)
    check = True
    for i in range(num_operands):
        content:Content = getattr(operand, f'_operand{i}').Content
        check = check and content.are_in(Content(content_array[i]))
        if i in identical_operands:
            if i == identical_operands[0]:
                root = content
            else:
                check = check and content.identical(root)
    return check


# Operator implementations

class Abs(UnaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [])
    def _apply(self, operand: Tensor) -> Tensor: return operand.abs()


class Sign(UnaryOperator):
    def _output_content(self) -> Content:
        return Content(['condition'])
    def _check_content(self) -> bool:
        return __check_content([
            ['ratio', 'misc', 'oscillator',],
            ], self, [])
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign() # -1. 0. 1.


class Log1p(UnaryOperator):
    def _output_content(self) -> Content:
        content = self._operand0.Content
        if ['oscillator'] in content: return Content(['misc'])
        return content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [])
    def _apply(self, operand: Tensor) -> Tensor: return operand.abs().log1p()


class CSRank(UnaryOperator):
    def _output_content(self) -> Content:
        return Content(['misc'])
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [])
    def _apply(self, operand: Tensor) -> Tensor:
        nan_mask = operand.isnan()
        n = (~nan_mask).sum(dim=1, keepdim=True)
        rank = operand.argsort().argsort() / n
        rank[nan_mask] = torch.nan
        return rank


class Add(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs + rhs


class Sub(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs - rhs


class Mul(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs * rhs


class Div(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs / rhs


class Pow(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs ** rhs


class Greater(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.max(rhs)


class Less(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.min(rhs)


class Ref(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    # Ref is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Ref only deal with the values
    # at -dt. Nonetheless, it should be classified as rolling since it modifies
    # the time window.

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop - self._delta_time
        return self._operand.evaluate(data, slice(start, stop))

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...


class Mean(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, operand: Tensor) -> Tensor: return operand.mean(dim=-1)


class Sum(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, operand: Tensor) -> Tensor: return operand.sum(dim=-1)


class Std(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, operand: Tensor) -> Tensor: return operand.std(dim=-1)


class Var(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, operand: Tensor) -> Tensor: return operand.var(dim=-1)


class Skew(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, operand: Tensor) -> Tensor:
        # skew = m3 / m2^(3/2)
        central = operand - operand.mean(dim=-1, keepdim=True)
        m3 = (central ** 3).mean(dim=-1)
        m2 = (central ** 2).mean(dim=-1)
        return m3 / m2 ** 1.5


class Kurt(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, operand: Tensor) -> Tensor:
        # kurt = m4 / var^2 - 3
        central = operand - operand.mean(dim=-1, keepdim=True)
        m4 = (central ** 4).mean(dim=-1)
        var = operand.var(dim=-1)
        return m4 / var ** 2 - 3


class Max(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, operand: Tensor) -> Tensor: return operand.max(dim=-1)[0]


class Min(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, operand: Tensor) -> Tensor: return operand.min(dim=-1)[0]


class Med(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(
        self, operand: Tensor) -> Tensor: return operand.median(dim=-1)[0]


class Mad(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, operand: Tensor) -> Tensor:
        central = operand - operand.mean(dim=-1, keepdim=True)
        return central.abs().mean(dim=-1)


class Rank(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        last = operand[:, :, -1, None]
        left = (last < operand).count_nonzero(dim=-1)
        right = (last <= operand).count_nonzero(dim=-1)
        result = (right + left + (right > left)) / (2 * n)
        return result


class Delta(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    # Delta is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Delta only deal with the values
    # at -dt and 0. Nonetheless, it should be classified as rolling since it
    # modifies the time window.

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop))
        return values[self._delta_time:] - values[:-self._delta_time]

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...


class WMA(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        weights = torch.arange(n, dtype=operand.dtype, device=operand.device)
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class EMA(BinaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        alpha = 1 - 2 / (1 + n)
        power = torch.arange(n, 0, -1, dtype=operand.dtype,
                             device=operand.device)
        weights = alpha ** power
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class Cov(TernaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        n = lhs.shape[-1]
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        return (clhs * crhs).sum(dim=-1) / (n - 1)


class Corr(TernaryOperator):
    def _output_content(self) -> Content:
        return self._operand0.Content
    def _check_content(self) -> bool:
        return __check_content([
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ['price', 'volume', 'ratio', 'misc', 'oscillator',],
            ], self, [0,1,])
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        ncov = (clhs * crhs).sum(dim=-1)
        nlvar = (clhs ** 2).sum(dim=-1)
        nrvar = (crhs ** 2).sum(dim=-1)
        stdmul = (nlvar * nrvar).sqrt()
        stdmul[(nlvar < 1e-6) | (nrvar < 1e-6)] = 1
        return ncov / stdmul


Operators: List[Type[Operator]] = [
    # Unary
    Abs, Sign, Log1p, CSRank,
    # Binary
    Add, Sub, Mul, Div, Pow, Greater, Less,
    # Rolling
    Ref, Mean, Sum, Std, Var, Skew, Kurt, Max, Min,
    Med, Mad, Rank, Delta, WMA, EMA,
    # Pair rolling
    Cov, Corr
]
