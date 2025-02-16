import torch
from torch import Tensor
from AlphaMining.Mining.Expression.Content import \
    Value, Dimension, Token, UnaryOpToken, BinaryOpToken, TernaryOpToken
from Mining.Expression.Operands import \
    register_operands, scalar_operands, vector_operands, matrix_operands

from copy import copy


def copy_dim(dim):
    # copy the dimension
    return Dimension(copy(dim))


def add_sub_dim(dim1, dim2):
    # return the new dimension for the addition of two dimensions
    return Dimension([])


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

# Dynamic slicing along any axis


def get_subtensor(X: Tensor, i: slice, axis: int) -> Tensor:
    # Create a list of slices (slice(None) for all dimensions initially)
    slices = [slice(None)] * X.ndim  # Initialize with selecting all elements
    # Set the index for the chosen axis (i is not a slice anymore)
    slices[axis] = i

    # Return the subtensor at index i along the given axis
    return X[tuple(slices)]


# Placeholder ===========================================================================


class StartToken(Token):
    def __init__(self):
        self.s = "Start"
        self.name = 'Start'


class FinishToken(Token):
    def __init__(self):
        self.s = "Finish"
        self.name = 'Finish'

# Register ==============================================================================


class RegisterToken(Token):  # the number of registers corresponds to the max fan-out
    def __init__(self, idx: str):
        self.idx = idx
        self.name = "Reg"+str(idx)
        self.s = "Reg"+str(idx)  # just for debugging
        self.value = None  # for type check, should not be assigned

# Operators =============================================================================
# Unary Operator


class RevToken(UnaryOpToken):
    def __init__(self):
        self.s = "-{}"
        self.name = 'Reverse'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val = values[0].value
        res_val = - val
        res_dim = copy_dim(values[0].dimension)
        return Value(value=res_val, dimension=res_dim)


class SignToken(UnaryOpToken):
    def __init__(self):
        self.s = "Sign{}"
        self.name = 'Sign'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val = values[0].value
        res_val = Tensor.sign(val)
        res_dim = copy_dim(values[0].dimension)
        return Value(value=res_val, dimension=res_dim)


class AbsToken(UnaryOpToken):
    def __init__(self):
        self.s = "Abs{}"
        self.name = 'Abs'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val = values[0].value
        res_val = Tensor.abs(val)
        res_dim = copy_dim(values[0].dimension)
        return Value(value=res_val, dimension=res_dim)


class LogToken(UnaryOpToken):
    """
    Log1p
    """

    def __init__(self):
        self.s = "Log1p{}"
        self.name = 'Log1p'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val = values[0].value
        res_val = Tensor.log1p(val)
        res_dim = copy_dim(values[0].dimension)
        return Value(value=res_val, dimension=res_dim)


class CSRankToken(UnaryOpToken):
    """
    The cross-sectional rank (CSRank) is an
    operator that returns the rank of the current
    stock’s feature value x relative to the feature
    values of all stocks on today’s date.
    """

    def __init__(self):
        self.s = "CSRank{}"
        self.name = 'CSRank'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val = values[0].value
        # Create a mask for NaN values in the feature values.
        nan_mask = val.isnan()
        # Count the number of valid (non-NaN) entries along the specified dimension.
        n = (~nan_mask).sum(dim=1, keepdim=True)
        # Compute the ranks by sorting the feature values twice,
        # then normalize by the count of valid entries.
        res_val = val.argsort().argsort() / n
        # Assign NaN to ranks that correspond to NaN entries in the original values.
        res_val[nan_mask] = torch.nan
        res_dim = copy_dim(values[0].dimension)
        return Value(value=res_val, dimension=res_dim)

# Binary Operator


class AddToken(BinaryOpToken):
    def __init__(self,):
        self.s = "({} + {})"
        self.name = 'Add'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val0, val1 = values
        res_val = Tensor.add(val0.value, val1.value)
        res_dim = add_sub_dim(val0.dimension, val1.dimension)
        return Value(value=res_val, dimension=res_dim)


class SubToken(BinaryOpToken):
    def __init__(self,):
        self.s = "({} - {})"
        self.name = 'Sub'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val0, val1 = values
        res_val = Tensor.sub(val0.value, val1.value)
        res_dim = add_sub_dim(val0.dimension, val1.dimension)
        return Value(value=res_val, dimension=res_dim)


class MulToken(BinaryOpToken):
    def __init__(self,):
        self.s = "({} * {})"
        self.name = 'Mul'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val0, val1 = values
        res_val = Tensor.mul(val0.value, val1.value)
        res_dim = add_sub_dim(val0.dimension, val1.dimension)
        return Value(value=res_val, dimension=res_dim)


class DivToken(BinaryOpToken):
    def __init__(self,):
        self.s = "({} / {})"
        self.name = 'Div'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val0, val1 = values
        res_val = Tensor.div(val0.value, val1.value)
        res_dim = add_sub_dim(val0.dimension, val1.dimension)
        return Value(value=res_val, dimension=res_dim)


class PowToken(BinaryOpToken):
    def __init__(self,):
        self.s = "({} ^ {})"
        self.name = 'Pow'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val0, val1 = values
        res_val = Tensor.pow(val0.value, val1.value)
        res_dim = add_sub_dim(val0.dimension, val1.dimension)
        return Value(value=res_val, dimension=res_dim)


class GreaterToken(BinaryOpToken):
    def __init__(self,):
        self.s = "({} > {})"
        self.name = 'Greater'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val0, val1 = values
        res_val = Tensor.greater(val0.value, val1.value)
        res_dim = add_sub_dim(val0.dimension, val1.dimension)
        return Value(value=res_val, dimension=res_dim)


class LessToken(BinaryOpToken):
    def __init__(self,):
        self.s = "({} < {})"
        self.name = 'Less'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val0, val1 = values
        res_val = Tensor.less(val0.value, val1.value)
        res_dim = add_sub_dim(val0.dimension, val1.dimension)
        return Value(value=res_val, dimension=res_dim)


class DelTimeRToken(BinaryOpToken):
    """
    Delta time Rolling Token:
    other rolling tokens deal with the values in (-dt, 0],
    this token only deal with value at -dt
    """

    def __init__(self,):
        self.s = "DelT_R({}, {})"
        self.name = 'DelT_R'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        def _DelT(X: Tensor, dim: int):
            return get_subtensor(X=X, i=slice(-1, None), axis=dim)

        val0, val1 = values
        res_val = RollingOp_1D(_DelT, val0.value, val1.value, 0)
        res_dim = copy_dim(val0.dimension)
        return Value(value=res_val, dimension=res_dim)


class DelValDelTimeRToken(BinaryOpToken):
    """
    Delta Value over Delta time Rolling Token
    """

    def __init__(self,):
        self.s = "DVDT_R({}, {})"
        self.name = 'DVDT_R'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        def _DelVDelT(X: Tensor, dim: int):
            return get_subtensor(X=X, i=slice(0, None), axis=dim) \
                - get_subtensor(X=X, i=slice(-1, None), axis=dim)

        val0, val1 = values
        res_val = RollingOp_1D(_DelVDelT, val0.value, val1.value, 0)
        res_dim = copy_dim(val0.dimension)
        return Value(value=res_val, dimension=res_dim)


class MeanRToken(BinaryOpToken):
    def __init__(self,):
        self.s = "Mean_R({}, {})"
        self.name = 'Mean_R'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val0, val1 = values
        res_val = RollingOp_1D(Tensor.mean, val0.value, val1.value, 0)
        res_dim = copy_dim(val0.dimension)
        return Value(value=res_val, dimension=res_dim)


class SumRToken(BinaryOpToken):
    def __init__(self,):
        self.s = "Sum_R({}, {})"
        self.name = 'Sum_R'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val0, val1 = values
        res_val = RollingOp_1D(Tensor.sum, val0.value, val1.value, 0)
        res_dim = copy_dim(val0.dimension)
        return Value(value=res_val, dimension=res_dim)


class StdRToken(BinaryOpToken):
    def __init__(self,):
        self.s = "Std_R({}, {})"
        self.name = 'Std_R'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val0, val1 = values
        res_val = RollingOp_1D(Tensor.std, val0.value, val1.value, 0)
        res_dim = copy_dim(val0.dimension)
        return Value(value=res_val, dimension=res_dim)


class VarRToken(BinaryOpToken):
    def __init__(self,):
        self.s = "Var_R({}, {})"
        self.name = 'Var_R'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val0, val1 = values
        res_val = RollingOp_1D(Tensor.var, val0.value, val1.value, 0)
        res_dim = copy_dim(val0.dimension)
        return Value(value=res_val, dimension=res_dim)


class SkewRToken(BinaryOpToken):
    def __init__(self,):
        self.s = "Skew_R({}, {})"
        self.name = 'Skew_R'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        def _Skew(X: Tensor, dim: int):
            # skew = m3 / m2^(3/2)
            central = X - X.mean(dim=dim, keepdim=True)
            m3 = (central ** 3).mean(dim=dim)
            m2 = (central ** 2).mean(dim=dim)
            return m3 / m2 ** 1.5

        val0, val1 = values
        res_val = RollingOp_1D(_Skew, val0.value, val1.value, 0)
        res_dim = copy_dim(val0.dimension)
        return Value(value=res_val, dimension=res_dim)


class KurtToken(BinaryOpToken):
    def __init__(self,):
        self.s = "Kurt_R({}, {})"
        self.name = 'Kurt_R'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        def _Kurt(X: Tensor, dim: int):
            # kurt = m4 / var^2 - 3
            central = val0.value - val0.value.mean(dim=dim, keepdim=True)
            m4 = (central ** 4).mean(dim=dim)
            var = val0.value.var(dim=dim)
            return m4 / var ** 2 - 3

        val0, val1 = values
        res_val = RollingOp_1D(_Kurt, val0.value, val1.value, 0)
        res_dim = copy_dim(val0.dimension)
        return Value(value=res_val, dimension=res_dim)


class MaxRToken(BinaryOpToken):
    def __init__(self,):
        self.s = "Max_R({}, {})"
        self.name = 'Max_R'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val0, val1 = values
        res_val = RollingOp_1D(Tensor.max, val0.value, val1.value, 0)[0]
        res_dim = copy_dim(val0.dimension)
        return Value(value=res_val, dimension=res_dim)


class MinRToken(BinaryOpToken):
    def __init__(self,):
        self.s = "Min_R({}, {})"
        self.name = 'Min_R'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val0, val1 = values
        res_val = RollingOp_1D(Tensor.min, val0.value, val1.value, 0)[0]
        res_dim = copy_dim(val0.dimension)
        return Value(value=res_val, dimension=res_dim)


class MedRToken(BinaryOpToken):
    def __init__(self,):
        self.s = "Med_R({}, {})"
        self.name = 'Med_R'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        val0, val1 = values
        res_val = RollingOp_1D(Tensor.median, val0.value, val1.value, 0)[0]
        res_dim = copy_dim(val0.dimension)
        return Value(value=res_val, dimension=res_dim)


class MadRToken(BinaryOpToken):
    """
    Mean Absolute Deviation
    """

    def __init__(self,):
        self.s = "Mad_R({}, {})"
        self.name = 'Mad_R'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        def _Mad(X: Tensor, dim: int):
            central = X - X.mean(dim=dim, keepdim=True)
            return central.abs().mean(dim=dim)
        val0, val1 = values
        res_val = RollingOp_1D(_Mad, val0.value, val1.value, 0)
        res_dim = copy_dim(val0.dimension)
        return Value(value=res_val, dimension=res_dim)


class RankRToken(BinaryOpToken):
    def __init__(self,):
        self.s = "Rank_R({}, {})"
        self.name = 'Rank_R'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        def _Rank(X: Tensor, dim: int):
            n = X.shape[dim]
            # Extract the last value along the specified axis
            last = X.index_select(dim, torch.tensor([-1])).unsqueeze(dim)
            left = (last < X).count_nonzero(dim=dim)
            right = (last <= X).count_nonzero(dim=dim)
            return (right + left + (right > left)) / (2 * n)

        val0, val1 = values
        res_val = RollingOp_1D(_Rank, val0.value, val1.value, 0)
        res_dim = copy_dim(val0.dimension)
        return Value(value=res_val, dimension=res_dim)


class WMARToken(BinaryOpToken):
    """
    Weighted Moving Average Rolling Token
    """

    def __init__(self,):
        self.s = "WMA_R({}, {})"
        self.name = 'WMA_R'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        def _WMA(X: Tensor, dim: int):
            n = X.shape[-1]
            weights = torch.arange(n, dtype=X.dtype, device=X.device)
            weights /= weights.sum()
            return (weights * X).sum(dim=-1)

        val0, val1 = values
        res_val = RollingOp_1D(_WMA, val0.value, val1.value, 0)
        res_dim = copy_dim(val0.dimension)
        return Value(value=res_val, dimension=res_dim)


class EMARToken(BinaryOpToken):
    """
    Exponential Moving Average Rolling Token
    """

    def __init__(self,):
        self.s = "EMA_R({}, {})"
        self.name = 'EMA_R'

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        def _EMA(X: Tensor, dim: int):
            n = X.shape[-1]
            alpha = 1 - 2 / (1 + n)
            power = torch.arange(n, 0, -1, dtype=X.dtype, device=X.device)
            weights = alpha ** power
            weights /= weights.sum()
            return (weights * X).sum(dim=-1)

        val0, val1 = values
        res_val = RollingOp_1D(_EMA, val0.value, val1.value, 0)
        res_dim = copy_dim(val0.dimension)
        return Value(value=res_val, dimension=res_dim)

# ternary tokens


class CovRToken(TernaryOpToken):
    def __init__(self, s=None):
        self.s = "Cov_R({},{},{})"
        self.name = "Cov_R"

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        def _Cov(X: Tensor, Y: Tensor, dim: int):
            n = X.shape[-1]
            clhs = X - X.mean(dim=dim, keepdim=True)
            crhs = Y - Y.mean(dim=dim, keepdim=True)
            return (clhs * crhs).sum(dim=-1) / (n - 1)

        val0, val1, val2 = values
        res_val = RollingOp_2D(_Cov, val0.value, val1.value, val2.value, 0)
        res_dim = add_sub_dim(val0.dimension, val1.dimension)
        return Value(value=res_val, dimension=res_dim)


class CorrRToken(TernaryOpToken):
    def __init__(self, s=None):
        self.s = "Corr_R({},{},{})"
        self.name = "Corr_R"

    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True

    @staticmethod
    def cal(*values: Value):
        def _Corr(X: Tensor, Y: Tensor, dim: int):
            clhs = X - X.mean(dim=dim, keepdim=True)
            crhs = Y - Y.mean(dim=dim, keepdim=True)
            ncov = (clhs * crhs).sum(dim=dim)
            nlvar = (clhs ** 2).sum(dim=dim)
            nrvar = (crhs ** 2).sum(dim=dim)
            stdmul = (nlvar * nrvar).sqrt()
            stdmul[(nlvar < 1e-6) | (nrvar < 1e-6)] = 1
            return ncov / stdmul

        val0, val1, val2 = values
        res_val = RollingOp_2D(_Corr, val0.value, val1.value, val2.value, 0)
        res_dim = add_sub_dim(val0.dimension, val1.dimension)
        return Value(value=res_val, dimension=res_dim)

# Operands ==============================================================================


class ConstToken(Token):
    def __init__(self, value, s=None):
        self.value = value
        if s is None:
            self.s = str(s)
        else:
            self.s = s
        self.name = self.s

#  NULL for padding actions to the same length


class NullToken(Token):
    def __init__(self):
        self.s = "Null"
        self.name = 'Null'
        self.value = None

    def cal(self, *values):
        return None


UNARY_OP_TOKENS = [  # arity=1
    RevToken(), SignToken(), AbsToken(), LogToken(), CSRankToken()
]

BINARY_OP_TOKENS = [  # arity=2
    # normal -> tensor
    AddToken(), SubToken(), MulToken(), DivToken(
    ), PowToken(), GreaterToken(), LessToken(),
    # rolling_1d -> tensor
    DelTimeRToken(), DelValDelTimeRToken(), MeanRToken(), SumRToken(
    ), StdRToken(), VarRToken(), SkewRToken(), KurtToken(),
    MaxRToken(), MinRToken(), MedRToken(), MadRToken(
    ), RankRToken(), WMARToken(), EMARToken(),

]
TERNARY_OP_TOKENS = [  # arity=3
    # rolling_2d -> tensor
    CovRToken(), CorrRToken(),
]

# Operators
start_token = StartToken()
finish_token = FinishToken()
START_FINISH_TOKENS = [start_token, finish_token]
operator_tokens = \
UNARY_OP_TOKENS + BINARY_OP_TOKENS + TERNARY_OP_TOKENS + START_FINISH_TOKENS
start_token_idx = len(operator_tokens) - 2
finish_token_idx = len(operator_tokens) - 1
NUM_OPERATORS = len(operator_tokens)

# Operands
null_token = NullToken()
register_tokens = [RegisterToken(str(i)) for i in register_operands]

operand_tokens = \
    [ConstToken(value=v, s=str(v.value)) for v in scalar_operands] + \
    [ConstToken(value=v, s=k) for k, v in vector_operands.items()] + \
    [ConstToken(value=v, s=k) for k, v in matrix_operands.items()] + \
    register_tokens + \
    [null_token]
null_token_idx = operand_tokens.index(null_token)
NUM_OPERANDS = len(operand_tokens)
