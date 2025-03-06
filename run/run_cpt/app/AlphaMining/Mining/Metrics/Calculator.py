

from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, Sequence
from torch import Tensor
import torch

from Mining.Data.Data import Data
from Mining.Expression.Operand import Operand
from Mining.Util.Calc_Util import batch_pearsonr, batch_spearmanr

"""
+--------------------------------+----------------------------------------+---------------+
| Metric                         | Formula                                | Range         |
+--------------------------------+----------------------------------------+---------------+
| IC (Info Coeff)                | IC = Cov(P, A) / (σ_P * σ_A)           | [-1, 1]       |
| rIC (Rank Info Coef)           | rIC = corr(rank(P), rank(A))           | [-1, 1]       |
| ICIR (Info Coeff in Rank)      | ICIR = Mean(IC) / Std(IC)              | Typically > 0 |
| rICIR (Rank Info Coef in Rank) | rICIR = Mean(RIC) / Std(RIC)           | Typically > 0 |
| MIC (Maximal Info Coeff)       | Derived from an optimization process.  | [0, 1]        |
+--------------------------------+----------------------------------------+---------------+

+-----------------+-------------------------------------------------------+-----------------------------------------------------------+
| Metric Name     | Definition                                            | Formula                                                   |
+-----------------+-------------------------------------------------------+-----------------------------------------------------------+
| calc_single_IC  | Measures the Information Coefficient (IC) between     | IC = Corr(Alpha_Value, Target_Alpha)                      |
|                 | a single alpha and its target label.                  |                                                           |
+-----------------+-------------------------------------------------------+-----------------------------------------------------------+
| calc_single_rIC | Measures the Rank Information Coefficient (Rank IC)   | rIC = Corr(Rank(Alpha_Value), Rank(Target_Alpha))         |
|                 | between a single alpha and its target label.          |                                                           |
+-----------------+-------------------------------------------------------+-----------------------------------------------------------+
| calc_single_all | Provides both IC and Rank IC for a single alpha.      | (IC, rIC)                                                 |
|                 |                                                       |                                                           |
+-----------------+-------------------------------------------------------+-----------------------------------------------------------+
| calc_mutual_IC  | Measures the IC between two different alpha values.   | IC = Corr(Alpha1, Alpha2)                                 |
+-----------------+-------------------------------------------------------+-----------------------------------------------------------+
| calc_pool_IC    | Calculates the IC of a pool of alphas relative to a   | IC = Corr(Pooled_Alpha_Values, Target_Alpha)              |
|                 | predefined target label.                              |                                                           |
+-----------------+-------------------------------------------------------+-----------------------------------------------------------+
| calc_pool_rIC   | Calculates the Rank IC of a pool of alphas against a  | rIC = Corr(Rank(Pooled_Alpha_Values), Rank(Target_Alpha)) |
|                 | predefined target label.                              |                                                           |
+-----------------+-------------------------------------------------------+-----------------------------------------------------------+
| calc_pool_all   | Provides both IC and Rank IC for a pool of alphas.    | (IC, rIC)                                                 |
+-----------------+-------------------------------------------------------+-----------------------------------------------------------+
"""


class AlphaCalculator(metaclass=ABCMeta):
    @abstractmethod
    def calc_single_IC(self, expr: Operand) -> float:
        'Calculate IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_single_rIC(self, expr: Operand) -> float:
        'Calculate Rank IC between a single alpha and a predefined target.'

    def calc_single_all(self, expr: Operand) -> Tuple[float, float]:
        return self.calc_single_IC(expr), self.calc_single_rIC(expr)

    @abstractmethod
    def calc_mutual_IC(self, expr1: Operand, expr2: Operand) -> float:
        'Calculate IC between two alphas.'

    @abstractmethod
    def calc_pool_IC(self, exprs: Sequence[Operand], weights: Sequence[float]) -> float:
        'First combine the alphas linearly,'
        'then Calculate IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_rIC(self, exprs: Sequence[Operand], weights: Sequence[float]) -> float:
        'First combine the alphas linearly,'
        'then Calculate Rank IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_all(self, exprs: Sequence[Operand], weights: Sequence[float]) -> Tuple[float, float]:
        'First combine the alphas linearly,'
        'then Calculate both IC and Rank IC between the linear combination and a predefined target.'

class TensorAlphaCalculator(AlphaCalculator):
    def __init__(self, target: Optional[Tensor]) -> None:
        self._target = target

    @property
    @abstractmethod
    def n_timestamps(self) -> int: ...

    @property
    def target(self) -> Tensor:
        if self._target is None:
            raise ValueError(
                "A target must be set before calculating non-mutual IC.")
        return self._target

    @abstractmethod
    def evaluate_alpha(self, expr: Operand) -> Tensor:
        'Evaluate an alpha into a `Tensor` of shape (timestamps, codes).'

    def make_composite_alpha(self, exprs: Sequence[Operand], weights: Sequence[float]) -> Tensor:
        n = len(exprs)
        factors = [self.evaluate_alpha(exprs[i]) * weights[i]for i in range(n)]
        return torch.sum(torch.stack(factors, dim=0), dim=0)

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()
    
    def _IR_from_batch(self, batch: Tensor) -> float:
        mean, std = batch.mean(), batch.std()
        return (mean / std).item()
    
    def _calc_ICIR(self, value1: Tensor, value2: Tensor) -> float:
        return self._IR_from_batch(batch_pearsonr(value1, value2))
    
    def _calc_rICIR(self, value1: Tensor, value2: Tensor) -> float:
        return self._IR_from_batch(batch_spearmanr(value1, value2))

    def calc_single_IC(self, expr: Operand) -> float:
        return self._calc_IC(self.evaluate_alpha(expr), self.target)

    def calc_single_IC_vector(self, expr: Operand) -> Tensor:
        return batch_pearsonr(self.evaluate_alpha(expr), self.target)

    def calc_single_rIC(self, expr: Operand) -> float:
        return self._calc_rIC(self.evaluate_alpha(expr), self.target)

    def calc_single_rIC_vector(self, expr: Operand) -> Tensor:
        return batch_spearmanr(self.evaluate_alpha(expr), self.target)

    def calc_single_all(self, expr: Operand) -> Tuple[float, float]:
        value = self.evaluate_alpha(expr)
        target = self.target
        return self._calc_IC(value, target), self._calc_rIC(value, target)

    def calc_mutual_IC(self, expr1: Operand, expr2: Operand) -> float:
        return self._calc_IC(self.evaluate_alpha(expr1), self.evaluate_alpha(expr2))

    def calc_mutual_IC_vector(self, expr1: Operand, expr2: Operand) -> Tensor:
        return batch_pearsonr(self.evaluate_alpha(expr1), self.evaluate_alpha(expr2))

    def calc_pool_IC(self, exprs: Sequence[Operand], weights: Sequence[float]) -> float:
        with torch.no_grad():
            value = self.make_composite_alpha(exprs, weights)
            return self._calc_IC(value, self.target)

    def calc_pool_rIC(self, exprs: Sequence[Operand], weights: Sequence[float]) -> float:
        with torch.no_grad():
            value = self.make_composite_alpha(exprs, weights)
            return self._calc_rIC(value, self.target)

    def calc_pool_all(self, exprs: Sequence[Operand], weights: Sequence[float]) -> Tuple[float, float]:
        with torch.no_grad():
            value = self.make_composite_alpha(exprs, weights)
            target = self.target
            return self._calc_IC(value, target), self._calc_rIC(value, target)

    def calc_pool_all_with_ir(self, exprs: Sequence[Operand], weights: Sequence[float]) -> Tuple[float, float, float, float]:
        "Returns IC, ICIR, Rank IC, Rank ICIR"
        with torch.no_grad():
            value = self.make_composite_alpha(exprs, weights)
            target = self.target
            ics = batch_pearsonr(value, target)
            rics = batch_spearmanr(value, target)
            ic_mean, ic_std = ics.mean().item(), ics.std().item()
            ric_mean, ric_std = rics.mean().item(), rics.std().item()
            return ic_mean, ic_mean / ic_std, ric_mean, ric_mean / ric_std


class ExpressionCalculator(TensorAlphaCalculator):
    """
    Used to evaluate alpha formula of 
    linked(but not evaluated) Operators/Operands
    """

    def __init__(
            self,
            data: Data,
            target: Optional[str] = None):
        self.data = data
        if target:
            label_idx = self.data.labels.index(target)
            start = data.max_past
            stop = data.max_past + data.n_timestamps + data.max_future - 1
            target_tensor = data.labels_tensor[start:stop, label_idx, :]
            super().__init__(target_tensor)
        else:
            super().__init__(None)

    def evaluate_alpha(self, expr: Operand) -> Tensor:
        return expr.final_evaluate(self.data)

    @property
    def n_timestamps(self) -> int:
        return self.data.n_timestamps
