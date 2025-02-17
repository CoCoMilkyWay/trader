import math
from itertools import count
from typing import List, Optional, Tuple, Iterable, Dict, Any, Set
from abc import ABCMeta, abstractmethod

import numpy as np
import torch

from Mining.Expression.Operand import Operand
from Mining.AlphaPool.AlphaPoolBase import AlphaPoolBase
from Mining.AlphaPool.AlphaPoolUpdate import PoolUpdate, AddRemoveAlphas
from Mining.Metrics.Calculator import TensorAlphaCalculator, ExpressionCalculator
from Mining.Util.Calc_Util import batch_pearsonr, batch_spearmanr
from Mining.Config import IC_LOWER_BOUND


class LinearAlphaPool(AlphaPoolBase, metaclass=ABCMeta):
    def __init__(
        self,
        capacity: int,              # Maximum number of expressions in the pool
        calculator: TensorAlphaCalculator,  # Calculator for assessing performance metrics
        # Device to perform calculations on, default is CPU
        device: torch.device = torch.device("cpu")
    ):
        # Initialize the parent class
        super().__init__(capacity, calculator, device)

        # Properties of the pool
        self.capacity = capacity  # Set the capacity of the pool
        # Storage for expressions added to the pool
        self._exprs: List[Operand] = []
        # Initialize weights for the expressions
        self._weights: np.ndarray = np.zeros(capacity + 1)
        self._single_ics: np.ndarray = np.zeros(capacity + 1)
        self._mutual_ics: np.ndarray = np.identity(capacity + 1)
        # Store extra information for each expression
        self._extra_info = [None for _ in range(capacity + 1)]
        self._ic_lower_bound = IC_LOWER_BOUND  # Lower bound threshold for IC values

        # Internal state
        self.size = 0  # Current number of expressions in the pool
        self.best_obj = -1.0  # Best objective value found so far
        self.best_ic = 0.0  # Best IC value found so far
        self.eval_cnt = 0  # Count of evaluations made
        # History of updates to the pool
        self.update_history: List[PoolUpdate] = []
        # Cache for failed expressions to avoid re-evaluation
        self.failure_cache: Set[str] = set()

    # --- Properties ---

    @property
    def exprs(self) -> List[str]:
        """Get the expressions of the linear model as a numpy array of shape (size,)."""
        return [str(expr) for expr in self._exprs[:self.size]]  # Convert expressions to strings for output

    @property
    def weights(self) -> np.ndarray:
        """Get the weights of the linear model as a numpy array of shape (size,)."""
        return self._weights[:self.size]  # Return only the weights corresponding to current expressions

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        """Set the weights of the linear model with a numpy array of shape (size,)."""
        assert value.shape == (
            # Ensure that the weights array shape matches the size
            self.size,), f"Invalid weights shape: {value.shape}"
        # Update weights for the current expressions
        self._weights[:self.size] = value

    @property
    def state(self) -> Dict[str, Any]:
        """Return the current state of the pool."""
        return {
            "_exprs": self._exprs[:self.size],    # Current expressions
            "weights": list(self._weights[:self.size]),  # Current weights
            "ics": list(self._single_ics[:self.size]),  # Current single ICs
            "best_ic": self.best_ic  # Best IC value found so far
        }

    # --- JSON Serialization ---
    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize the pool to a JSON-compatible dictionary."""
        return {
            "exprs": self.exprs,
            "weights": list(self.weights)
        }

    # --- Core Methods ---
    def try_new_formula(self, expr: Operand) -> float:
        """Attempt to add a new formula to the pool, updating weights and objectives."""
        ic, ic_mut = self._calc_ics(
            expr, ic_mut_threshold=0.99)  # Calculate single and mutual ICs for the expression

        # Check if the calculated IC values are valid
        if ic is None or ic_mut is None or np.isnan(ic) or np.isnan(ic_mut).any():
            return 0.0  # Skip if IC values are invalid
        if str(expr) in self.failure_cache:  # Skip if this expression has failed in the past
            return self.best_obj

        self.eval_cnt += 1  # Increment the evaluation count
        # Store the old pool of expressions
        old_pool: List[Operand] = self._exprs[:self.size]

        # Add the new factor (expression) to the pool
        self._add_factor(expr, ic, ic_mut)

        if self.size > 1:
            new_weights = self.optimize()  # Optimize the weights of the pool
            worst_idx = None

            if self.size > self.capacity:  # Check if the pool is over capacity
                # Find the index of the worst expression
                worst_idx = int(np.argmin(np.abs(new_weights)))
                if worst_idx == self.capacity:  # Revert addition if the new expression is the worst
                    self._pop(worst_idx)  # Remove the worst expression
                    # Cache this expression as failed
                    self.failure_cache.add(str(expr))
                    return self.best_obj  # Return the best objective without changes

            self.weights = new_weights  # Update the weights of the pool

            # Store index of removed expression, if any
            removed_idx = [worst_idx] if worst_idx is not None else []
            self.update_history.append(
                AddRemoveAlphas(  # Log the addition and removal of expressions
                    added_exprs=[expr],
                    removed_idx=removed_idx,
                    old_pool=old_pool,
                    old_pool_ic=self.best_ic,
                    new_pool_ic=ic,
                )
            )

            if worst_idx is not None:
                # remove again so there is 1 vacancy next time
                self._pop(worst_idx)
        else:
            self.update_history.append(
                AddRemoveAlphas(  # For the first expression, simply log its addition
                    added_exprs=[expr],
                    removed_idx=[],
                    old_pool=[],
                    old_pool_ic=0.0,
                    new_pool_ic=ic,
                )
            )

        self.failure_cache = set()  # Clear the failure cache after a successful addition
        # Calculate new IC and objective
        new_ic, new_obj = self.calculate_ic_and_objective()
        # Update best values if the new ones are better
        self._update_best_if_better(new_ic, new_obj)
        return new_obj  # Return the new objective value

    # --- Operand Management ---
    def load_formulas(self, _exprs: List[Operand], weights: Optional[List[float]] = None) -> None:
        """Forcefully load formulas and optionally set weights."""
        self.failure_cache = set()  # Clear failure cache
        old_ic = self.evaluate_composite()  # Evaluate the old composite IC
        # Store old pool state
        old_pool: List[Operand] = self._exprs[:self.size]
        added = []  # Keep track of added expressions

        for expr in _exprs:
            if self.size >= self.capacity:  # Stop if capacity is reached
                break
            try:
                ic, ic_mut = self._calc_ics(expr, ic_mut_threshold=None)
            except:
                # Handle calculation errors
                raise RuntimeError(
                    f"Data range error performing ic calculations")

            assert ic is not None and ic_mut is not None  # Ensure IC values are valid
            # Add the expression to the pool
            self._add_factor(expr, ic, ic_mut)
            added.append(expr)  # Track the added expression
            assert self.size <= self.capacity  # Ensure size is within capacity

        # Set weights if provided or optimize
        self.weights = np.array(weights) if weights else self.optimize()

        # Calculate new IC and objective
        new_ic, new_obj = self.calculate_ic_and_objective()
        # Update best IC and objective if better
        self._update_best_if_better(new_ic, new_obj)

        self.update_history.append(  # Log the additions to the pool
            AddRemoveAlphas(
                added_exprs=added,
                removed_idx=[],
                old_pool=old_pool,
                old_pool_ic=old_ic,
                new_pool_ic=new_ic,
            )
        )

    def leave_only(self, indices: Iterable[int]) -> None:
        """Leaves only the alphas at the given indices intact, and removes all others."""
        self.failure_cache = set()  # Clear the failure cache
        indices = sorted(indices)  # Sort indices for consistency
        for i, j in enumerate(indices):  # Iterate over indices to swap
            # Swap expressions to maintain only the specified ones
            self._swap_idx(i, j)
        self.size = len(indices)  # Update the size of the pool

    def bulk_edit(self, removed_indices: Iterable[int], added_exprs: List[Operand]) -> None:
        """Bulk remove and add formulas to the pool."""
        self.failure_cache = set()  # Clear the failure cache
        old_ic = self.evaluate_composite()  # Evaluate the old composite IC
        # Save the old pool state
        old_pool: List[Operand] = self._exprs[:self.size]

        # Create a set of removed indices for efficiency
        removed_indices = set(removed_indices)
        # Get remaining valid indices
        remain = [i for i in range(self.size) if i not in removed_indices]

        # Map of old expressions by ID
        old_exprs = {id(self._exprs[i]): i for i in range(self.size)}
        # Save the original state of update history
        original_update_count = len(self.update_history)

        self.leave_only(remain)  # Keep only the remaining expressions

        for expr in added_exprs:  # Add new expressions
            self.try_new_formula(expr)

        # Restore update history to original state
        self.update_history = self.update_history[:original_update_count]

        # Create a map of new expressions by ID
        new_exprs = {id(expr): expr for expr in self._exprs[:self.size]}
        added_exprs = [
            expr for expr in self._exprs[:self.size]
            if id(expr) not in old_exprs]  # Identify newly added expressions

        removed_indices = sorted(
            idx for expr_id, idx in old_exprs.items() if expr_id not in new_exprs
        )  # Sort removed indices for logging

        # Calculate new IC and objective
        new_ic, new_obj = self.calculate_ic_and_objective()
        # Update best metrics if applicable
        self._update_best_if_better(new_ic, new_obj)

        self.update_history.append(  # Log the additions and removals
            AddRemoveAlphas(
                added_exprs=added_exprs,
                removed_idx=removed_indices,
                old_pool=old_pool,
                old_pool_ic=old_ic,
                new_pool_ic=new_ic,
            )
        )

    # --- Objective Calculation ---
    def calculate_ic_and_objective(self) -> Tuple[float, float]:
        """Calculate and return the IC and the main optimization objective."""
        ic = self.evaluate_composite()  # Evaluate the composite IC
        # Calculate the objective, fallback to IC if None
        obj = self._calc_main_objective() or ic
        return ic, obj  # Return both values

    def _calc_main_objective(self) -> Optional[float]:
        """Get the main optimization objective."""
        # Override this for custom optimization objectives
        pass  # Placeholder method for derived classes to implement

    def _update_best_if_better(self, ic: float, obj: float) -> bool:
        """Update the best objective if the current one is better."""
        if obj <= self.best_obj:  # Check if the new objective is worse than or equal to the best
            return False  # No update occurs
        self.best_obj = obj  # Update best objective value
        self.best_ic = ic  # Update best IC value
        return True  # Indicate that an update occurred

    # --- Optimization (Abstract) ---
    @abstractmethod
    def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:
        """Optimize the weights of the linear model and return the new weights as a numpy array."""
        pass  # Abstract method to be implemented in derived classes

    # --- Evaluation ---
    def test_composite(self, calculator: TensorAlphaCalculator) -> Tuple[float, float]:
        """Test the composite using the provided calculator."""
        weights = [float(weight) for weight in self.weights]
        # Utilize calculator to evaluate composite
        return calculator.calc_pool_all(self._exprs[:self.size], weights)

    def evaluate_composite(self) -> float:
        """Evaluate the IC of the composite."""
        if self.size == 0:  # Handle case where there are no expressions
            return 0.0  # Return zero if the pool is empty
        # Evaluate the IC of the current composite
        weights = [float(weight) for weight in self.weights]
        return self.calculator.calc_pool_IC(self._exprs[:self.size], weights)

    # --- Internal Methods ---
    def _calc_ics(
        self,
        expr: Operand,
        # Threshold for mutual IC values
        ic_mut_threshold: Optional[float] = None
    ) -> Tuple[float, Optional[List[float]]]:
        """Calculate the single IC and mutual ICs for a given formula."""
        single_ic = self.calculator.calc_single_IC(
            expr)  # Calculate the single IC for the expression
        if not self._under_thres_alpha and single_ic < self._ic_lower_bound:  # Check IC lower bound
            return single_ic, None  # Return single IC if it fails threshold

        mutual_ics = []  # Initialize list for mutual IC values
        for i in range(self.size):  # Calculate mutual IC for each expression in the pool
            mutual_ic = self.calculator.calc_mutual_IC(expr, self._exprs[i])
            if ic_mut_threshold is not None and mutual_ic > ic_mut_threshold:
                return single_ic, None  # Return if mutual IC exceeds threshold
            mutual_ics.append(mutual_ic)  # Append valid mutual IC value

        return single_ic, mutual_ics  # Return both single IC and mutual ICs

    def _add_factor(self, expr: Operand, ic: float, ic_mut: List[float]) -> None:
        """Add a new factor (formula) to the pool."""
        if self._under_thres_alpha and self.size == 1:
            self._pop()  # Remove if threshold is under and pool has only one expression

        n = self.size  # Current size (number of expressions)
        self._exprs.append(expr)  # Add the new expression
        self._single_ics[n] = ic  # Store single IC for new expression

        for i in range(n):  # Update mutual ICs in the matrix
            self._mutual_ics[i][n] = self._mutual_ics[n][i] = ic_mut[i]

        # Store extra information for the new expression
        self._extra_info[n] = self._get_extra_info(expr)
        # Initialize weight for the new expression
        self._weights[n] = max(ic, 0.01) if n == 0 else self.weights.mean()
        self.size += 1  # Increment the size of the pool

    @abstractmethod
    def _get_extra_info(self, expr: Operand) -> Any: ...

    def _pop(self, index_hint: Optional[int] = None) -> None:
        """Remove an expression from the pool based on the given index."""
        if self.size <= self.capacity:  # Do not remove if within capacity
            return

        index = int(np.argmin(np.abs(self.weights))
                    # Identify the index of the expression to remove
                    ) if index_hint is None else index_hint
        # Swap the expression to remove with the last one
        self._swap_idx(index, self.capacity)
        self.size = self.capacity  # Update the size of the pool

    def _swap_idx(self, i: int, j: int) -> None:
        """Swap the indices of two formulas in the pool."""
        if i == j:  # No need to swap if indices are the same
            return

        def swap_in_list(lst, i: int, j: int) -> None:
            """Helper to swap two elements in a list."""
            lst[i], lst[j] = lst[j], lst[i]  # Swap elements

        swap_in_list(self._exprs, i, j)  # Swap expressions
        swap_in_list(self._single_ics, i, j)  # Swap single ICs
        # Swap mutual ICs for the two expressions
        self._mutual_ics[:, [i, j]] = self._mutual_ics[:, [j, i]]
        self._mutual_ics[[i, j], :] = self._mutual_ics[[j, i], :]
        swap_in_list(self._weights, i, j)  # Swap weights
        swap_in_list(self._extra_info, i, j)  # Swap extra information

    @property
    def _under_thres_alpha(self) -> bool:
        """Check if the pool contains an alpha below the lower threshold."""
        if self._ic_lower_bound is None or self.size > 1:  # If no lower bound or multiple expressions don't consider threshold
            return False
        # Check for threshold condition
        return self.size == 0 or abs(self._single_ics[0]) < self._ic_lower_bound

    # --- Utility ---
    def most_significant_indices(self, k: int) -> List[int]:
        """Return the indices of the K most significant weights."""
        if self.size == 0:  # Handle empty pool case
            return []
        # Rank the weights in descending order
        ranks = (-np.abs(self.weights)).argsort().argsort()
        # Return indices of the most significant weights
        return [i for i in range(self.size) if ranks[i] < k]


class MseAlphaPool(LinearAlphaPool):
    def __init__(
        self,
        capacity: int,
        calculator: TensorAlphaCalculator,
        l1_alpha: float = 5e-3,  # L1 regularization parameter for weight optimization
        # Device for tensor operations (CPU by default)
        device: torch.device = torch.device("cpu")
    ):
        # Initialize the base class with provided parameters
        super().__init__(capacity, calculator, device)
        self.calculator: TensorAlphaCalculator  # Instance of TensorAlphaCalculator
        self._l1_alpha = l1_alpha  # Set the L1 regularization strength

    def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:
        alpha = self._l1_alpha
        # If L1 regularization is effectively disabled (alpha close to 0), use least squares optimization
        if math.isclose(alpha, 0.):
            return self._optimize_lstsq()

        # Prepare tensors for responsible inputs
        ics_ret = torch.tensor(
            self._single_ics[:self.size], device=self.device)
        ics_mut = torch.tensor(
            self._mutual_ics[:self.size, :self.size], device=self.device)
        # Initialize weights for optimization
        weights = torch.tensor(
            self.weights, device=self.device, requires_grad=True)
        # Use Adam optimizer for weight optimization
        optim = torch.optim.Adam([weights], lr=lr)

        # Initialize tracking variables for optimization process
        loss_ic_min = float("inf")  # Best loss initialized to infinity
        best_weights = weights  # To store the best weights found during optimization
        tolerance_count = 0  # Counter for tolerance checking

        # Optimization loop, iterating until max steps or tolerance is reached
        for step in count():
            # Calculate the intermediate cash flows
            ret_ic_sum = (weights * ics_ret).sum()
            # Mutual information calculation
            mut_ic_sum = (torch.outer(weights, weights) * ics_mut).sum()
            # Loss function based on info coefficients
            loss_ic = mut_ic_sum - 2 * ret_ic_sum + 1
            loss_ic_curr = loss_ic.item()  # Current loss value

            # L1 regularization term calculation
            loss_l1 = torch.norm(weights, p=1)
            loss = loss_ic + alpha * loss_l1  # Total loss includes L1 regularization

            optim.zero_grad()  # Zero gradients
            loss.backward()  # Compute gradients
            optim.step()  # Update weights

            # Check if the loss has improved to reset tolerance count
            if loss_ic_min - loss_ic_curr > 1e-6:
                tolerance_count = 0
            else:
                tolerance_count += 1  # Increment tolerance counter if loss has not improved

            # Update the best weights found so far
            if loss_ic_curr < loss_ic_min:
                best_weights = weights
                loss_ic_min = loss_ic_curr

            # Break if tolerance or maximum steps reached
            if tolerance_count >= tolerance or step >= max_steps:
                break

        # Return optimized weights as a numpy array
        return best_weights.cpu().detach().numpy()

    def _optimize_lstsq(self) -> np.ndarray:
        """Perform least-squares optimization to find weights."""
        try:
            # Solve the least squares problem
            return np.linalg.lstsq(self._mutual_ics[:self.size, :self.size], self._single_ics[:self.size], rcond=None)[0]
        except (np.linalg.LinAlgError, ValueError):
            # Return current weights if there's a numerical issue
            return self.weights

    def _get_extra_info(self, expr: Operand) -> Any:
        """Evaluate additional information based on the given expression."""
        return self.calculator.evaluate_alpha(expr)


class MeanStdAlphaPool(LinearAlphaPool):
    def __init__(
        self,
        capacity: int,
        calculator: TensorAlphaCalculator,
        l1_alpha: float = 5e-3,  # L1 regularization coefficient
        # Lower confidence bound beta, if optimizing LCB
        lcb_beta: Optional[float] = None,
        device: torch.device = torch.device(
            "cpu")  # Device for tensor operations
    ):
        """
        l1_alpha: the L1 regularization coefficient.
        lcb_beta: for optimizing the lower-confidence-bound: LCB = mean - beta * std. 
                  If None, optimize ICIR (mean / std) instead.
        """
        super().__init__(capacity, calculator, device)  # Initialize the base class
        self.calculator: TensorAlphaCalculator  # Link to the calculator
        self._l1_alpha = l1_alpha  # Set L1 regularization coefficient
        self._lcb_beta = lcb_beta  # Optional beta value for LCB optimization

    def _get_extra_info(self, expr: Operand) -> Any:
        """Retrieve extra information based on the given expression."""
        return self.calculator.evaluate_alpha(expr)

    def _calc_main_objective(self) -> float:
        """Calculate the main objective function value to be maximized."""
        # Stack extra info into tensor
        alpha_values = torch.stack(
            self._extra_info[:self.size])  # type: ignore
        # Convert weights to tensor
        weights = torch.tensor(self.weights, device=self.device)
        # Compute objective
        return self._calc_obj_impl(alpha_values, weights).item()

    def _calc_obj_impl(self, alpha_values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Implementation of the objective function:
        This can either be the ICIR or LCB based on the presence of lcb_beta.
        """
        target_value = self.calculator.target  # Retrieve the target values
        # Calculate weighted alpha values
        weighted = (weights[:, None, None] * alpha_values).sum(dim=0)
        # Calculate Pearson correlation coefficients
        ics = batch_pearsonr(weighted, target_value)
        # Compute mean and standard deviation of the ICs
        mean, std = ics.mean(), ics.std()
        if self._lcb_beta is not None:
            return mean - self._lcb_beta * std  # Return LCB
        else:
            return mean / std  # Return ICIR

    def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:
        """
        Perform optimization over the weights 
        to maximize either ICIR or LCB based on specified parameters.
        """
        # Prepare alpha values tensor
        alpha_values = torch.stack(
            self._extra_info[:self.size])  # type: ignore
        weights = torch.tensor(
            # Initialize weights for optimization
            self.weights, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([weights], lr=lr)  # Use Adam optimizer

        min_loss = float("inf")  # Keep track of the minimum loss
        best_weights = weights  # Initialize best weights tracking
        tol_count = 0  # Tolerance counter

        for step in count():
            # Calculate the objective
            obj = self._calc_obj_impl(alpha_values, weights)
            # Compute L1 regularization term
            loss_l1 = torch.norm(weights, p=1)
            # Formulate the total loss (maximize objective)
            loss = self._l1_alpha * loss_l1 - obj
            curr_loss = loss.item()  # Get the current loss value

            optimizer.zero_grad()  # Zero gradients before backpropagation
            loss.backward()  # Backpropagate to compute gradients
            optimizer.step()  # Update weights

            # Check if the current loss has improved
            if min_loss - curr_loss > 1e-6:
                tol_count = 0  # Reset tolerance count
            else:
                tol_count += 1  # Increment if no improvement

            # Update best weights if current loss is less than min loss
            if curr_loss < min_loss:
                best_weights = weights
                min_loss = curr_loss

            # Break loop if either tolerance or maximum steps is reached
            if tol_count >= tolerance or step >= max_steps:
                break

        # Return best weights as a numpy array
        return best_weights.cpu().detach().numpy()
