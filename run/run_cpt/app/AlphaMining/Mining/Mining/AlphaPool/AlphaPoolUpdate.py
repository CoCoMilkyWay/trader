from abc import ABCMeta, abstractmethod
from typing import List, Optional, cast
from dataclasses import dataclass, MISSING
from Mining.Expression.Operand import Operand


@dataclass
class PoolUpdate(metaclass=ABCMeta):
    """Abstract base class for pool updates."""

    @property
    @abstractmethod
    def old_pool(self) -> List[Operand]:
        """Return the old pool of operands."""
        ...

    @property
    @abstractmethod
    def new_pool(self) -> List[Operand]:
        """Return the new pool of operands."""
        ...

    @property
    @abstractmethod
    def old_pool_ic(self) -> Optional[float]:
        """Return the information coefficient (IC) of the old pool."""
        ...

    @property
    @abstractmethod
    def new_pool_ic(self) -> float:
        """Return the information coefficient (IC) of the new pool."""
        ...

    @property
    def ic_increment(self) -> float:
        """Calculate the increment in information coefficient (IC)."""
        return self.new_pool_ic - (self.old_pool_ic or 0.)

    @abstractmethod
    def describe(self) -> str:
        """Provide a description of the update."""
        ...

    def describe_verbose(self) -> str:
        """Provide a verbose description of the update."""
        return self.describe()

    def _describe_ic_diff(self) -> str:
        """Describe the difference in IC between the old and new pools."""
        return (
            f"{self.old_pool_ic:.4f} -> {self.new_pool_ic:.4f} "
            f"(increment of {self.ic_increment:.4f})"
        )

    def _describe_pool(self, title: str, pool: List[Operand]) -> str:
        """Describe a list of operands in the pool."""
        list_exprs = "\n".join([f"  {expr}" for expr in pool])
        return f"{title}\n{list_exprs}"


class _PoolUpdateStub:
    """Stub for pool updates, allowing default values to be cast."""

    old_pool: List[Operand] = cast(List[Operand], MISSING)
    new_pool: List[Operand] = cast(List[Operand], MISSING)
    old_pool_ic: Optional[float] = cast(Optional[float], MISSING)
    new_pool_ic: float = cast(float, MISSING)


@dataclass
class SetPool(_PoolUpdateStub, PoolUpdate):
    """Class representing the action of setting a new pool."""

    # Properties defining the state of the pool
    old_pool: List[Operand]
    new_pool: List[Operand]
    old_pool_ic: Optional[float]
    new_pool_ic: float

    def describe(self) -> str:
        """Describe the update by showing the new pool and its IC."""
        pool = self._describe_pool("Alpha pool:", self.new_pool)
        return f"{pool}\nIC of the combination: {self.new_pool_ic:.4f}"

    def describe_verbose(self) -> str:
        """Provide a detailed description including old and new pools."""
        if len(self.old_pool) == 0:
            return self.describe()
        old_pool = self._describe_pool("Old alpha pool:", self.old_pool)
        new_pool = self._describe_pool("New alpha pool:", self.new_pool)
        perf = f"IC of the pools: {self._describe_ic_diff()})"
        return f"{old_pool}\n{new_pool}\n{perf}"


@dataclass
class AddRemoveAlphas(_PoolUpdateStub, PoolUpdate):  # Method Resolution Order (MRO)
    """Class representing the action of adding and removing alphas in a pool."""

    # Newly added and removed indices
    added_exprs: List[Operand]
    removed_idx: List[int]
    old_pool: List[Operand]
    old_pool_ic: float
    new_pool_ic: float

    @property
    def new_pool(self) -> List[Operand]:
        """Calculate the new pool based on removed indices and added expressions."""
        remain = [True] * len(self.old_pool)
        for i in self.removed_idx:
            remain[i] = False
        return [expr for i, expr in enumerate(self.old_pool) if remain[i]] + self.added_exprs

    def describe(self) -> str:
        """Describe the update by showing added and removed alphas."""
        def describe_exprs(title: str, exprs: List[Operand]) -> str:
            """Format the description of expressions."""
            if len(exprs) == 0:
                return ""
            if len(exprs) == 1:
                return f"{title}: {exprs[0]}\n"
            exprs_str = "\n".join([f"  {expr}" for expr in exprs])
            return f"{title}s:\n{exprs_str}\n"

        added = describe_exprs("Added alpha", self.added_exprs)
        removed = describe_exprs(
            "Removed alpha", [self.old_pool[i] for i in self.removed_idx])
        perf = f"IC of the combination: {self._describe_ic_diff()}"
        return f"{added}{removed}{perf}"

    def describe_verbose(self) -> str:
        """Provide a detailed description including the old pool."""
        old = self._describe_pool("Old alpha pool:", self.old_pool)
        return f"{old}\n{self.describe()}"
