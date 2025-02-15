import heapq
import jax.numpy as jnp
from copy import deepcopy
from typing import Sequence, NamedTuple, Dict

class ActionHistory(object):
    """Simple history container used inside the search.

    Only used to keep track of the actions executed.
    """

    def __init__(self, history: Sequence[int], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: int):
        self.history.append(action)

    def last_action(self) -> int:
        return self.history[-1]

    def action_space(self) -> Sequence[int]:
        return [i for i in range(self.action_space_size)]


class Target(NamedTuple):
    performance_value: float
    heuristic_reward_value: float
    policy: Sequence[int]
    bootstrap_discount: float


class Sample(NamedTuple):
    observation: Dict[str, jnp.ndarray]
    bootstrap_observation: Dict[str, jnp.ndarray]
    target: Target


class Node(object):
    """MCTS node."""

    def __init__(self, k:int, prior: float):
        self.visit_count = 0
        self.children = {}
        self.prior = prior
        self.reward = None
        self.k=k
        self.hidden_state = None
        self.heap = []
        heapq.heapify(self.heap)

    def clone(self):
        return deepcopy(self)

    def expanded(self) -> bool:
        return self.children is not None and bool(self.children)

    def add_value(self, value):
        # add value to the node
        pass

    
    def value(self) -> float:
        # calculate the value of the node
        return 0
    
class NetworkOutput(NamedTuple):
  value: float
  metric_logits: jnp.ndarray
  heuristic_reward_logits: jnp.ndarray
  policy_logits: Dict[int, float]
