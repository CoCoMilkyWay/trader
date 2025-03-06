import math
import numpy as np
import gymnasium as gym
from typing import List, Dict, Tuple, Optional

from Mining.Config import *
from Mining.Expression.Expression import Expression
from Mining.Expression.Operator import Operator, UnaryOperator, BinaryOperator, TernaryOperator
from Mining.Expression.Operand import Operand
from Mining.Expression.Token import *
from Mining.Expression.Builder import ExpressionBuilder
from Mining.Expression.Parser import ExpressionParser
from Mining.Metrics.Calculator import ExpressionCalculator
from Mining.AlphaPool.AlphaPoolBase import AlphaPoolBase


class TokenGenEnv(gym.Env):
    """Reinforcement Learning environment for generating expressions based on Gymnasium framework."""

    def __init__(
        self,
        expression_builder: ExpressionBuilder,
        expression_parser: ExpressionParser,
        alpha_pool: AlphaPoolBase,
    ):
        super().__init__()
        self.builder = expression_builder
        self.parser = expression_parser
        self.pool = alpha_pool

        self.action_size = SIZE_ACTION
        self.action_space = gym.spaces.Discrete(self.action_size)
        self.observation_space = gym.spaces.Box(
            low=0, high=self.action_size + SIZE_NULL - 1,
            shape=(MAX_EXPR_LENGTH, ),
            dtype=np.uint8,
        )

        # State variables
        self.counter: int = 0
        self.state: np.ndarray = np.zeros(MAX_EXPR_LENGTH, dtype=np.uint8)
        self._tokens: List[Token] = [BEG_TOKEN]

        self.eval_count = 0  # Count of evaluations performed
        self.render_mode: Optional[str] = None
        self.reset()

    def reset(
        self, *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment and return the initial observation and any additional info."""
        self.counter = 0
        self.state.fill(0)  # Initialize state to all zeros
        self._tokens = [BEG_TOKEN]  # Reset tokens to the beginning
        self.builder.reset()  # Reset the ExpressionBuilder
        observation = self.state
        info = self.info = self.builder.get_init_action_masks()  # misc info (e.g. action masks)
        return observation, info  # Return observation with init mask

    def step(self, action_index: int):
        """Apply the action to the environment, update the state, and return the result of the step."""
        # if self.counter == 0:
        #     self.info = self.builder.get_action_masks()
        # mask = self.info['action_masks']
        # true_indices = np.where(mask)[0]
        # if true_indices.size > 0:  # Check if there are any True values
        #     action_index = np.random.choice(true_indices)
        # else:
        #     raise RuntimeError(
        #         f"Garbage Expression generated:{self._tokens}")
        #     return self.state, 0.0, True, False, {}

        action = self.get_action(action_index)

        if isinstance(action, SyntaxToken) and action.syntax == SyntaxType.SEP:
            reward = self._evaluate_expression()
            done = True
        elif len(self._tokens) < MAX_EXPR_LENGTH:
            self._tokens.append(action)
            self.builder.add_token(action)
            done = False
            reward = 0.0
        else:
            done = True
            reward = 0.0
            print(f"Discarding: {self._tokens}")
        # Ensure reward is not NaN
        reward = 0.0 if math.isnan(reward) else reward

        if not done:
            # Update state with action index
            self.state[self.counter] = action_index
            self.counter += 1

        observation = self.state  # Current state
        reward = self.calculate_reward(reward)  # Adjust reward
        terminated = done  # Episode termination flag
        truncated = False  # Not used; can be adjusted as needed
        info = self.info = self.builder.get_action_masks()  # misc info (e.g. action masks)
        # print(self._tokens)
        # print(f"op_uni:{len(info['op'][1])}, op_bin:{len(info['op'][2])}, op_ter:{len(info['op'][3])}, "
        #       f"feature:{info['valid'][1]}, "
        #       f"con_dt:{info['valid'][2]}, con_rt:{info['valid'][3]}, con_os:{info['valid'][4]}, "
        #       f"stop:{info['valid'][5]}"
        # )
        return observation, reward, terminated, truncated, info

    def get_action(self, action: int) -> Token:
        """Convert action index to corresponding Token."""
        return self.convert_action_idx_to_token(action)

    def calculate_reward(self, reward: float) -> float:
        """Calculate the adjusted reward for this step."""
        return reward + REWARD_PER_STEP

    def _evaluate_expression(self) -> float:
        """Evaluate the built expression and return the reward."""
        expression: Operand = self.builder.get_built_expression()
        reward = self.pool.try_new_formula(expression)
        self.eval_count += 1  # Increment evaluation count
        print(reward)
        return reward

    def convert_action_idx_to_token(self, action: int) -> Token:
        """Convert an action index to its corresponding token."""
        if action < 0:
            raise ValueError("Action index must be non-negative.")

        # Determine the type of token based on the action index
        if action < SIZE_OP:
            return OperatorToken(OPERATORS[action])
        action -= SIZE_OP

        if action < SIZE_FEATURE:
            return FeatureToken(FEATURES[action], DIMENSIONS[action])
        action -= SIZE_FEATURE

        if action < SIZE_CONSTANT_TD:
            return ConstantTDToken(CONST_TIMEDELTAS[action])
        action -= SIZE_CONSTANT_TD

        if action < SIZE_CONSTANT_RT:
            return ConstantRTToken(CONST_RATIOS[action])
        action -= SIZE_CONSTANT_RT

        if action < SIZE_CONSTANT_OS:
            return ConstantOSToken(CONST_OSCILLATORS[action])
        action -= SIZE_CONSTANT_OS

        if action == 0:
            return SyntaxToken(SyntaxType.SEP)

        raise AssertionError(
            "Invalid action index: unable to convert to token.")


"""
import gymnasium as gym

class TokenGenWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs
    
    def step(self, action: int):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, info
"""
