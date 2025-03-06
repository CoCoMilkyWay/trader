import numpy as np
from numpy.typing import NDArray

from typing import List, Optional

from Mining.RL.Agent.Core.MCTS import Node


class Trajectory:
    """
    Stores key information from a self-play game.

    This buffer keeps track of observations, actions, rewards, player identities,
    and Monte Carlo Tree Search (MCTS) statistics. These data are later used for
    training the MuZero model.
    """

    def __init__(self, config):
        # Stores the player identifier for each move (useful in multi-player games).
        self.players: List[int] = []
        # Stores each observation (e.g., game board state) from the self-play game.
        self.observations: List[NDArray] = []
        # Stores the policy distributions from MCTS search (visit counts normalized).
        self.policies: List[List[float]] = []
        # Stores the root value estimates produced by MCTS.
        self.values: List[float] = []  # predicted v-value
        # Stores the actions taken by the agent at each step.
        self.actions: List[int] = []
        # Stores the rewards received after each action.
        self.rewards: List[float] = []

        # NOTE: observations/policies/actions/rewards here are all "real"/"sampled" results that can be used as targets
        #       However, for v-value, there are better ways to calculate more accurate value as target

        # For Prioritized Experience Replay (PER) setup.
        # target v-value after MCTS selection
        self.target_values: List[float] = []
        self.priorities: NDArray[np.float32] = np.array(0.0)  # node priorities
        self.traj_priority: np.float32 = np.float32(0.0)  # worst-case(max)

        # configs
        self.future_discount: float = config.future_discount
        self.future_steps: int = config.future_steps
        self.PER_alpha: float = config.PER_alpha

    def get_stacked_observations(self, index: int, num_stacked: int, action_space_size: int) -> NDArray:
        """
        Constructs a stacked observation by combining the current observation with a history
        of past observation-action pairs.

        This method creates a richer input for the model by stacking the current observation
        with several previous observations and their corresponding actions (encoded as channels).
        If a past observation is not available, a zero-filled placeholder is used.

        Parameters:
            index (int): The index of the current observation.
            num_stacked (int): The number of past observation-action pairs to include.
            action_space_size (int): The total number of possible actions (used to scale the action value).

        Returns:
            np.ndarray: The resulting stacked observation array.
        """
        # Ensure the index is within bounds (wrap-around using modulo if needed).
        index = index % len(self.observations)

        # Start with a copy of the current observation.
        stacked_obs = self.observations[index].copy()

        # Loop over the past observations in reverse order, starting from (index - num_stacked) up to (index - 1).
        for past_idx in reversed(range(index - num_stacked, index)):
            if past_idx >= 0:
                # Retrieve the action taken immediately after the past observation.
                # Divide by the action space size to normalize the action value.
                action_value = self.actions[past_idx + 1] / action_space_size

                # Create an "action channel" array that has the same shape as one channel of the observation.
                action_channel = np.ones_like(stacked_obs[0]) * action_value

                # Concatenate the past observation with its corresponding action channel.
                previous_obs = np.concatenate(
                    (self.observations[past_idx], [action_channel]))

            else:
                # If the past index is negative (i.e., not available), use a zero-filled observation.
                zeros_obs = np.zeros_like(self.observations[index])
                zeros_action_channel = np.zeros_like(stacked_obs[0])
                previous_obs = np.concatenate(
                    (zeros_obs, [zeros_action_channel]))

            # Append the constructed past observation-action pair to the current stacked observation.
            stacked_obs = np.concatenate((stacked_obs, previous_obs))

        # -> (obv_t2, act_t2, obv_t1, act_t1, obv_t0) starts and ends with obv
        return stacked_obs

    def update_policy_and_value(self, root: Optional[Node], action_space: List[int]):
        """
        Stores MCTS search statistics from the given root node.
        The method converts the visit counts from each child of the MCTS root node
        into a normalized probability distribution (policy) over the available actions.
        It also stores the root value produced by the search.
        Parameters:
            root: The MCTS root node that contains children with visit counts.
                  If None, no search statistics will be stored.
            action_space: An iterable of all possible actions.
        """
        if root is not None:
            # Calculate the total number of visits across all children.
            total_visits = sum(
                child.visit_count for child in root.children.values())
            # Compute normalized visit counts (policy distribution) for each action in the action space.
            policy = [
                # If the action exists in the root's children, normalize its visit count.
                root.children[a].visit_count /
                total_visits if a in root.children else 0.0
                for a in action_space
            ]
            # Store the computed policy distribution.
            self.policies.append(policy)
            # Store the value of the root node (e.g., predicted value from the MCTS).
            self.values.append(root.value_mean())
        else:
            # If no root is provided, store None for the root value.
            self.policies.append([0.0 for _ in range(len(action_space))])
            self.values.append(0.0)

    def get_target_node_value(self, index: int):
        """
        the v-value in trajectory is purely results from model, it is far from real value, 
        especially when model is not sufficiently trained

        v-value defined here = transformed(discounted) expected cumulative future rewards
        get more accurate v-value from rewards + leaf_point_value from the real future trajectory (bootstrapping)
        """

        # bootstrapping over [index + 1, bootstrap_index + 1]
        bootstrap_index = index + self.future_steps

        current_player = self.players[index]
        value = 0.0

        # Sum discounted rewards from index+1 to bootstrap_index (or end)
        for i, reward in enumerate(self.rewards[index + 1: bootstrap_index + 1]):
            bootstrap_player = self.players[index + 1 + i]
            value += reward * self.future_discount ** i * \
                (1 if bootstrap_player == current_player else -1)

        # use value estimate at the end index (to cover rewards from end index to infinity)
        if bootstrap_index < len(self.values):
            value += reward * self.future_discount ** self.future_steps * \
                (1 if bootstrap_player == current_player else -1)

        return value

    def update_priorities(self):
        if self.priorities is not None:
            # Avoid read-only issues when loading from disk
            self.priorities = np.copy(self.priorities)
        else:
            # Calculate initial priorities using the target value computation (see appendix Training)
            target_values = []
            priorities = []
            for i, pred_value in enumerate(self.values):
                target_value = self.get_target_node_value(i)
                priority = np.abs(pred_value - target_value) \
                    ** self.PER_alpha
                target_values.append(target_value)
                priorities.append(priority)

            self.target_values = target_values
            self.priorities = np.array(priorities, dtype="float32")
            self.traj_priority = np.max(self.priorities)
