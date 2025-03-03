import numpy as np

from typing import List, Optional

from Mining.RL.Agent.Core.MCTS import Node


class Trajectories:
    """
    Stores key information from a self-play game.

    This buffer keeps track of observations, actions, rewards, player identities,
    and Monte Carlo Tree Search (MCTS) statistics. These data are later used for
    training the MuZero model.
    """

    def __init__(self):
        # Stores each observation (e.g., game board state) from the self-play game.
        self.observations = []
        # Stores the actions taken by the agent at each step.
        self.actions = []
        # Stores the rewards received after each action.
        self.rewards = []
        # Stores the root value estimates produced by MCTS.
        self.values = []
        # Stores the policy distributions from MCTS search (visit counts normalized).
        self.policies = []

        # For Prioritized Experience Replay (PER) setup.
        self.priorities = None
        self.game_priority = None

        # Misc
        # Stores the player identifier for each move (useful in multi-player games).
        self.players = []
        # Optionally stores reanalysed predicted root values.
        # self.reanalysed_predicted_root_values = None

    def store_search_statistics(self, root: Optional[Node], action_space: List[int]):
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
                total_visits if a in root.children else 0
                for a in action_space
            ]

            # Store the computed policy distribution.
            self.policies.append(policy)
            # Store the value of the root node (e.g., predicted value from the MCTS).
            self.values.append(root.value())
        else:
            # If no root is provided, store None for the root value.
            self.values.append(None)

    def get_stacked_observations(self, index: int, num_stacked: int, action_space_size: int) -> np.ndarray:
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
