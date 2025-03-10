import ray
import copy
import numpy as np
from numpy.typing import NDArray

from typing import List, Dict, Tuple, Any

# from Mining.Config import *
from dotmap import DotMap

from Mining.RL.Agent.Core.Trajectory import Trajectory


@ray.remote
class ReplayBuffer:
    """
    ReplayBuffer runs as a dedicated thread to store game traj and generate training batches.

    It is designed to support prioritized experience replay (PER) and is aligned with the MuZero training loop.
    """

    def __init__(self, config, initial_checkpoint, initial_buffer: Dict[int, Trajectory]):
        self.config = DotMap(config)

        # Deep copy to ensure an independent buffer instance
        self.buffer = copy.deepcopy(initial_buffer)

        # Retrieve initial game and step counters from the checkpoint
        self.num_played_games: int = ray.get(initial_checkpoint.get_info.remote(
            "num_played_games"))
        self.num_played_steps: int = ray.get(initial_checkpoint.get_info.remote(
            "num_played_steps"))

        # Total number of samples across all stored traj
        self.total_samples = sum(
            [len(traj.values) for traj in self.buffer.values()]
        )
        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples "
                f"({self.num_played_games} games).\n"
            )
        # Fix random generator seed for reproducibility
        np.random.seed(self.config.seed)
        print('ReplayBuffer inited...')

    def save_game(self, traj: Trajectory, checkpoint=None):
        """
        Save a new game trajectory to the buffer and update counters.
        Optionally, update checkpoint information.
        """
        # If prioritized experience replay is enabled, compute or copy priorities
        if self.config.PER:
            traj.update_value_targets_and_priorities()

        # Add the new game trajectory to the buffer
        self.buffer[self.num_played_games] = traj

        # Update counters
        self.num_played_games += 1
        self.num_played_steps += len(traj.values)
        self.total_samples += len(traj.values)

        # If buffer exceeds maximum size, remove the oldest game
        if len(self.buffer) > self.config.replay_buffer_size:
            del_game_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_game_id].values)
            del self.buffer[del_game_id]

        # Update checkpoint info if provided
        if checkpoint:
            checkpoint.set_info.remote(
                "num_played_games", self.num_played_games)
            checkpoint.set_info.remote(
                "num_played_steps", self.num_played_steps)

    def get_batch(self):
        """
        Sample a batch of training data from the replay buffer.
        Returns:
            - index_batch: List of (game_id, position) tuples.
            - Data tuple containing observations, actions, target values, rewards,
              target policies, weights (if PER), and gradient scales.
        """
        # Initialize batch lists
        index_batch: List[List[int]] = []
        observation_batch: List[NDArray] = []
        policy_batch: List[List[List[float]]] = []
        value_batch: List[List[float]] = []
        action_batch: List[List[int]] = []
        reward_batch: List[List[float]] = []
        weight_batch: List[float] = []
        gradient_scale_batch: List[List[float]] = []

        # Sample a batch of games from the replay buffer
        for game_id, game_traj, game_prob in self.sample_n_games(self.config.batch_size):
            # Sample a position within the selected game trajectory
            # This is because the networks need to perform well for any intermediate steps
            pos_index, pos_prob = self.sample_position(game_traj)

            # Compute training targets using unroll steps (as in pseudocode Step 5.4)
            observations, policies, values, actions, rewards = \
                self.make_target(game_traj, pos_index)

            # Accumulate batch data
            index_batch.append([game_id, pos_index])
            observation_batch.append(observations)
            policy_batch.append(policies)
            value_batch.append(values)
            action_batch.append(actions)
            reward_batch.append(rewards)
            # Gradient scaling based on how many unroll steps are valid
            # NOTE: when the game is about to end, you should focus more on using deadly combos
            #       thus should use larger gradient scaling to efficiently learn those combos
            unfinished_steps = min(self.config.future_steps,
                                   (len(game_traj.actions) - pos_index))
            gradient_scale_batch.append([1/unfinished_steps] * len(actions))
            if self.config.PER:
                # NOTE: less probability, more weight
                prob_game_pos = (self.total_samples * game_prob * pos_prob)
                weight_batch.append(1 / prob_game_pos)

        if self.config.PER:
            weight_batch = [w/max(weight_batch) for w in weight_batch]

        # Return the batch data, formatted as described in the pseudocode comments.
        # observation_batch: batch, (stacked_observations*2+1)*channels, height, width
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # value_batch: batch, num_unroll_steps+1
        # action_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return index_batch, (
            observation_batch,
            policy_batch,
            value_batch,
            action_batch,
            reward_batch,
            weight_batch,
            gradient_scale_batch,
        )

    def sample_n_games(self, n_games: int, force_uniform=False):
        """
        Sample n games from the buffer.
        Returns a list of tuples (game_id, trajectory, game_probability).
        """
        if self.config.PER and not force_uniform:
            ids: List[int] = []
            probs: List[np.float32] = []
            for traj_id, traj in self.buffer.items():
                ids.append(traj_id)
                probs.append(traj.traj_priority)
            game_probs: NDArray[np.float32] = np.array(probs, dtype="float32")
            game_probs /= np.sum(game_probs)
            game_prob_dict = {gid: float(gprob) for gid, gprob in
                              zip(ids, probs)}
            selected_game_ids: List[int] = np.random.choice(
                ids, n_games, p=game_probs).tolist()
        else:
            selected_game_ids: List[int] = np.random.choice(
                list(self.buffer.keys()), n_games).tolist()
            game_prob_dict = {}

        return [(gid, self.buffer[gid], game_prob_dict[gid])
                for gid in selected_game_ids]

    def sample_position(self, traj: Trajectory, force_uniform=False):
        """
        Sample a position (time step) within a game trajectory.
        Sampling can be prioritized or uniform.
        """
        if self.config.PER and not force_uniform:
            pos_probs = traj.priorities / np.sum(traj.priorities)
            selected_index = np.random.choice(len(pos_probs), p=pos_probs)
            selected_prob = float(pos_probs[selected_index])
        else:
            selected_index = np.random.choice(len(traj.values))
            selected_prob = 0.0
        return selected_index, selected_prob

    def update_trajectories(self, game_id, updated_traj):
        """
        Update a trajectory in the buffer with new data.
        """
        # Ensure that the game exists (it might have been removed)
        if next(iter(self.buffer)) <= game_id:
            if self.config.PER:
                updated_traj.priorities = np.copy(updated_traj.priorities)
            self.buffer[game_id] = updated_traj

    def update_priorities(self, new_priorities, index_info):
        """
        Update the priorities for positions in traj after training.
        See Distributed Prioritized Experience Replay (https://arxiv.org/abs/1803.00933).

        index_info: list of (game_id, game_position) tuples.
        new_priorities: numpy array with updated priorities.
        """
        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]
            # Verify that the game is still in the buffer
            if next(iter(self.buffer)) <= game_id:
                priority = new_priorities[i, :]
                start_index = game_pos
                end_index = min(
                    game_pos +
                    len(priority), len(self.buffer[game_id].priorities)
                )
                self.buffer[game_id].priorities[start_index:end_index] = \
                    priority[: end_index - start_index]
                self.buffer[game_id].traj_priority = \
                    np.max(self.buffer[game_id].priorities)

    def make_target(self, traj: Trajectory, pos_idx: int):
        """
        Generate targets for each unroll step starting from a given state index.
        Returns:
            observations, policies, values, actions, rewards
        These correspond to the targets used in training the models (reward, value, and policy).
        """
        observations: NDArray = np.array(0.0)
        policies: List[List[float]] = []
        values: List[float] = []
        actions: List[int] = []
        rewards: List[float] = []

        # representation logits (thus its target) is only used once
        observations = traj.get_stacked_observations(
            pos_idx, self.config.stacked_observations, len(self.config.action_space))
        
        # Loop over unroll steps (including the initial state)
        for current_index in range(pos_idx, pos_idx + self.config.future_steps + 1):
            if current_index < len(traj.values):
                policies.append(traj.policies[current_index])
                values.append(traj.target_values[current_index])
                actions.append(traj.actions[current_index])
                rewards.append(traj.rewards[current_index])
            # elif current_index == len(traj.values):  # End of game
            #     # Use a uniform policy distribution
            #     policies.append(
            #         [1 / len(traj.policies[0])
            #          for _ in range(len(traj.policies[0]))]
            #     )
            #     values.append(0.0)
            #     actions.append(traj.actions[current_index-1])
            #     rewards.append(traj.rewards[current_index-1])
            else:
                # For positions past the end of the game, treat as absorbing state
                sup_size = len(traj.policies[0])
                policies.append([1 / sup_size for _ in range(sup_size)])
                values.append(0.0)
                # Sample a random action from the action space
                actions.append(
                    np.random.choice(self.config.action_space))
                rewards.append(0.0)

        return observations, policies, values, actions, rewards
