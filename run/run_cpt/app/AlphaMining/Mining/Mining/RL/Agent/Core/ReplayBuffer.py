import ray
import copy
import numpy as np

from typing import Dict

from Mining.Config import *

from Mining.RL.Agent.Core.Trajectories import Trajectories

@ray.remote
class ReplayBuffer:
    """
    ReplayBuffer runs as a dedicated thread to store game traj and generate training batches.

    It is designed to support prioritized experience replay (PER) and is aligned with the MuZero training loop.
    """

    def __init__(self, initial_checkpoint, initial_buffer: Dict):
        # Deep copy to ensure an independent buffer instance
        self.buffer = copy.deepcopy(initial_buffer)

        # Retrieve initial game and step counters from the checkpoint
        self.num_played_games = initial_checkpoint.get_info.remote(
            "num_played_games")
        self.num_played_steps = initial_checkpoint.get_info.remote(
            "num_played_steps")

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
        np.random.seed(seed)
        print('ReplayBuffer inited...')

    def save_game(self, traj: Trajectories, checkpoint=None):
        """
        Save a new game trajectory to the buffer and update counters.
        Optionally, update checkpoint information.
        """
        # If prioritized experience replay is enabled, compute or copy priorities
        if PER:
            if traj.priorities is not None:
                # Avoid read-only issues when loading from disk
                traj.priorities = np.copy(  # type: ignore
                    traj.priorities)
            else:
                # Calculate initial priorities using the target value computation (see appendix Training)
                priorities = []
                for i, root_value in enumerate(traj.values):
                    priority = (
                        np.abs(root_value -
                               self.compute_target_value(traj, i))
                        ** PER_alpha
                    )
                    priorities.append(priority)
                traj.priorities = np.array(  # type: ignore
                    priorities, dtype="float32")
                traj.game_priority = np.max(traj.priorities)  # type: ignore

        # Add the new game trajectory to the buffer
        self.buffer[self.num_played_games] = traj

        # Update counters
        self.num_played_games += 1
        self.num_played_steps += len(traj.values)
        self.total_samples += len(traj.values)

        # If buffer exceeds maximum size, remove the oldest game
        if replay_buffer_size < len(self.buffer):
            del_game_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_game_id].values)
            del self.buffer[del_game_id]

        # Update checkpoint info if provided
        if checkpoint:
            checkpoint.set_info.remote(
                "num_played_games", self.num_played_games)
            checkpoint.set_info.remote(
                "num_played_steps", self.num_played_steps)

    def get_buffer(self):
        """
        Return the current replay buffer.
        """
        return self.buffer

    def get_batch(self):
        """
        Sample a batch of training data from the replay buffer.
        Returns:
            - index_batch: List of (game_id, position) tuples.
            - Data tuple containing observations, actions, target values, rewards,
              target policies, weights (if PER), and gradient scales.
        """
        # Initialize batch lists
        index_batch = []
        observation_batch = []
        action_batch = []
        reward_batch = []
        value_batch = []
        policy_batch = []
        gradient_scale_batch = []
        weight_batch = [] if PER else None

        # Sample a batch of games from the replay buffer
        for game_id, game_traj, game_prob in self.sample_n_games(batch_size):
            # Sample a position within the selected game trajectory
            pos_index, pos_prob = self.sample_position(game_traj)

            # Compute training targets using unroll steps (as in pseudocode Step 5.4)
            target_values, target_rewards, target_policies, actions = self.make_target(
                game_traj, pos_index)

            # Accumulate batch data
            index_batch.append([game_id, pos_index])
            observation_batch.append(
                game_traj.get_stacked_observations(
                    pos_index, stacked_observations, len(action_space))
            )
            action_batch.append(actions)
            value_batch.append(target_values)
            reward_batch.append(target_rewards)
            policy_batch.append(target_policies)
            # Gradient scaling based on how many unroll steps are valid
            gradient_scale_batch.append(
                [min(future_steps, len(game_traj.action_history) - pos_index)] * len(actions)
            )
            if PER:
                weight_batch.append(1 / (self.total_samples * game_prob * pos_prob)) # type: ignore

        if PER:
            weight_batch = np.array(weight_batch, dtype="float32") / max(weight_batch) # type: ignore

        # Return the batch data, formatted as described in the pseudocode comments.
        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return index_batch, (
            observation_batch,
            action_batch,
            value_batch,
            reward_batch,
            policy_batch,
            weight_batch,
            gradient_scale_batch,
        )

    def sample_game(self, force_uniform=False):
        """
        Sample a single game trajectory from the buffer.
        Sampling can be prioritized or uniform.
        """
        game_prob = None
        if PER and not force_uniform:
            game_probs = np.array(
                [traj.game_priority for traj in self.buffer.values()],
                dtype="float32",
            )
            game_probs /= np.sum(game_probs)
            game_index = np.random.choice(len(self.buffer), p=game_probs)
            game_prob = game_probs[game_index]
        else:
            game_index = np.random.choice(len(self.buffer))
        # Calculate game_id relative to the full game count
        game_id = self.num_played_games - len(self.buffer) + game_index
        return game_id, self.buffer[game_id], game_prob

    def sample_n_games(self, n_games, force_uniform=False):
        """
        Sample n games from the buffer.
        Returns a list of tuples (game_id, trajectory, game_probability).
        """
        if PER and not force_uniform:
            game_ids = []
            game_probs = []
            for game_id, traj in self.buffer.items():
                game_ids.append(game_id)
                game_probs.append(traj.game_priority)
            game_probs = np.array(game_probs, dtype="float32")
            game_probs /= np.sum(game_probs)
            game_prob_dict = {gid: prob for gid,
                              prob in zip(game_ids, game_probs)}
            selected_game_ids = np.random.choice(
                game_ids, n_games, p=game_probs)
        else:
            selected_game_ids = np.random.choice(
                list(self.buffer.keys()), n_games)
            game_prob_dict = {}

        return [
            (gid, self.buffer[gid], game_prob_dict.get(gid))
            for gid in selected_game_ids
        ]

    def sample_position(self, traj, force_uniform=False):
        """
        Sample a position (time step) within a game trajectory.
        Sampling can be prioritized or uniform.
        """
        position_prob = None
        if PER and not force_uniform:
            pos_probs = traj.priorities / np.sum(traj.priorities)
            pos_index = np.random.choice(len(pos_probs), p=pos_probs)
            position_prob = pos_probs[pos_index]
        else:
            pos_index = np.random.choice(len(traj.values))
        return pos_index, position_prob

    def update_trajectories(self, game_id, updated_traj):
        """
        Update a trajectory in the buffer with new data.
        """
        # Ensure that the game exists (it might have been removed)
        if next(iter(self.buffer)) <= game_id:
            if PER:
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
                self.buffer[game_id].priorities[start_index:
                                                end_index] = priority[: end_index - start_index]
                # Update overall game priority to the maximum priority in the trajectory
                self.buffer[game_id].game_priority = np.max(
                    self.buffer[game_id].priorities)

    def compute_target_value(self, traj, index):
        """
        Compute the target value for a given state in a game trajectory.
        The target value is the discounted future root value (bootstrap) plus the discounted sum of rewards.
        """
        bootstrap_index = index + future_steps
        if bootstrap_index < len(traj.values):
            # Determine the last-step value based on which player's turn it is
            if traj.players[bootstrap_index] == traj.players[index]:
                last_step_value = traj.values[bootstrap_index]
            else:
                last_step_value = -traj.values[bootstrap_index]
            value = last_step_value * future_discount ** future_steps
        else:
            value = 0

        # Sum discounted rewards from index+1 to bootstrap_index (or end)
        for i, reward in enumerate(traj.rewards[index + 1: bootstrap_index + 1]):
            value += (reward if traj.players[index] ==
                      traj.players[index + i] else -reward) * future_discount ** i
        return value

    def make_target(self, traj, pos_idx):
        """
        Generate targets for each unroll step starting from a given state index.
        Returns:
            target_values, target_rewards, target_policies, actions
        These correspond to the targets used in training the models (reward, value, and policy).
        """
        target_values = []
        target_rewards = []
        target_policies = []
        actions = []

        # Loop over unroll steps (including the initial state)
        for current_index in range(pos_idx, pos_idx + future_steps + 1):
            value = self.compute_target_value(traj, current_index)

            if current_index < len(traj.values):
                target_values.append(value)
                target_rewards.append(traj.reward_history[current_index])
                target_policies.append(traj.child_visits[current_index])
                actions.append(traj.action_history[current_index])
            elif current_index == len(traj.values):
                # End of game: provide default targets
                target_values.append(0)
                target_rewards.append(traj.reward_history[current_index])
                # Use a uniform policy distribution
                target_policies.append(
                    [1 / len(traj.child_visits[0])
                     for _ in range(len(traj.child_visits[0]))]
                )
                actions.append(traj.action_history[current_index])
            else:
                # For positions past the end of the game, treat as absorbing state
                target_values.append(0)
                target_rewards.append(0)
                target_policies.append(
                    [1 / len(traj.child_visits[0])
                     for _ in range(len(traj.child_visits[0]))]
                )
                # Sample a random action from the action space
                actions.append(np.random.choice(action_space))

        return target_values, target_rewards, target_policies, actions
