import time
import numpy as np
import torch
import ray

from Mining.Config import *

from Mining.RL.Agent.Core.Game import Game  # M: Environment interface
# Unified model (R, D, O, V, π)
from Mining.RL.Agent.Core.Network import Network
from Mining.RL.Agent.Core.Trajectories import Trajectories
from Mining.RL.Agent.Core.MCTS import MCTS


@ray.remote
class PlayLoop:
    """
    SelfPlay agent for MuZero training.

    This agent continuously interacts with the environment (M), plays episodes,
    and saves the resulting trajectories to the replay buffer. The self-play
    process follows the MuZero pseudocode by:
      - Resetting the environment to obtain the initial state s₀.
      - Using the unified model to generate latent representations.
      - Running MCTS at each decision point to produce a policy (p_MCTS) and value (v_MCTS).
      - Sampling an action from p_MCTS and interacting with the environment.
      - Storing (s, a, r, p_MCTS, v_MCTS) in the trajectory.
    """

    def __init__(self, initial_checkpoint, environment: Game):
        # Save the environment (M) for interactions.
        self.environment = environment

        # Fix random seeds for reproducibility.
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the unified model from the checkpoint.
        self.model = Network()
        initial_weights = initial_checkpoint.get_info.remote("weights")
        self.model.set_weights(initial_weights)
        device = torch.device("cuda" if selfplay_on_gpu else "cpu")
        self.model.to(device)
        self.model.eval()

    def play_loop(self, checkpoint, replay_buffer):
        """
        Main self-play loop that continuously:
          - Updates model weights from the checkpoint.
          - Plays episodes (each episode is one trajectory).
          - Saves trajectories to the replay buffer.
          - Enforces the desired training-to-play ratio.

        The loop terminates when the number of trained steps exceeds max_training_steps
        or if a termination flag is set in the checkpoint.
        """
        while True:
            num_trained_steps = int(
                ray.get(checkpoint.get_info.remote("num_trained_steps")))
            terminate = ray.get(checkpoint.get_info.remote("terminate"))

            if num_trained_steps > max_training_steps or terminate:
                break

            # Update model weights from the checkpoint.
            updated_weights = ray.get(checkpoint.get_info.remote("weights"))
            self.model.set_weights(updated_weights)

            # Compute dynamic temperature based on training progress.
            temperature = temperature_func(self, num_trained_steps)
            
            # Play one full episode to generate a trajectory.
            trajectory = self.play_step(
                temperature=temperature,
                render=False,
                opponent="self",
                muzero_player=0  # Index of the player controlled by MuZero.
            )

            # Save the generated trajectory to the replay buffer.
            replay_buffer.save_game.remote(trajectory, checkpoint)

            # Ensure the training-to-play ratio is maintained.
            self._enforce_training_ratio(checkpoint)

        self._close_environment()

    def play_step(self, temperature, render, opponent, muzero_player):
        """
        Plays a single episode using MCTS for action selection at every move.

        This corresponds to one complete trajectory of interaction:
          - Reset environment to obtain initial state s₀.
          - Use the unified model and MCTS to obtain p_MCTS and v_MCTS.
          - Sample an action from p_MCTS, execute it in M, and record the result.

        Args:
            temperature (float): Exploration parameter to adjust action sampling.
            render (bool): If True, render environment state and info.
            opponent (str): Opponent strategy ("self", "human", "expert", "random").
            muzero_player (int): Player index controlled by MuZero.
        Returns:
            Trajectories: Object containing the episode’s states, actions, rewards, and MCTS statistics.
        """
        trajectory = Trajectories()
        # Reset environment and get the initial state s₀.
        # because observed states maybe incomplete, we use observation rather than state here
        observation = self.environment.reset()

        # Initialize trajectory with s₀; dummy action and reward for alignment.
        trajectory.observations.append(observation)
        trajectory.actions.append(0)    # Dummy action.
        trajectory.rewards.append(0)    # Dummy reward.
        trajectory.players.append(self.environment.player_id())

        done = False

        if render:
            self.environment.render()

        with torch.no_grad():
            while not done:
                # Validate the observation dimensions.
                assert observation.ndim == 3, f"Expected 3-dimensional state but got {observation.ndim} dimensions, shape: {observation.shape}"
                assert observation.shape == observation_shape, f"Expected state shape {observation_shape} but got {observation.shape}"

                # Retrieve stacked observations from trajectory history.
                # This stacks previous frames if needed (useful for partial observability).
                stacked_observations = trajectory.get_stacked_observations(
                    index=-1, num_stacked=0, action_space_size=len(action_space)
                )

                # Decide which agent selects the action.
                if opponent == "self" or muzero_player == self.environment.player_id():
                    # Run MCTS with the unified model.
                    root, mcts_info = MCTS().run(
                        self.model,
                        stacked_observations,
                        self.environment.legal_actions(),
                        self.environment.player_id(),
                        add_exploration_noise=True,
                    )
                    # Sample action from the MCTS visit counts.
                    action = self.select_action_from_node(
                        root, temperature)
                    if render:
                        print(f"Tree depth: {mcts_info['max_tree_depth']}")
                        print(
                            f"Root value for player {self.environment.player_id()}: {root.value():.2f}")
                else:
                    # For non-self opponents, choose an action based on opponent type.
                    action, root = self._select_opponent_action(
                        opponent, stacked_observations)

                # Execute the chosen action in the environment M.
                observation, reward, done = self.environment.step(action)
                if render:
                    action_str = self.environment.action_to_string(action)
                    print(f"Played action: {action_str}")
                    self.environment.render()

                # Record MCTS statistics and update trajectory.
                trajectory.store_search_statistics(
                    root, action_space)
                trajectory.actions.append(action)
                trajectory.observations.append(observation)
                trajectory.rewards.append(reward)
                trajectory.players.append(self.environment.player_id())

        return trajectory

    def _enforce_training_ratio(self, checkpoint):
        """
        Wait until the ratio of trained steps to played steps is below the threshold.

        This ensures that the model is sufficiently trained before generating more self-play data.
        """
        while True:
            num_trained_steps = int(
                ray.get(checkpoint.get_info.remote("num_trained_steps")))
            num_played_steps = max(
                1, int(ray.get(checkpoint.get_info.remote("num_played_steps"))))
            if num_trained_steps / num_played_steps < ratio_train_play:
                break
            time.sleep(0.5)

    def _close_environment(self):
        """Close the environment M."""
        self.environment.close()

    def _select_opponent_action(self, opponent, stacked_s):
        """
        Select an action for a non-self opponent.

        Depending on the opponent type, this method:
          - Uses MCTS and human input (for "human" opponents).
          - Uses an expert agent (for "expert" opponents).
          - Randomly selects an action (for "random" opponents).

        Args:
            opponent (str): Opponent type ("human", "expert", or "random").
            stacked_s: Stacked state observations.
        Returns:
            (action, root): Tuple with the chosen action and the associated MCTS node (if applicable).
        """
        if opponent == "human":
            root, mcts_info = MCTS().run(
                self.model,
                stacked_s,
                self.environment.legal_actions(),
                self.environment.player_id(),
                add_exploration_noise=True,
            )
            print(f"Tree depth: {mcts_info['max_tree_depth']}")
            print(
                f"Root value for player {self.environment.player_id()}: {root.value():.2f}")
            suggestion = self.select_action_from_node(root, temperature=0)
            print(
                f"Player {self.environment.player_id()} turn. MuZero suggests {self.environment.action_to_string(suggestion)}")
            return self.environment.human_to_action(), root
        elif opponent == "expert":
            return self.environment.expert_agent(), None
        elif opponent == "random":
            legal_actions = self.environment.legal_actions()
            assert legal_actions, f"Legal actions should not be empty. Got {legal_actions}."
            assert set(legal_actions).issubset(set(action_space)
                                               ), "Legal actions must be a subset of the action space."
            return np.random.choice(legal_actions), None
        else:
            raise NotImplementedError(
                'Invalid opponent: must be "self", "human", "expert", or "random".')

    @staticmethod
    def select_action_from_node(node, temperature):
        """
        Select an action based on the visit count distribution of a node's children.

        The temperature parameter adjusts the exploration: 0 for deterministic selection
        (choosing the most visited child), and higher values for more uniform random sampling.

        Args:
            node: MCTS search tree node.
            temperature (float): Exploration parameter.
        Returns:
            The selected action.
        """
        children = node.children
        actions = list(children.keys())
        visit_counts = np.array(
            [child.visit_count for child in children.values()], dtype="int32")

        if temperature == 0:
            return actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            return np.random.choice(actions)
        else:
            # Adjust visit counts according to temperature and compute probabilities.
            adjusted_counts = visit_counts ** (1 / temperature)
            probabilities = adjusted_counts / np.sum(adjusted_counts)
            return np.random.choice(actions, p=probabilities)
