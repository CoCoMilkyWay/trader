import ray
import time
import copy
import torch
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple, Dict, List

# from Mining.Config import *
from dotmap import DotMap

from Mining.RL.Agent.Core.Game import AbstractGame  # M: Environment interface
# Unified model (R, D, O, V, π)
from Mining.RL.Agent.Core.Network import Network
from Mining.RL.Agent.Core.Trajectory import Trajectory
from Mining.RL.Agent.Core.MCTS import MCTS, Node


@ray.remote
class PlayLoop:
    """
    SelfPlay agent for MuZero training.

    This agent continuously interacts with the environment (M), plays episodes,
    and saves the resulting Trajectory to the replay buffer. The self-play
    process follows the MuZero pseudocode by:
      - Resetting the environment to obtain the initial state s₀.
      - Using the unified model to generate latent representations.
      - Running MCTS at each decision point to produce a policy (p_MCTS) and value (v_MCTS).
      - Sampling an action from p_MCTS and interacting with the environment.
      - Storing (s, a, r, p_MCTS, v_MCTS) in the trajectory.
    """

    def __init__(self, config, initial_checkpoint, environment: AbstractGame):
        self.config = DotMap(config)

        # Save the environment (M) for interactions.
        self.environment = copy.deepcopy(environment)

        # Fix random seeds for reproducibility.
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the unified model from the checkpoint.
        self.model = Network(self.config)
        initial_weights = ray.get(
            initial_checkpoint.get_info.remote("weights"))
        self.model.set_weights(initial_weights)  # type: ignore
        device = torch.device("cuda" if self.config.selfplay_on_gpu else "cpu")
        self.model.to(device)
        self.model.eval()

        self.mcts_info: Dict[str, Any] = {}

        self.num_trained_steps: int = 0
        self.num_played_steps: int = 0

        print(f"PlayLoop inited on {device}")

    def play_loop(self, checkpoint, replay_buffer):
        """
        Main self-play loop that continuously:
          - Updates model weights from the checkpoint.
          - Plays episodes (each episode is one trajectory).
          - Saves Trajectory to the replay buffer.
          - Enforces the desired training-to-play ratio.

        The loop terminates when the number of trained steps exceeds max_training_steps
        or if a termination flag is set in the checkpoint.
        """
        self.checkpoint = checkpoint

        while True:
            self._update_steps()
            terminate = ray.get(checkpoint.get_info.remote("terminate"))

            if self.num_trained_steps > self.config.max_training_steps or terminate:
                break

            # Update model weights from the checkpoint.
            updated_weights = ray.get(checkpoint.get_info.remote("weights"))
            self.model.set_weights(updated_weights)  # type: ignore

            # Compute smooth decay of dynamic temperature based on training progress.
            temperature = max(self.config.min_temperature, self.config.max_temperature *
                              (self.config.decay_factor ** (self.num_trained_steps / (0.1 * self.config.max_training_steps))))

            # Play one full episode to generate a trajectory.
            trajectory = self.play_step(
                temperature=temperature,
                render=self.config.render,
                player_type="self",
            )

            # Save the generated trajectory to the replay buffer.
            replay_buffer.save_game.remote(trajectory, checkpoint)

            # Ensure the training-to-play ratio is maintained.
            self._enforce_training_ratio(checkpoint)

        self._close_environment()

    def play_step(self, temperature: float, render: bool, player_type: str):
        """
        Plays a single episode using MCTS for action selection at every move.

        This corresponds to one complete trajectory of interaction:
          - Reset environment to obtain initial state s₀.
          - Use the unified model and MCTS to obtain p_MCTS and v_MCTS.
          - Sample an action from p_MCTS, execute it in M, and record the result.

        Args:
            temperature (float): Exploration parameter to adjust action sampling.
            render (bool): If True, render environment state and info.
            player_type (str): player strategy ("self", "human", "expert", "random").
            player (int): Player index controlled by MuZero.
        Returns:
            Trajectory: Object containing the episode’s states, actions, rewards, and MCTS statistics.
        """
        trajectory = Trajectory(self.config)
        # Reset environment and get the initial state s₀.
        # because observed states maybe incomplete, we use observation rather than state here
        observation = self.environment.reset()

        #   player  obv     policy  value   action  rewards
        #   t0      t0      t0      t0      t0      t0
        #   ...     ...     ...     ...     ...     ...

        #   e.g. (obv_t2, act_t1, obv_t1, act_t0, obv_t0) are used to generate representation

        done = False

        if render:
            self.environment.render()

        with torch.no_grad():
            while not done:
                # Validate the observation dimensions.
                assert observation.ndim == 3, f"Expected 3-dimensional state but got {observation.ndim} dimensions, shape: {observation.shape}"
                assert observation.shape == self.config.observation_shape, f"Expected state shape {self.config.observation_shape} but got {observation.shape}"

                trajectory.players.append(self.environment.player_id())
                trajectory.observations.append(observation)

                # Retrieve stacked observations from trajectory history.
                # This stacks previous frames if needed (useful for partial observability and video games).
                stacked_obvs = trajectory.get_stacked_observations(
                    index=-1, num_stacked=self.config.stacked_observations, action_space_size=len(self.config.action_space)
                )

                # Decide which agent selects the action.
                if player_type == "self":
                    action, root = self._select_player_action(
                        stacked_obvs, temperature)
                else:
                    # For non-self opponents, choose an action based on opponent type.
                    action, root = self._select_opponent_action(
                        player_type, stacked_obvs)

                trajectory.update_policy_and_value(
                    root, self.config.action_space)
                trajectory.actions.append(action)

                # Execute the chosen action in the environment M.
                observation, reward, done = self.environment.step(action)

                # Record MCTS statistics and update trajectory.
                trajectory.rewards.append(reward)

                if render:
                    action_str = self.environment.action_to_string(action)
                    self.environment.render()
                    self.print_traj_node_results(trajectory, -1)

        return trajectory

    def print_traj_node_results(self, traj: Trajectory, node: int):
        lr = ray.get(self.checkpoint.get_info.remote("learning_rate"))
        policy_loss = ray.get(self.checkpoint.get_info.remote("policy_loss"))
        value_loss = ray.get(self.checkpoint.get_info.remote("value_loss"))
        reward_loss = ray.get(self.checkpoint.get_info.remote("reward_loss"))

        # state = [f"{l:+.2f}" for l in traj.observations[node].squeeze().tolist()]
        policy = [f"{l:+.2f}" for l in traj.policies[node]]
        print(
            f"init_value:{traj.values[0]:+2.2f}, "
            f"player:{traj.players[node]:1}, "
            # f"state:{state}, "
            f"policy:{policy}, "
            f"value:{traj.values[node]:2.2f}, "
            f"action:{traj.actions[node]:2}, "
            f"rewards:{traj.rewards[node]:1.1f}, "
            f"tree_depth:{self.mcts_info['max_tree_depth']:2}, "
            f"policy_loss:{policy_loss:2.2f}, "
            f"value_loss:{value_loss:2.2f}, "
            f"reward_loss:{reward_loss:2.2f}, "
            f"learn_rate:{lr:0.2f}, "
            f"trained/played:{self.num_trained_steps:5}/{self.num_played_steps:5}, "
        )

    def _select_player_action(self, stacked_obvs: NDArray, temperature: float):
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
        # Run MCTS with the unified model.
        root, self.mcts_info = MCTS(self.config).run(
            self.model,
            stacked_obvs,
            self.environment.legal_actions(),
            self.environment.player_id(),
            add_exploration_noise=True,
        )
        # Sample action from the MCTS visit counts.
        action = self.select_action_from_node(root, temperature)
        return action, root

    def _select_opponent_action(self, opponent: str, stacked_s: NDArray):
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
            root, self.mcts_info = MCTS(self.config).run(
                self.model,
                stacked_s,
                self.environment.legal_actions(),
                self.environment.player_id(),
                add_exploration_noise=True,
            )
            suggestion = self.select_action_from_node(root, temperature=0)
            return self.environment.human_to_action(), root
        elif opponent == "expert":
            return self.environment.expert_agent(), None
        elif opponent == "random":
            legal_actions = self.environment.legal_actions()
            assert legal_actions, f"Legal actions should not be empty. Got {legal_actions}."
            assert set(legal_actions).issubset(set(self.config.action_space)
                                               ), "Legal actions must be a subset of the action space."
            action = int(np.random.choice(legal_actions))
            return action, None
        else:
            raise NotImplementedError(
                'Invalid opponent: must be "self", "human", "expert", or "random".')

    @staticmethod
    def select_action_from_node(node: Node, temperature: float):
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

        if temperature == 0.0:
            return actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            return int(np.random.choice(actions))
        else:
            # Adjust visit counts according to temperature and compute probabilities.
            adjusted_counts = visit_counts ** (1 / temperature)
            probabilities = adjusted_counts / np.sum(adjusted_counts)
            return int(np.random.choice(actions, p=probabilities))

    def _enforce_training_ratio(self, checkpoint):
        """
        Wait until the ratio of trained steps to played steps is below the threshold.

        This ensures that the model is sufficiently trained before generating more self-play data.
        """
        printed = False
        while True:
            self._update_steps()
            if not printed:
                printed = True
            # if num_trained_steps >= num_played_steps * self.config.ratio_train_play:
            #     break
            break
            time.sleep(0.5)

    def _update_steps(self):
        self.num_trained_steps = \
            int(ray.get(self.checkpoint.get_info.remote("num_trained_steps")))
        self.num_played_steps = \
            max(1, int(ray.get(self.checkpoint.get_info.remote("num_played_steps"))))

    def _close_environment(self):
        """Close the environment M."""
        self.environment.close()
