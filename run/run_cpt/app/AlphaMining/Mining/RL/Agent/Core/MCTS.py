import math
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from typing import List, Dict, Tuple, Any

# Placeholder imports for MuZero-specific utilities (to be provided by the user’s environment)
from Mining.RL.Agent.Core.Network import AbstractNetwork
from Mining.RL.Agent.Core.Util.Network_Util import support_to_scalar, Objective_Value_Stats


class Node:
    """
    Represents a node in the MCTS tree, encapsulating the state and statistics
    for a given position in the game.
    """

    def __init__(self, prior: float):
        """Initialize a node with a prior probability."""
        # dynamics(current latent states, predicted from parent latent states)
        self.latent_rep: Tensor = torch.empty(0)  # Latent state representation

        # reward
        self.reward: float = 0.0  # Predicted reward transitioning to this node

        # value (v-value)
        # Q-value: expected cumulative rewards from now = E[R(t0+)]
        # objective-value: usually a variant of Q-value (e.g. (discount/weights/time-shift) cumulative rewards from now = f0(R(t0+)) )
        # v-value: usually a part of objective value (e.g. cumulative future rewards f1(R(t1+)))
        self.value_sum: float = 0.0  # Sum of objective-value of all visit times
        self.visit_count: int = 0  # Number of times this node has been visited

        # policy
        self.prior: float = prior  # Prior probability from the policy network

        # misc
        self.player: int = -1  # Player index (-1 if unset)
        self.children: Dict[int, Node] = {}  # Maps actions to child nodes

    def expanded(self) -> bool:
        """Check if the node has been expanded (i.e., has children)."""
        return bool(self.children)

    def value_mean(self) -> float:
        """Calculate the expected objective of the node."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def expand(self, actions: List[int], player: int, reward: float, policy_logits: Tensor, latent_rep: Tensor):
        """
        Expand the node by creating child nodes for each action using model predictions.

        Args:
            actions: List of legal actions from this state.
            player: The player whose turn it is at this node.
            reward: Predicted reward for reaching this node.
            policy_logits: Policy logits from the model for all actions.
            latent_rep: Latent state representation from the model.
        """
        self.player = player
        self.reward = reward
        self.latent_rep = latent_rep
        policy_probs = torch.softmax(torch.tensor(
            [policy_logits[0][a] for a in actions]), dim=0).tolist()
        for i, action in enumerate(actions):
            self.children[action] = Node(prior=policy_probs[i])

    def add_exploration_noise(self, dirichlet_alpha: float, exploration_fraction: float):
        """
        Add Dirichlet noise to children's priors to encourage exploration.

        Args:
            dirichlet_alpha: Parameter for Dirichlet distribution.
            exploration_fraction: Fraction of noise to mix with prior.
        """
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        for action, noise_value in zip(actions, noise):
            prior = self.children[action].prior
            self.children[action].prior = prior * \
                (1 - exploration_fraction) + noise_value * exploration_fraction

    def update_stats(self, value: float):
        """
        Update the node's statistics based on the simulation outcome.

        Args:
            value: Value prediction from the rollout.
            player: Player whose perspective the value is from.
        """
        self.visit_count += 1
        self.value_sum += value


class MCTS:
    """Manages the Monte Carlo Tree Search process with explicit steps."""

    def __init__(self, config):
        """
        Initialize MCTS with a configuration object.

        Args:
            config: Object containing parameters like action_space, num_rollout_sims, etc.
        """
        self.config = config
        self.is_zero_sum_game = len(config.players) != 1

    def run(self, model: AbstractNetwork, stacked_obvs: NDArray, legal_actions: List[int], root_player: int, add_exploration_noise: bool) -> Tuple[Node, Dict[str, Any]]:
        """
        Run MCTS simulations to build the search tree and return the root node with extra info.
        Explicitly follows the four steps: Selection, Expansion, Rollout, and Backpropagation.

        Args:
            model: Neural network model for predictions.
            state: Current game state as a numpy array.
            legal_actions: List of legal actions from the current state.
            current_player: Player whose turn it is.
            add_exploration_noise: Whether to add Dirichlet noise to the root.

        Returns:
            Tuple of (root node, extra info dictionary).
        """

        # Root calculation ==============================================================
        root_node = Node(prior=0)
        # shape: [1, (stacked_observations*2+1)*channels, height, width,]
        stacked_obvs_tensor = torch.tensor(stacked_obvs).float().unsqueeze(0).\
            to(next(model.parameters()).device)

        # the model outputs logits even for scalar values then perform transformation
        # this is because the transformation behaves like a custom(more advanced) last layer
        # may have benefits in both algo performance and descent speed
        repr_logits, policy_logits, value_logits, reward_logits = \
            model.initial_inference(stacked_obvs_tensor)
        root_v_pred = support_to_scalar(  # shape: (batch_size, 1)
            value_logits, self.config.support_size).item()
        root_r_pred = support_to_scalar(  # shape: (batch_size, 1)
            reward_logits, self.config.support_size).item()

        # Validate inputs
        assert legal_actions, f"Legal actions should not be empty. Got {legal_actions}."
        assert set(legal_actions).issubset(set(self.config.action_space)
                                           ), "Legal actions must be a subset of the action space."

        # Expand the root node
        root_node.expand(
            actions=legal_actions,
            player=root_player,
            reward=root_r_pred,
            policy_logits=policy_logits,
            latent_rep=repr_logits,
        )

        # Add exploration noise to the root if specified
        if add_exploration_noise:
            root_node.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        objective_value_stats = Objective_Value_Stats()
        max_tree_depth = 0

        # Main simulation loop with explicit four steps
        for _ in range(self.config.num_rollout_sims):
            # from an empty tree, at each time:
            #       1. select semi-randomly horizontally (ucb)
            #       2. expand by 1 vertically
            # after many loops, the tree can grow both horizontally and vertically
            # thus able to select more complicated leaf path that never explored

            # Step 1: Selection - Traverse the tree to find the leaf node
            leaf_node, leaf_player, search_path_nodes, search_path_actions = self.select(
                root_node, root_player, objective_value_stats)

            # Step 2 & 3: Expansion and Rollout
            if search_path_actions:
                parent_node_to_leaf = search_path_nodes[-2]
                action_to_leaf = search_path_actions[-1]
                leaf_v_pred = self.expand_and_rollout(
                    parent_node_to_leaf, leaf_node, leaf_player, action_to_leaf, model)
            else:
                # Root case: use initial value prediction
                leaf_v_pred = root_v_pred

            # Step 4: Backpropagation
            self.backpropagate(search_path_nodes,
                               leaf_v_pred, objective_value_stats)

            # Track tree depth
            tree_depth = len(search_path_nodes) - 1
            max_tree_depth = max(max_tree_depth, tree_depth)

        extra_info = {"max_tree_depth": max_tree_depth,
                      "root_v_pred": root_v_pred}
        return root_node, extra_info

    def select(self, root_node: Node, root_player: int, objective_value_stats: Objective_Value_Stats) -> tuple:
        """
        **Selection Step**: Traverse from the root to a leaf node using UCB scores.

        Args:
            root_node: The root node of the search tree.
            current_player: The player whose turn it is at the root.

        Returns:
            Tuple of (leaf node, search path, actions taken, simulation player).
        """
        current_node = root_node
        search_path_nodes = [current_node]
        search_path_actions = []
        sim_player = root_player

        while current_node.expanded():
            best_score = -float('inf')
            for action, child in current_node.children.items():
                score = self.ucb_score(
                    current_node, child, objective_value_stats)
                if score > best_score:  # if value is carefully crafted, best node is unique
                    best_score = score
                    best_action = action
                    best_child = child

            search_path_nodes.append(best_child)
            search_path_actions.append(best_action)
            current_node = best_child

            # Players play turn by turn
            sim_player = self.config.players[(
                sim_player + 1) % len(self.config.players)]

        leaf_node = current_node
        leaf_player = sim_player

        return leaf_node, leaf_player, search_path_nodes, search_path_actions

    def expand_and_rollout(self, parent_node_to_leaf: Node, leaf_node: Node, leaf_player: int, action_to_leaf: int, model: AbstractNetwork) -> float:
        """
        **Expansion and Rollout Steps**: Expand the leaf node and predict the value using the model.
        In MuZero, rollout is replaced by a single model prediction.

        Args:
            parent_node_to_leaf: The parent of the leaf node.
            leaf_node: The leaf node to expand.
            leaf_player: Player whose turn it is at the leaf.
            action_to_leaf: Action taken to reach the leaf.
            model: Neural network model for predictions.

        Returns:
            Predicted value for the leaf node.
        """
        action_to_leaf_tensor = torch.tensor([[action_to_leaf]]).to(
            parent_node_to_leaf.latent_rep.device)
        dynamic_logits, policy_logits, value_logits, reward_logits = model.recurrent_inference(
            parent_node_to_leaf.latent_rep, action_to_leaf_tensor)
        leaf_v_pred = support_to_scalar(
            value_logits, self.config.support_size).item()
        leaf_r_pred = support_to_scalar(
            reward_logits, self.config.support_size).item()
        leaf_node.expand(
            actions=self.config.action_space,  # this is simulation, we allow all actions here
            player=leaf_player,
            reward=leaf_r_pred,
            policy_logits=policy_logits,
            latent_rep=dynamic_logits,
        )
        return leaf_v_pred

    def backpropagate(self, search_path: List[Node], leaf_value: float, objective_value_stats: Objective_Value_Stats):
        """
        **Backpropagation Step**: Update node statistics along the search path.

        Args:
            search_path: List of nodes from root to leaf.
            leaf_value: Value prediction from the rollout (positive).
        """
        players = self.config.players
        num_player = len(players)

        if num_player not in [1, 2]:
            raise NotImplementedError(
                "More than two player mode not implemented")

        current_v_value = leaf_value

        for node in reversed(search_path):
            # search_path = [root, ..., leaf] (NOTE: newly expanded node is not in list)

            # update v-value first:
            node.update_stats(current_v_value)

            # NOTE: tricky here:
            # check UCB definition to see why it works lke this
            immediate_reward = node.reward
            expected_cumulative_future_rewards = \
                -node.value_mean() if self.is_zero_sum_game else node.value_mean()
            objective_value = immediate_reward + \
                self.config.future_discount * expected_cumulative_future_rewards
            objective_value_stats.update(objective_value)

            # NOTE: tricky here:
            # update v-value for the next(parent) node in search path
            # for both zero-sum and non-zero-sum games, need to determine what rewards are benefiting whom
            # v-value: e.g. transformed expected cumulative future rewards
            parent_player = players[(node.player-1) % num_player]
            current_player = node.player
            child_player = players[(node.player+1) % num_player]

            next_v_value = \
                node.reward * (1 if parent_player == current_player else -1) +\
                current_v_value * self.config.future_discount * \
                (1 if parent_player == child_player else -1)
            current_v_value = next_v_value

    def ucb_score(self, parent: Node, child: Node, objective_value_stats: Objective_Value_Stats) -> float:
        """
        Calculate the UCB(Upper Confidence Bound) score for a child node to balance exploration and exploitation.
        Args:
            parent: Parent node.
            child: Child node to score.
        Returns:
            UCB score as a float.
        """

        pb_c_init = float(self.config.pb_c_init)
        pb_c_base = float(self.config.pb_c_base)

        # Calculate the exploration factor (pb_c) using a formula that includes:
        # - log of the count of visits to the parent + a base configuration value
        # - the square root of the visit count of the parent
        # - normalization of the visit count of the child to ensure small values don't dominate
        pb_c = (math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) +
                pb_c_init) * math.sqrt(parent.visit_count) / (child.visit_count + 1)

        # Calculate the prior bonus by multiplying the exploration factor with the child's prior probability
        prior_score = pb_c * child.prior

        # Get the Q(value) score from the child's evaluation
        # Q(s,t) = E[R(t+) | S(t) = s, A(t) = a]
        # Q-value: expected cumulative rewards from now = E[R(t0+)]
        # objective-value: usually a variant of Q-value (e.g. (discounted/weighted/time-shifted) expected cumulative rewards from now = E[f0(R(t0+))] )
        # v-value: usually a part of objective value (e.g. expected cumulative future rewards E[f1(R(t1+))] )
        # NOTE: it is a bit tricky here, for multiple players with expert-level skills:
        #       1. Q-value at time t0 means E[R(t0+)] for player 1
        #       2. Q-value at time t1 means E[R(t1+)] for player 2 (different player)
        #       4. essentially a zero sum game, as 1 player's gain is another's loss
        #       5. by minimax principle, you should maximize your immediate reward at t0 while minimize cumulative rewards from t1 for your opponent
        #       6. the objective/v value is a variant of Q-value, but it follows this rule too

        immediate_reward = child.reward
        expected_cumulative_future_rewards = \
            -child.value_mean() if self.is_zero_sum_game else child.value_mean()

        objective_value = immediate_reward + \
            self.config.future_discount * expected_cumulative_future_rewards
        objective_score = objective_value_stats.normalize(objective_value) \
            if child.visit_count > 0 else 0.0

        # The final UCB score combines the prior score and the Q score
        return prior_score + objective_score
