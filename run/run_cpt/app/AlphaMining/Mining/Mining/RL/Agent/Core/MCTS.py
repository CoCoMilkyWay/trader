import math
import numpy as np
import torch

from Mining.Config import *

# Import configuration parameters and network utility functions.
from Mining.RL.Agent.Core.Util.Network_Util import support_to_scalar, scalar_to_support, MinMaxStats


class MCTS:
    def __init__(self):
        """
        Monte Carlo Tree Search (MCTS) for MuZero.

        MCTS simulates multiple futures starting from a given state (s). In each simulation,
        the tree is traversed using an Upper Confidence Bound (UCB) formula, and the model's
        predictions (from the unified network model that includes R, D, O, V, and π) are used
        to expand nodes and evaluate states.

        The process involves:
          - Converting the raw state (s) to a latent representation (rep) using the representation model R.
          - Running simulations to obtain a policy (action probabilities) and value estimate via MCTS.
          - Using dynamics (D), reward (O), value (V), and policy (π) models in recurrent inference.
        """

    def run(self, model, s, legal_actions, current_player, add_exploration_noise):
        """
        Run a series of MCTS simulations starting from state s.

        Parameters:
            model (torch.nn.Module): Unified MuZero model that implements:
                - Representation model R (for s -> latent representation rep),
                - Dynamics model D,
                - Reward model O,
                - Value model V,
                - Policy model π.
            s (array-like): The current state (s).
            legal_actions (list): List of allowed actions in the current state.
            current_player (int): Index of the current player.
            add_exploration_noise (bool): Whether to add Dirichlet noise to encourage exploration at the root.

        Returns:
            root (Node): The root node after expanding the search tree via MCTS.
            extra_info (dict): Contains extra info such as maximum tree depth and predicted root value.
        """
        # Create the root node with an initial prior of 0.
        root = Node(prior=0)

        # Convert state s to a tensor and ensure it is on the same device as the model.
        s_tensor = torch.tensor(s).float().unsqueeze(
            0).to(next(model.parameters()).device)

        # Use the representation model R to get the latent representation (rep),
        # and also get the predicted value (v_pred), reward (r_pred), and policy logits.
        (
            v_pred_tensor,
            r_pred_tensor,
            policy_logits,
            latent_rep,  # This is the latent representation from R(s)
        ) = model.initial_inference(s_tensor)

        # Convert the predicted value and reward from support representation to scalar numbers.
        root_v_pred = support_to_scalar(
            v_pred_tensor, support_size).item()
        initial_r_pred = support_to_scalar(
            r_pred_tensor, support_size).item()

        # Check that legal actions are provided and are within the allowed action space.
        assert legal_actions, f"Legal actions should not be empty. Got {legal_actions}."
        assert set(legal_actions).issubset(set(action_space)
                                           ), "Legal actions must be a subset of the action space."

        # Expand the root node using the initial predictions from the network.
        root.expand(
            actions=legal_actions,
            player=current_player,
            reward=initial_r_pred,
            policy_logits=policy_logits,
            latent_rep=latent_rep,
        )

        # Optionally add Dirichlet exploration noise to the root to help explore less-visited actions.
        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=root_dirichlet_alpha,
                exploration_fraction=root_exploration_fraction,
            )

        # Create a MinMaxStats object to keep track of the range of value estimates (for normalization).
        min_max_stats = MinMaxStats()
        # To record the deepest level reached in the tree during simulations.
        max_tree_depth = 0

        # Run a fixed number of simulations (k) to build the search tree.
        for _ in range(num_rollout_sims):
            sim_player = current_player  # The simulated player's turn.
            node = root  # Start each simulation at the root.
            # Keep track of the nodes visited in this simulation.
            search_path = [node]
            tree_depth = 0

            # Traverse the tree until reaching a leaf (node not yet expanded).
            while node.expanded():
                tree_depth += 1

                # Select the child node that maximizes the UCB score.
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

                # In a multi-player setting, cycle to the next player's turn.
                if sim_player + 1 < len(players):
                    sim_player = players[sim_player + 1]
                else:
                    sim_player = players[0]

            # Once a leaf is reached, use the dynamics (D), reward (O), value (V), and policy (π) models
            # to predict the outcome for the chosen action.
            parent_node = search_path[-2]
            action_tensor = torch.tensor([[action]]).to(
                parent_node.latent_rep.device)  # type: ignore
            v_pred_tensor, r_pred_tensor, policy_logits, next_latent_rep = model.recurrent_inference(
                parent_node.latent_rep, action_tensor
            )
            v_pred = support_to_scalar(
                v_pred_tensor, support_size).item()
            r_pred = support_to_scalar(
                r_pred_tensor, support_size).item()

            # Expand the leaf node with predictions over the complete action space.
            node.expand(
                actions=action_space,
                player=sim_player,
                reward=r_pred,
                policy_logits=policy_logits,
                latent_rep=next_latent_rep,
            )

            # Propagate the predicted value back up the search path.
            self.backpropagate(search_path, v_pred, sim_player, min_max_stats)
            max_tree_depth = max(max_tree_depth, tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_v_pred": root_v_pred,
        }
        return root, extra_info

    def select_child(self, parent_node, min_max_stats):
        """
        Select a child node based on the UCB (Upper Confidence Bound) score.

        UCB combines the predicted value (exploitation) and a bonus for less-visited nodes (exploration).

        Parameters:
            parent_node (Node): The node from which a child is selected.
            min_max_stats (MinMaxStats): Normalizer for node values.

        Returns:
            action (int): The action corresponding to the selected child.
            child_node (Node): The child node with the highest UCB score.
        """
        # Compute UCB scores for all children of the parent node.
        ucb_scores = {
            action: self.ucb_score(parent_node, child, min_max_stats)
            for action, child in parent_node.children.items()
        }
        max_ucb = max(ucb_scores.values())
        # In case several actions share the maximum score, choose one randomly.
        best_actions = [action for action, score in
                        ucb_scores.items() if score == max_ucb]
        chosen_action = np.random.choice(best_actions)
        return chosen_action, parent_node.children[chosen_action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        Compute the UCB score for a child node.

        The score consists of two parts:
          - Prior bonus: Encourages exploration based on the prior probability.
          - Value score: Normalized value from simulations.

        Parameters:
            parent (Node): The parent node.
            child (Node): The child node.
            min_max_stats (MinMaxStats): Used to normalize the value term.

        Returns:
            score (float): The computed UCB score.
        """
        # Calculate exploration coefficient based on the visit counts.
        pb_c = (math.log((parent.visit_count + pb_c_base +
                1) / pb_c_base) + pb_c_init)
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_bonus = pb_c * child.prior

        # If the child has been visited, calculate a normalized value score.
        if child.visit_count > 0:
            # In a multi-player setting, the value sign might be flipped for the opponent.
            adjusted_value = child.value() if len(
                players) == 1 else -child.value()
            value_score = min_max_stats.normalize(
                child.reward + future_discount * adjusted_value)
        else:
            value_score = 0

        return prior_bonus + value_score

    def backpropagate(self, search_path, v_pred, sim_player, min_max_stats):
        """
        Propagate the predicted value (v_pred) from the leaf node back up to the root.

        Each node in the path updates its visit count and cumulative value sum.
        The value is adjusted by the reward and discount factor as it moves upward.

        Parameters:
            search_path (list[Node]): The list of nodes from the root to the leaf.
            v_pred (float): The predicted value from the leaf node.
            sim_player (int): The player index at the end of the simulation.
            min_max_stats (MinMaxStats): Used for normalizing value estimates.
        """
        if len(players) == 1:
            # Single-player scenario: propagate value directly.
            for node in reversed(search_path):
                node.value_sum += v_pred
                node.visit_count += 1
                min_max_stats.update(
                    node.reward + future_discount * node.value())
                v_pred = node.reward + future_discount * v_pred
        elif len(players) == 2:
            # Two-player scenario: propagate with a sign flip for the opponent.
            for node in reversed(search_path):
                node.value_sum += v_pred if node.player == sim_player else -v_pred
                node.visit_count += 1
                min_max_stats.update(
                    node.reward + future_discount * -node.value())
                v_pred = (-node.reward if node.player ==
                          sim_player else node.reward) + future_discount * v_pred
        else:
            # More than two players is not implemented.
            raise NotImplementedError(
                "More than two player mode not implemented.")


class Node:
    """
    A node in the Monte Carlo Tree Search (MCTS) search tree.

    Each node represents a state in the game and stores information that helps
    the algorithm decide which moves to explore further. The key components are:
      - visit_count: Number of times this node has been visited during simulations.
      - player: The player index associated with this state (helps in alternating turns).
      - prior: The prior probability (from the policy network) that this node will be selected.
      - value_sum: Cumulative sum of simulation results (used to compute the average value, Q).
      - children: A dictionary that maps actions to the corresponding child nodes.
      - latent_rep: A compact representation of the state (from the representation or dynamics model).
      - reward: The predicted reward when reaching this state (from the reward model).
    """

    def __init__(self, prior):
        # Initialize a new node with a given prior probability.
        # How many times this node has been visited.
        self.visit_count = 0
        # The player index; -1 indicates it is not yet set.
        self.player = -1
        # The prior probability for selecting this node.
        self.prior = prior
        # Sum of values from sims; used to compute the mean value.
        self.value_sum = 0
        # Dictionary to hold child nodes; keys are actions.
        self.children = {}
        # Latent req of the state (filled during expansion).
        self.latent_rep = None
        # Predicted reward for this state.
        self.reward = 0

    def expanded(self):
        """
        Check if the node has been expanded.

        Returns:
            bool: True if the node has any children, False otherwise.
        """
        # If self.children is not empty, the node is considered expanded.
        return bool(self.children)

    def value(self):
        """
        Compute the mean value (Q value) of the node based on the simulations.

        Returns:
            float: The average value if visited; returns 0 if the node hasn't been visited.
        """
        # To avoid division by zero, return 0 if no visits have been made.
        if self.visit_count == 0:
            return 0
        # Return the average value (cumulative value divided by the number of visits).
        return self.value_sum / self.visit_count

    def expand(self, actions, player, reward, policy_logits, latent_rep):
        """
        Expand the node using predictions from the model.

        This method initializes child nodes for each possible action available in this state.
        It uses the policy model's logits to assign prior probabilities to each action.

        Parameters:
            actions (list): List of possible actions to expand from this node.
            player (int): The player index for this node.
            reward (float): The predicted reward for reaching this state.
            policy_logits (tensor): Raw output (logits) from the policy network for these actions.
            latent_rep (tensor): The latent representation of the state from the model.
        """
        # Set the player, reward, and latent representation for the current node.
        self.player = player
        self.reward = reward
        self.latent_rep = latent_rep

        # Compute probabilities for each action using softmax, converting logits to probabilities.
        policy_probs = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()

        # For each action, create a child node and set its prior probability.
        for i, action in enumerate(actions):
            self.children[action] = Node(prior=policy_probs[i])
            # Now each child node is initialized with its corresponding probability
            # which will guide the MCTS exploration later on.

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        Add Dirichlet noise to the prior probabilities of the root node's children.

        The noise helps to encourage exploration by slightly perturbing the priors,
        preventing the algorithm from getting stuck in local optima.

        Parameters:
            dirichlet_alpha (float): Parameter controlling the spread of the Dirichlet distribution.
            exploration_fraction (float): The fraction of the noise to blend with the original prior.
        """
        # Get a list of all available actions from the children.
        actions = list(self.children.keys())
        # Sample noise values from a Dirichlet distribution. The list length equals the number of actions.
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        # Update each child's prior probability by blending the original prior with the noise.
        for action, noise_value in zip(actions, noise):
            original_prior = self.children[action].prior
            # New prior is a weighted sum: (1 - exploration_fraction) * original + exploration_fraction * noise.
            self.children[action].prior = original_prior * \
                (1 - exploration_fraction) + noise_value * exploration_fraction
