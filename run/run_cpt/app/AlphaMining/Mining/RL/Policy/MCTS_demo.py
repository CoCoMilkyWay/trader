import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from collections import defaultdict
from typing import Optional, Tuple, List

# Define the neural network policy for PPO
class PPOPolicy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PPOPolicy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_policy = nn.Linear(hidden_size, action_size)
        self.fc_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        policy_logits = self.fc_policy(x)
        value = self.fc_value(x)
        return policy_logits, value

# Node class for MCTS
class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior_prob=0.0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.action = action
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_prob = prior_prob

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, actions, priors):
        for action, prior in zip(actions, priors):
            next_state = self.state.copy()
            next_state[self.parent.counter] = action
            self.children[action] = MCTSNode(
                state=next_state, parent=self, action=action, prior_prob=prior
            )

    def update(self, value):
        self.visit_count += 1
        self.total_value += value

    def ucb_score(self, c_puct):
        if self.visit_count == 0:
            q_value = 0
        else:
            q_value = self.total_value / self.visit_count
        u_value = c_puct * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return q_value + u_value

# MCTS with PPO as policy core
class MCTSPPOAgent:
    def __init__(
        self,
        env,
        policy,
        c_puct=1.0,
        n_simulations=50,
        gamma=0.99,
        lr=1e-3,
        ppo_epochs=10,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01
    ):
        self.env = env
        self.policy = policy
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def select_action(self, node):
        for _ in range(self.n_simulations):
            self.simulate(node)
        # Choose action with maximum visit count
        visits = [(child.visit_count, action) for action, child in node.children.items()]
        _, best_action = max(visits)
        return best_action

    def simulate(self, node):
        path = []
        current_node = node
        while not current_node.is_leaf():
            # Select the child with highest UCB score
            action, current_node = self.select_child(current_node)
            path.append(current_node)
        # Expand if not terminal
        is_terminal = self.is_terminal(current_node)
        if not is_terminal:
            self.expand_node(current_node)
        # Estimate value
        value = self.evaluate(current_node)
        # Backpropagate
        self.backpropagate(path, value)

    def select_child(self, node):
        ucb_scores = {action: child.ucb_score(self.c_puct) for action, child in node.children.items()}
        best_action = max(ucb_scores, key=ucb_scores.get)
        return best_action, node.children[best_action]

    def expand_node(self, node):
        state_tensor = torch.FloatTensor(node.state).unsqueeze(0)
        with torch.no_grad():
            policy_logits, _ = self.policy(state_tensor)
        action_probs = torch.softmax(policy_logits, dim=-1).squeeze(0).numpy()
        # Get valid actions from the environment
        mask = self.env.info['action_masks']
        valid_actions = np.where(mask)[0]
        priors = action_probs[valid_actions]
        # Normalize priors
        priors /= np.sum(priors)
        node.expand(valid_actions, priors)

    def evaluate(self, node):
        state_tensor = torch.FloatTensor(node.state).unsqueeze(0)
        with torch.no_grad():
            _, value = self.policy(state_tensor)
        return value.item()

    def backpropagate(self, path, value):
        for node in reversed(path):
            node.update(value)
            value = self.gamma * value

    def is_terminal(self, node):
        # Check if the node corresponds to a terminal state
        return self.env.counter >= self.env.MAX_EXPR_LENGTH

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            root_node = MCTSNode(state)
            done = False
            trajectory = []
            while not done:
                action = self.select_action(root_node)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                # Store in trajectory for PPO update
                trajectory.append((state, action, reward))
                # Move to next node
                if action in root_node.children:
                    root_node = root_node.children[action]
                else:
                    root_node = MCTSNode(next_state)
                state = next_state
            # Update policy using PPO
            self.update_policy(trajectory)
            print(f"Episode {episode + 1}/{num_episodes} completed.")

    def update_policy(self, trajectory):
        states = torch.FloatTensor([t[0] for t in trajectory])
        actions = torch.LongTensor([t[1] for t in trajectory]).unsqueeze(-1)
        rewards = [t[2] for t in trajectory]
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).unsqueeze(-1)
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # Old policy
        with torch.no_grad():
            old_logits, old_values = self.policy(states)
            old_log_probs = torch.gather(torch.log_softmax(old_logits, dim=-1), 1, actions)
        # PPO update
        for _ in range(self.ppo_epochs):
            logits, values = self.policy(states)
            log_probs = torch.gather(torch.log_softmax(logits, dim=-1), 1, actions)
            entropy = -torch.mean(torch.sum(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1), dim=-1))
            ratios = torch.exp(log_probs - old_log_probs)
            advantages = returns - values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.value_loss_coef * (returns - values).pow(2).mean()
            loss = policy_loss + value_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Usage
if __name__ == "__main__":
    # Initialize the environment (assuming necessary components are defined)
    env = TokenGenEnv(expression_builder, expression_parser, alpha_pool)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy = PPOPolicy(state_size=state_size, action_size=action_size)

    agent = MCTSPPOAgent(env, policy)
    agent.train(num_episodes=100)