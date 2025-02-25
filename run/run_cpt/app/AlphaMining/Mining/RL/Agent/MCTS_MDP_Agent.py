import torch
import numpy as np
import torch.optim as optim
from typing import Tuple, List, Dict
from torch.distributions import Categorical

from Mining.Config import EPISODES
from Mining.RL.Env.TokenGenEnv import TokenGenEnv
from Mining.RL.Policy.MaskedSimpleNetwork import MaskedSimpleNetwork

ENV = TokenGenEnv

class MCTS_MDP_Agent:
    """
    MDP with MCTS as policy core
    
    MCTS (Monte Carlo Tree Search) with:
        Selection, Expansion, Rollout, Back-propagation
    MDP (Markov Decision Process) with:
        1. model-free
        2. discrete action space, discrete state space
        3. off-policy learning:
            a. behavior policy: balance exploration/exploitation
            b. target policy: use replay Buffer (like DQN/DDPG/...)
        4. policy-based (Actor-Critic specifically):
            a. utilize policy networks (like A3C/PPO/...) to implement policy gradient (Actor)
            b. with quantile optimization to emphasize upper quantile returns
            c. has critic part to minimize variance and improve sample efficiency of policy networks, 
        5. temporal structure: Learn based on previous actions
        6. exploration: risk-seeking like A3C/PPO/TD2 with (e.g. entropy bonuses, action noise, ...)
    """
    
    def __init__(self, env: ENV):
        self.env = env
        self.obv_space = env.observation_space # Box(low=0, high=140, shape=(15,), uint8)
        self.atc_space = env.action_space # Discrete(140)
        self.state_size = int(self.obv_space.shape[0]) # 1D-state # type: ignore
        self.action_size = int(self.atc_space.n) # type: ignore
        self.policy_net = MaskedSimpleNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.gamma = 0.99  # Discount factor
        
        self.state:np.ndarray
        self.action:int
        self.action_mask:List[bool]
        
    def action_policy(self) -> Tuple[int, float]:
        if np.where(self.action_mask)[0].size == 0:  # Check if there are any True values
            raise RuntimeError(
                f"Garbage Expression generated:{self.env._tokens}")
        
        state_tensor = torch.FloatTensor(self.state)
        action_probs = self.policy_net.forward(state_tensor, self.action_mask)
        m = Categorical(torch.exp(action_probs))
        sample = m.sample()
        action = int(sample.item())
        prob = m.log_prob(sample)
        return action, prob

    def update_policy(self, rewards, log_probs):
        # Calculate the discounted rewards
        discounted_rewards = []
        R = 0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)

        # Normalize the rewards
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-10)

        # Perform policy gradient step
        loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            loss.append(-log_prob * reward)  # Policy gradient loss

        self.optimizer.zero_grad()
        loss = torch.cat(loss).sum()  # Combine losses
        loss.backward()
        self.optimizer.step()
    
    def run(self):
        env = self.env
        agent = self
        for episode in range(EPISODES):
            done = False
            self.state, info = env.reset()
            self.action_mask = info['action_masks']
            # log_probs = []
            # rewards = []

            while not done:
                self.action, prob = agent.action_policy()
                self.state, reward, done, _, info = env.step(self.action)
                self.action_mask = info['action_masks']
                
                # # Store log probs and rewards
                # log_probs.append(log_prob)
                # rewards.append(reward)
                
            # # Update the policy
            # agent.update_policy(rewards, log_probs)

            # if episode % 10 == 0:
            #     print(f"Episode {episode}: Total Reward: {sum(rewards)}")
            