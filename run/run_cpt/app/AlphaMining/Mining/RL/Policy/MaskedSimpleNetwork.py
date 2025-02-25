import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional

class MaskedSimpleNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(MaskedSimpleNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Hidden layer
        self.fc2 = nn.Linear(128, output_size)  # Output layer
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, action_mask:Optional[List[bool]]=None):
        # x = torch.relu(self.fc1(x))
        # x = self.softmax(self.fc2(x))
        
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)  # Get logits
        if action_mask is not None:
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=logits.device)
            # Set invalid actions to a very low value
            logits += (1 - action_mask_tensor) * -1e10
        x = self.softmax(logits)
        return x

"""
self.policy_net = PolicyNetwork(self.state_size, self.action_size)
self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
self.gamma = 0.99  # Discount factor

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
"""
