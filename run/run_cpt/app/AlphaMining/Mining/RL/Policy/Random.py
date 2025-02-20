import gymnasium as gym
import numpy as np

class RandomMaskedAgent:
    def __init__(self, env):
        self.env = env
        
    def select_action(self, action_mask):
        """Select a random valid action based on the action mask."""
        valid_actions = np.where(action_mask)[0]
        action = np.random.choice(valid_actions)
        return action

    def run_episode(self):
        """Run a single episode using the random policy."""
        observation, info = self.env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            # Get the action mask from the environment's info
            action_mask = info['action_masks']
            # Select a random valid action
            action = self.select_action(action_mask)
            # Take the action in the environment
            observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            # Optional: print the action and current total reward
            print(f"Step {step}: Action {action}, Reward {reward}, Total Reward {total_reward}")

        print(f"Episode finished after {step} steps with total reward {total_reward}")
        return total_reward

# if __name__ == "__main__":
#     # Initialize the environment (ensure required components are properly defined)
#     env = TokenGenEnv(expression_builder, expression_parser, alpha_pool)
#     agent = RandomMaskedAgent(env)
# 
#     num_episodes = 10  # Specify the number of episodes you want to run
#     for episode in range(num_episodes):
#         print(f"\nStarting Episode {episode + 1}")
#         agent.run_episode()