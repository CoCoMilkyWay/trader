import numpy as np
from stable_baselines3.common.policies import BasePolicy

class RandomMaskedPolicy(BasePolicy):
    """
    A custom policy that randomly selects actions based on the action mask provided 
    by TokenGenEnv, without changing the environment itself.
    """
    def __init__(self, observation_space, action_space):
        super(RandomMaskedPolicy, self).__init__(observation_space, action_space)

    def predict(self, observation: np.ndarray, action_mask: np.ndarray, deterministic: bool = False) -> int:
        # Get the valid action indices based on the action mask
        valid_actions = np.where(action_mask)[0]

        if valid_actions.size == 0:  # If no valid actions are available
            raise RuntimeError("No valid actions are available. This should not happen!")

        # Randomly select one of the valid actions
        action_index = np.random.choice(valid_actions)
        return action_index

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        # This function is not used for random masking as we handle it in predict.
        raise NotImplementedError

    def get_action_mask(self, info):
        # Fetch action masks from the environment info if necessary
        return info['action_masks']  # Assuming your info dictionary from step returns this