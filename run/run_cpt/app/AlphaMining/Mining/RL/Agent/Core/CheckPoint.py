import ray
import copy
import torch

# from Mining.Config import *
from dotmap import DotMap

@ray.remote
class CheckPoint:
    def __init__(self, config):
        self.config = DotMap(config)
        # Player-> ...

        # Trainer -> ...
        self.num_trained_steps = 0
        self.weights = None
        self.optimizer_state = None

        self.learning_rate = 0.0  # Logs
        self.total_loss = 0.0  # Logs
        self.value_loss = 0.0  # Logs
        self.reward_loss = 0.0  # Logs
        self.policy_loss = 0.0  # Logs

        # ReplayBuffer -> ...
        self.num_played_games = 0
        self.num_played_steps = 0

        # self.total_reward         = None
        # self.muzero_reward        = None
        # self.opponent_reward      = None
        # self.episode_length       = None
        # self.mean_value           = None

        # reanalyze
        self.num_reanalysed_games = 0

        # Control
        self.terminate = False
        print(f"CheckPoint inited...")

    @property
    def current_checkpoint(self):
        # Create a dictionary of current attributes, excluding methods
        return {k: v for k, v in self.__dict__.items() if not callable(v) and not k.startswith("__")}

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def save_checkpoint(self, path=None):
        if not path:
            path = f"{self.config.results_path}/checkpoint"
        torch.save(self.current_checkpoint, path)

    def get_info(self, keys):
        if isinstance(keys, str):
            # Use .get to handle missing keys safely
            return self.current_checkpoint.get(keys)
        elif isinstance(keys, list):
            return {key: self.current_checkpoint.get(key) for key in keys}
        else:
            raise TypeError("keys must be a string or a list of strings")

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.__dict__[keys] = values  # Update the attribute directly
        elif isinstance(keys, dict):
            for key, value in keys.items():
                if key in self.__dict__:
                    self.__dict__[key] = value  # Update the attribute directly
                else:
                    raise KeyError(f"{key} is not a valid attribute")
        else:
            raise TypeError("keys must be a string or a dictionary")
