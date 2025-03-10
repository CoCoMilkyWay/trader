import time
import numpy
import ray
import torch

from Mining.Config import *

from Mining.RL.Agent.Core.CheckPoint import CheckPoint
from Mining.RL.Agent.Core.Network import Network

@ray.remote
class Reanalyse:
    """
    Class which run in a dedicated thread to update the replay buffer with fresh information.
    See paper appendix Reanalyse.
    """

    def __init__(self, initial_checkpoint):

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        self.net = Network()
        self.net.set_weights(initial_checkpoint.weights)
        self.net.to(torch.device("cuda" if reanalyse_on_gpu else "cpu"))
        self.net.eval()

        self.num_reanalysed_games = initial_checkpoint.num_reanalysed_games

    def reanalyse(self, replay_buffer, checkpoint):
        while ray.get(checkpoint.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        while ray.get(
            checkpoint.get_info.remote("num_trained_steps")
        ) < num_trained_steps and not ray.get(
            checkpoint.get_info.remote("terminate")
        ):
            self.net.set_weights(
                ray.get(checkpoint.get_info.remote("weights")))

            game_id, Trajectory, _ = ray.get(
                replay_buffer.sample_game.remote(force_uniform=True)
            )

            # Use the last net to provide a fresher, stable n-step value (See paper appendix Reanalyze)
            if use_last_net_value:
                observations = numpy.array(
                    [
                        Trajectory.get_stacked_observations(
                            i,
                            stacked_observations,
                            len(action_space),
                        )
                        for i in range(len(Trajectory.values))
                    ]
                )

                observations = (
                    torch.tensor(observations)
                    .float()
                    .to(next(self.net.parameters()).device)
                )
                values = nets.support_to_scalar(
                    self.net.initial_inference(observations)[0],
                    support_size,
                )
                Trajectory.reanalysed_predicted_root_values = (
                    torch.squeeze(values).detach().cpu().numpy()
                )

            replay_buffer.update_Trajectory.remote(game_id, Trajectory)
            self.num_reanalysed_games += 1
            checkpoint.set_info.remote(
                "num_reanalysed_games", self.num_reanalysed_games
            )
