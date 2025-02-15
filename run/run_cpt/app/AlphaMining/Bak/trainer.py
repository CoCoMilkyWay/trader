import jax
import optax
from jax import random

from Mining.Config.Config import Config
class AlphaSearcher:
    def __init__(self, computation_data, id):
         
        self.computation_data = computation_data
        self.id = id
        self.rng = random.PRNGKey(Config.seed + id)
        self.network = Network(config.network, config.task_spec, config.seed)
        self.uniform_network = make_uniform_network(config.task_spec.num_actions)
        self.num_alphas_found = 0
        self.gpu_id = int(ray.get_gpu_ids()[0])
        print(f"Searcher {id} initialized on gpu: {self.gpu_id}")


    def run_alpha_search(self, network_params, network_training_step, use_uniform_network=False): # task
        if use_uniform_network:
            game = play_game(
                self.config, self.uniform_network, self.rng, self.computation_data)
        else:
            self.network.set_params(network_params)
            game = play_game(
                self.config, self.network, self.rng, self.computation_data)
        self.num_alphas_found += 1
        metric =game.environment.evaluate()
        info = {"network_step": network_training_step, "searcher_id": self.id, "num_alphas_found": self.num_alphas_found,  'metric': metric}
        return game, info