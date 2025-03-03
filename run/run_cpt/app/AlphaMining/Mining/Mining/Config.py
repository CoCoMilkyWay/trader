import os
import datetime
from dataclasses import dataclass

from Mining.Data.Data import Data
from Mining.Expression.Operator import *
from Mining.Expression.Operand import *

# OPERANDS (CONSTANT) =========================================================
CONST_TIMEDELTAS = [1, 5, 10, 20, 30, 40, 50, 60, 120, 240]
CONST_RATIOS = [0.01, 0.05, 0.1, 0.3, 0.5, 1., 3., 5., 10.]
CONST_OSCILLATORS = [10., 20., 30., 40., 50., 60., 70., 80., 90.]
CONST_RATIOS.extend([-const for const in CONST_RATIOS])
CONST_OSCILLATORS.extend([-const for const in CONST_OSCILLATORS])

ALL_DIMENSION_TYPES = ['price', 'volume', 'ratio',
                       'misc', 'oscillator', 'timedelta', 'condition']

# OPERATORS and OPERAND ===================================================================
# use un-instanced class to avoid computation overhead (thus Type[Operator/Operand])
OPERATORS: List[Type[Operator]] = [
    # Unary
    Abs, Sign, Log1p, CS_Rank,
    # Binary
    Add, Sub, Mul, Div, Pow, Max, Min,
    # Rolling
    TS_Ref, TS_Mean, TS_Sum, TS_Std, TS_Var, TS_Skew, TS_Kurt, TS_Max, TS_Min,
    TS_Med, TS_Mad, TS_Rank, TS_Delta, TS_WMA, TS_EMA,
    # Pair rolling
    TS_Cov, TS_Corr
]
OPERAND: List[Type[Operand]] = [Operand]

# DATA and OPERANDS (FEATURES) ================================================
MAX_PAST = max(CONST_TIMEDELTAS)
MAX_FUTURE = 0
DATAPATH = f"{os.path.dirname(__file__)}/Data/Example/TimeSeries"
DATA = Data(DATAPATH, MAX_PAST, MAX_FUTURE, init=True)
FEATURES = DATA.features
LABELS = DATA.labels
DIMENSIONS = DATA.dimensions
SCALARS = DATA.scalar

# AlphaPool ===================================================================
CAPACITY = 5
IC_LOWER_BOUND = -1

# RL Env/Agent ================================================================
# Env(Simulator/Game)
SIZE_OP = len(OPERATORS)
SIZE_FEATURE = len(FEATURES)
SIZE_CONSTANT_TD = len(CONST_TIMEDELTAS)
SIZE_CONSTANT_RT = len(CONST_RATIOS)
SIZE_CONSTANT_OS = len(CONST_OSCILLATORS)
SIZE_SEP = 1
SIZE_ACTION = SIZE_OP + SIZE_FEATURE + \
    SIZE_CONSTANT_TD + SIZE_CONSTANT_RT + SIZE_CONSTANT_OS +\
    SIZE_SEP
SIZE_NULL = 1

MAX_EXPR_LENGTH = 100
MAX_EPISODE_LENGTH = 256
REWARD_PER_STEP = 0.

# Game
observation_shape = (1, 1, MAX_EXPR_LENGTH) # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
action_space = list(range(SIZE_ACTION)) # Fixed list of all possible actions. You should only edit the length
players = list(range(1))  # List of players. You should only edit the length
stacked_observations = 0 # Number of previous observations and previous actions to add to the current observation

# Networks
arch = 'resnet'  # mlp/resnet
support_size = 10 # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

# Residual Network
downsample = False # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
num_blocks = 6  # Number of blocks in the ResNet
num_channels = 128 # Number of features planned to extract(by filters) from the convolutional layers of the ResNet
reduced_channels_reward = 2  # Number of channels in reward head
reduced_channels_value = 2  # Number of channels in value head
reduced_channels_policy = 4  # Number of channels in policy head
res_reward_layers = [64] # Define the hidden layers in the reward head of the dynamic network
res_value_layers = [64] # Define the hidden layers in the value head of the prediction network
res_policy_layers = [64] # Define the hidden layers in the policy head of the prediction network

# Fully Connected Network
encoding_size = 8
mlp_representation_layers = [] # Define the hidden layers in the representation network
mlp_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
mlp_reward_layers = [16]  # Define the hidden layers in the reward network
mlp_value_layers = [16]  # Define the hidden layers in the value network
mlp_policy_layers = [16]  # Define the hidden layers in the policy network

# Self-Play
selfplay_on_gpu = False
num_workers = 2 # Number of simultaneous threads/workers self-playing to feed the replay buffer
ratio_train_play = 1 # make sure model is sufficiently trained before continue playing (set to None to disable in synchronous mode)

# Training
train_on_gpu = torch.cuda.is_available()
max_training_steps = 10000 # Total number of training steps (ie weights update according to a batch)
batch_size = 512  # Number of games to train each time(per batch)
update_interval = 50  # Number of batches before self-play, between updates, between checkpoint saves
value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
results_path = f"{os.path.dirname(__file__)}/RL/CheckPoints/{datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}" # Path to store the model weights and TensorBoard logs
save_model = True  # Save the checkpoint in results_path as model.checkpoint

# Net(model)
optimizer = "Adam"  # "Adam"(Preferred) or "SGD"
weight_decay = 1e-4  # L2 weights regularization
momentum = 0.9  # Used only if optimizer is SGD

# Exponential learning rate schedule
lr_init = 0.002  # Initial learning rate
lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
lr_decay_steps = 10000

# Replay Buffer
replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
future_steps = 121 # Number of steps in the future to take into account for calculating the target value
PER = True # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

# Reanalyze (See paper appendix Reanalyse)
reanalyse_on_gpu = False
# use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)

# MCTS
num_rollout_sims = 100 # Number of games moves for MCTS to simulate in rollout stage to estimate next best action
# 0(value immediate gain): agent only cares about immediate rewards;
# 1(value long-term planning): agent cares about immediate/future rewards equally
future_discount = 1 # future_reward_weight
# UCB formula
pb_c_base = 19652
pb_c_init = 1.25
# Exploration noise (only the 1st-level rollout is applied with noise)
root_exploration_fraction = 0.25 # Fraction of the exploration noise added to the root prior
root_dirichlet_alpha = 0.3 # Dirichlet noise applied to exploration part of the root prior

# Misc
seed = 0  # Seed for numpy, torch and the game

@staticmethod
def temperature_func(self, trained_steps):
    """
    Adjusts the softmax temperature dynamically to control the balance between 
    exploration (trying new actions) and exploitation (choosing the best-known action).

    Monte Carlo Tree Search (MCTS) selects actions based on visit counts. This function 
    modifies how strictly the agent follows the most-visited action versus selecting 
    actions more randomly.

    - **Early Training:** The agent explores different strategies to avoid getting stuck 
      in a suboptimal policy.
    - **Mid Training:** The agent starts balancing between exploration and choosing 
      well-learned moves.
    - **Late Training:** The agent mostly exploits the best-known moves to maximize 
      performance.
    """

    # Ensure the temperature does not drop too low (avoid complete determinism too early).
    max_temperature = 1.0  # Max temperature at the start (full exploration)
    min_temperature = 0.1  # Min temperature value to allow some exploration

    # The decay factor controls how fast the temperature decreases.
    decay_factor = 0.95  # Smaller values decay faster

    # Compute the smooth decay of temperature
    temperature = max(min_temperature, max_temperature *
                      (decay_factor ** (trained_steps / (0.1 * self.training_steps))))

    return temperature  # Returns a positive temperature that decreases smoothly over time
