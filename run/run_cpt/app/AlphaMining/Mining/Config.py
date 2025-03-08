import os
import datetime

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
DATAPATH = f"{os.path.dirname(__file__)}/../Example/TimeSeries"
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


class AgentConfig:
    def __init__(self) -> None:
        # Game
        self.observation_shape = (1, 1, MAX_EXPR_LENGTH) # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(SIZE_ACTION)) # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0 # Number of previous observations and previous actions to add to the current observation

        # Networks
        self.arch = 'mlp'  # mlp/resnet
        self.support_size = 10 # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.num_blocks = 6  # Number of blocks in the ResNet
        self.num_channels = 128 # Number of features planned to extract(by filters) from the convolutional layers of the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.res_reward_layers = [64] # Define the hidden layers in the reward head of the dynamic network
        self.res_value_layers = [64] # Define the hidden layers in the value head of the prediction network
        self.res_policy_layers = [64] # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 8
        self.mlp_representation_layers = [] # Define the hidden layers in the representation network
        self.mlp_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.mlp_reward_layers = [16]  # Define the hidden layers in the reward network
        self.mlp_value_layers = [16]  # Define the hidden layers in the value network
        self.mlp_policy_layers = [16]  # Define the hidden layers in the policy network

        # Self-Play
        self.selfplay_on_gpu = False
        self.num_workers = 1 # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.ratio_train_play = 0.8 # make sure model is sufficiently trained before continue playing (set to None to disable in synchronous mode)
        self.render = True

        # MCTS
        self.num_rollout_sims = 200 # Number of games moves for MCTS to simulate in rollout stage to estimate next best action
        #   UCB formula (prior score) (MCTS-Selection) (Root exploration noise)
        self.root_exploration_fraction = 0.25 # Fraction of the exploration noise added to the root prior
        self.root_dirichlet_alpha = 0.3 # Dirichlet noise applied to exploration part of the root prior
        #   UCB formula (prior score) (MCTS-Selection)
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        #   UCB formula (value score) (MCTS-Selection)
        self.future_discount = 0.98 # [0,1] future_reward_weight (trade-offs between immediate rewards and expected cumulative future rewards)
        #   Temperature (action) (post-MCTS)
        #   Ensure the temperature does not drop too low (avoid complete determinism too early).
        self.max_temperature = 1.0  # Max temperature at the start (full exploration)
        self.min_temperature = 0.1  # Min temperature value to allow some exploration
        self.decay_factor = 0.95  # The decay factor controls how fast the temperature decreases. (Smaller values decay faster)
        #   Recalculating target values (post-MCTS)
        self.future_steps = 1.0 * MAX_EXPR_LENGTH # Number of steps in the future to take into account for calculating the target value
        #   Prioritized Replay (post-MCTS)
        self.PER = True # select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # prioritization = diff(MCTS calculated value, target value) ** PER_alpha

        # Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer

        # Training
        self.train_on_gpu = torch.cuda.is_available()
        self.max_training_steps = 10000 # Total number of training steps (ie weights update according to a batch)
        self.update_interval = 1  # Number of batches before self-play, between updates, between checkpoint saves
        self.results_path = f"{os.path.dirname(__file__)}/RL/CheckPoints/{datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}" # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        #   Batching
        self.batch_size = 128  # Number of games to train each time(per batch)
        # unroll steps need to be equal to future_steps, otherwise have to recalculate priorities and others
        # batch data has trajectory[rand_idx, rand_idx + unroll_steps + 1] as training data
        #   Learning Rate
        self.lr_init = 0.02  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000
        #   Loss()
        self.value_loss_weight = 0.5  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        #   Optimizer
        self.optimizer = "Adam"  # "Adam"(Preferred) or "SGD"
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Reanalyze (See paper appendix Reanalyse)
        self.reanalyse_on_gpu = False
        # use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)

        # Misc
        self.seed = 0  # Seed for numpy, torch and the game

        # Overwrite TODO
        self.observation_shape = (1, 1, 4)
        self.action_space = list(range(2))
        self.players = list(range(1))  # List of players
        self.future_steps = 100
        self.num_rollout_sims = 200
        self.support_size = 2
        self.stacked_observations = 1
        
    def to_dict(self):
        return self.__dict__
      
