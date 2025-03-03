import math
import torch
from abc import ABC, abstractmethod

from Mining.Config import *

from Mining.RL.Agent.Core.Util.Network_Util import *


class Network:
    """
    Policy Networks for:
    Representation, Dynamics, Reward, Value, Policy
    """
    def __new__(cls):
        if arch == "mlp":
            return MLPNetwork()
        elif arch == "resnet":
            return ResNetwork()
        else:
            raise NotImplementedError(
                f'The network arch {arch} not implemented.'
            )


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        if weights is not None:
            self.load_state_dict(weights)


class MLPNetwork(AbstractNetwork):
    def __init__(self):
        super().__init__()

        # Define the action space size and the size of the full support for the reward distribution
        self.action_space_size = len(action_space)
        self.full_support_size = 2 * support_size + 1

        # Initialize the representation network: processes input observations into a latent space representation
        self.representation_network = torch.nn.DataParallel(
            mlp(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                # Input features include stacked observations
                * (stacked_observations + 1)
                + stacked_observations *
                # Additional features
                observation_shape[1] * observation_shape[2],
                mlp_representation_layers,  # Layer specifications for the MLP
                encoding_size,  # Size of the output encoding
            )
        )

        # Initialize the dynamics network to predict next state encoding based on current state and action
        self.dynamics_encoded_state_network = torch.nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,  # Combine encoded state and action
                mlp_dynamics_layers,  # Layer specifications for the dynamics network
                encoding_size,  # Output size of the next encoded state
            )
        )

        # Initialize the reward network to predict rewards based on the next encoded state
        self.dynamics_reward_network = torch.nn.DataParallel(
            # Output size is full support for reward
            mlp(encoding_size, mlp_reward_layers, self.full_support_size)
        )

        # Initialize the policy network to predict action probabilities based on the current encoded state
        self.prediction_policy_network = torch.nn.DataParallel(
            # Output size corresponds to action space
            mlp(encoding_size, mlp_policy_layers, self.action_space_size)
        )

        # Initialize the value network to predict state values based on current encoded state
        self.prediction_value_network = torch.nn.DataParallel(
            # Output size is full support for value
            mlp(encoding_size, mlp_value_layers, self.full_support_size)
        )

    def prediction(self, encoded_state):
        # Make predictions for policy logits and state value based on encoded state
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        # Transform the observation into an encoded state
        encoded_state = self.representation_network(
            # Flatten the input observation
            observation.view(observation.shape[0], -1)
        )

        # Scale encoded state between [0, 1] for normalization (influences training stability)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state <
                            1e-5] += 1e-5  # Avoid division by zero
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state  # Normalize the encoded state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a one-hot encoded representation of the action
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()  # Create a one-hot tensor for actions
        )
        # One-hot encoding of action
        action_one_hot.scatter_(1, action.long(), 1.0)
        # Concatenate encoded state and action
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        # Predict the next encoded state based on current state and action
        next_encoded_state = self.dynamics_encoded_state_network(x)

        # Predict the reward for the next encoded state
        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale next encoded state between [0, 1] for normalization
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        # Avoid division by zero
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state  # Normalize the next encoded state

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        # Initial inference given an observation
        encoded_state = self.representation(
            observation)  # Get the encoded state
        policy_logits, value = self.prediction(
            encoded_state)  # Get policy and value predictions

        # Create a reward distribution centered around zero for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                # Centered reward
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return (
            value,  # Predicted value for the state
            reward,  # Predicted reward (set to zero initially)
            policy_logits,  # Predicted policy logits
            encoded_state,  # The encoded representation of the observation
        )

    def recurrent_inference(self, encoded_state, action):
        # Perform inference for subsequent timesteps given encoded state and action
        next_encoded_state, reward = self.dynamics(
            encoded_state, action)  # Get next state and reward
        # Get policy and value predictions for next state
        policy_logits, value = self.prediction(next_encoded_state)
        # Return outputs for further processing
        return value, reward, policy_logits, next_encoded_state


class ResNetwork(AbstractNetwork):
    def __init__(self):
        """
        Initializes the Residual Network.

        This network handles representation, dynamics, and predictions for reinforcement learning tasks.

        ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ MuZero ResNet Architecture                                                                                                    │
        ├──────────────────────┬───────────────────────────────────────────────────┬────────────────────────────────────────────────────┤
        │ Representation Network (ResRepresentationNetwork)                        │ Dynamics Network (ResDynamicsNetwork)              │
        │ Input: Observation                                                       │ Input: Encoded State + Action (concatenated)       │
        ├──────────────────────┼───────────────────────────────────┬───────────────┼───────────────────────────────┬────────────────────┤
        │ Downsample (Optional):                                   │               │ Conv3x3 (128+1→127 channels)  │                    │
        │ - "resnet": Residual downsample                          │ Conv3x3       │ BatchNorm                     │                    │
        │ - "CNN": Strided CNN                                     │ (channels=128)│ ReLU                          │                    │
        │                                                          │ BatchNorm     │                               │                    │
        │                                                          │ ReLU          │                               │                    │
        ├──────────────────────┼───────────────────────────────────┼───────────────┼───────────────────────────────┼────────────────────┤
        │ Residual Blocks (×6):                                    │               │ Residual Blocks (×6):         │                    │
        │ ┌─────────────────┐                                      │               │ ┌─────────────────┐           │                    │
        │ │ Conv3x3 128     │                                      │               │ │ Conv3x3 127     │           │                    │
        │ │ BatchNorm       │                                      │               │ │ BatchNorm       │           │                    │
        │ │ ReLU            │                                      │               │ │ ReLU            │           │                    │
        │ │ Conv3x3 128     │                                      │               │ │ Conv3x3 127     │           │                    │
        │ │ BatchNorm       │                                      │               │ │ BatchNorm       │           │                    │
        │ │ Skip Connection │                                      │               │ │ Skip Connection │           │                    │
        │ └─────────────────┘                                      │               │ └─────────────────┘           │                    │
        ├──────────────────────┼───────────────────────────────────┼───────────────┼───────────────────────────────┼────────────────────┤
        │ Output: Encoded State (128 channels)                     │               │ Output: Next State (127 ch)   │ Reward Head:       │
        │                                                          │               │                               │ 1x1 Conv→2 channels│
        │                                                          │               │                               │ Flatten→[64] MLP   │
        │                                                          │               │                               │ →Full Support      │
        ├──────────────────────┴───────────────────────────────────┴───────────────┴───────────────────────────────┴────────────────────┤
        │ Prediction Network (ResPredictionNetwork)                                                                                     │
        │ Input: Encoded State                                                                                                          │
        ├──────────────────────┬───────────────────────────────────┬───────────────────────────────────────┬────────────────────────────┤
        │ Residual Blocks (×6):                                    │ Value Head:                           │ Policy Head:               │
        │ ┌─────────────────┐                                      │ 1x1 Conv→2 channels                   │ 1x1 Conv→4 channels        │
        │ │ Conv3x3 128     │                                      │ Flatten→[64] MLP                      │ Flatten→[64] MLP           │
        │ │ BatchNorm       │                                      │ →Full Support                         │ →Action Space Size         │
        │ │ ReLU            │                                      │                                       │                            │
        │ │ Conv3x3 128     │                                      │                                       │                            │
        │ │ BatchNorm       │                                      │                                       │                            │
        │ │ Skip Connection │                                      │                                       │                            │
        │ └─────────────────┘                                      │                                       │                            │
        └──────────────────────┴───────────────────────────────────┴───────────────────────────────────────┴────────────────────────────┘

        Key Architectural Details:
        1. Residual Block Structure (All Networks):
           - Two 3x3 Conv layers with BatchNorm and ReLU
           - Channel dimensions preserved (128 for repr/pred, 127 for dynamics)
           - Skip connection around both conv layers

        2. Parameter-driven Dimensions:
           - num_channels=128: Base channel count for all conv layers
           - num_blocks=6: Number of residual blocks in each network
           - reduced_channels_*: Bottleneck channels before MLP heads (2/2/4)
           - res_*_layers=[64]: Single hidden layer in MLP heads

        3. Normalization:
           - Encoded states normalized to [0,1] after representation/dynamics
           - BatchNorm used after every convolution

        4. Downsampling Options:
           - False: Raw resolution maintained
           - "CNN": Strided convolutions
           - "resnet": Residual-based downsampling
        """
        super().__init__()
        # Size of action space, determined by the available actions
        self.action_space_size = len(action_space)
        # Full support size for the reward distribution
        self.full_support_size = 2 * support_size + 1

        # Calculate output sizes for reward, value, and policy networks based on whether downsampling is performed
        block_output_size_reward = (
            (reduced_channels_reward *
             math.ceil(observation_shape[1] / 16) * math.ceil(observation_shape[2] / 16))
            if downsample
            else (reduced_channels_reward * observation_shape[1] * observation_shape[2])
        )

        block_output_size_value = (
            (reduced_channels_value *
             math.ceil(observation_shape[1] / 16) * math.ceil(observation_shape[2] / 16))
            if downsample
            else (reduced_channels_value * observation_shape[1] * observation_shape[2])
        )

        block_output_size_policy = (
            (reduced_channels_policy *
             math.ceil(observation_shape[1] / 16) * math.ceil(observation_shape[2] / 16))
            if downsample
            else (reduced_channels_policy * observation_shape[1] * observation_shape[2])
        )

        # Initialize the representation network as data parallel (useful for multi-gpu training)
        self.representation_network = torch.nn.DataParallel(
            ResRepresentationNetwork(
                observation_shape,
                stacked_observations,
                num_blocks,
                num_channels,
                downsample,
            )
        )

        # Initialize the dynamics network
        self.dynamics_network = torch.nn.DataParallel(
            ResDynamicsNetwork(
                num_blocks,
                num_channels + 1,
                reduced_channels_reward,
                res_reward_layers,
                self.full_support_size,
                block_output_size_reward,
            )
        )

        # Initialize the prediction network
        self.prediction_network = torch.nn.DataParallel(
            ResPredictionNetwork(
                self.action_space_size,
                num_blocks,
                num_channels,
                reduced_channels_value,
                reduced_channels_policy,
                res_value_layers,
                res_policy_layers,
                self.full_support_size,
                block_output_size_value,
                block_output_size_policy,
            )
        )

    def prediction(self, encoded_state):
        """
        Generates policy and value predictions from the encoded state.

        Args:
        - encoded_state: The input state to generate predictions from.

        Returns:
        - policy: The output policy logits.
        - value: The output state value.
        """
        policy, value = self.prediction_network(
            encoded_state)  # Get predictions from the prediction network
        return policy, value

    def representation(self, observation):
        """
        Encodes the observation into a latent state.

        Args:
        - observation: Input observation to encode.

        Returns:
        - encoded_state_normalized: The normalized encoded state.
        """
        encoded_state = self.representation_network(
            observation)  # Pass observation through representation network

        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state <
                            1e-5] += 1e-5  # Avoid division by zero
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state  # Normalize encoded state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        """
        Models the dynamics of the environment, predicting the next state and reward.

        Args:
        - encoded_state: The current encoded state.
        - action: The action taken.

        Returns:
        - next_encoded_state_normalized: The normalized next encoded state.
        - reward: The predicted reward for the action taken.
        """
        # Stack encoded_state with a game-specific one-hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    1,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot = (
            action[:, :, None, None] * action_one_hot /
            self.action_space_size  # Scale action
        )
        # Concatenate state and action
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward = self.dynamics_network(
            x)  # Predict next state and reward

        # Scale next encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        # Avoid division by zero
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state  # Normalize next encoded state
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        """
        Performs initial inference from the observation.

        Args:
        - observation: Input observation to process.

        Returns:
        - value: The predicted value of the state.
        - reward: The predicted reward (initialized to zero for consistency).
        - policy_logits: The predicted policy logits.
        - encoded_state: The encoded representation of the state.
        """
        encoded_state = self.representation(
            observation)  # Encode the observation
        policy_logits, value = self.prediction(
            encoded_state)  # Get policy logits and value
        # Reward initialized to zero for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )
        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        """
        Performs recurrent inference given the encoded state and action.

        Args:
        - encoded_state: The current encoded state.
        - action: The action taken.

        Returns:
        - value: The predicted value for the next state.
        - reward: The predicted reward for the action taken.
        - policy_logits: The predicted policy logits for the next state.
        - next_encoded_state: The encoded representation of the next state.
        """
        next_encoded_state, reward = self.dynamics(
            encoded_state, action)  # Predict next state and reward
        policy_logits, value = self.prediction(
            next_encoded_state)  # Get predictions for next state
        return value, reward, policy_logits, next_encoded_state


class ResRepresentationNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        num_blocks,
        num_channels,
        downsample,
    ):
        """
        Initializes the Representation Network.

        This network encodes observations into a latent representation.

        Args:
        - observation_shape: Shape of the input observations.
        - stacked_observations: Number of stacked observations.
        - num_blocks: Number of residual blocks in the network.
        - num_channels: Number of channels for the convolutional layers.
        - downsample: Whether to downsample the input.
        """
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            # Select downsampling method based on the specified type
            if self.downsample == "resnet":
                self.downsample_net = DownSample(
                    observation_shape[0] *
                    (stacked_observations + 1) + stacked_observations,
                    num_channels,
                )
            elif self.downsample == "CNN":
                self.downsample_net = DownsampleCNN(
                    observation_shape[0] *
                    (stacked_observations + 1) + stacked_observations,
                    num_channels,
                    (
                        math.ceil(observation_shape[1] / 16),
                        math.ceil(observation_shape[2] / 16),
                    ),
                )
            else:
                raise NotImplementedError(
                    'downsample should be "resnet" or "CNN".')
        # Initial convolution followed by batch normalization
        self.conv = conv3x3(
            observation_shape[0] *
            (stacked_observations + 1) + stacked_observations,
            num_channels,
        )
        self.bn = torch.nn.BatchNorm2d(num_channels)
        # Residual blocks for deeper representation learning
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        """
        Forward pass through the representation network.

        Args:
        - x: Input observations to encode.

        Returns:
        - x: The encoded representation.
        """
        if self.downsample:
            x = self.downsample_net(x)  # Apply downsampling
        else:
            # If not downsampling, apply convolution and normalization
            x = self.conv(x)
            x = self.bn(x)
            x = torch.nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)  # Pass through residual blocks
        return x  # Return the encoded representation


class ResDynamicsNetwork(torch.nn.Module):
    def __init__(
        self,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
    ):
        """
        Initializes the Dynamics Network.

        This network predicts the next state and reward from the current state and action.

        Args:
        - num_blocks: Number of residual blocks.
        - num_channels: Number of input channels.
        - reduced_channels_reward: Number of output channels for the reward prediction.
        - fc_reward_layers: Structure of the fully connected layers for reward prediction.
        - full_support_size: Size of the support for the reward distribution.
        - block_output_size_reward: Size of the output for reward prediction.
        """
        super().__init__()
        # Convolutional layer followed by batch normalization
        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = torch.nn.BatchNorm2d(num_channels - 1)
        # Residual blocks
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels - 1) for _ in range(num_blocks)]
        )

        # 1x1 convolution to reduce output channels for reward prediction
        self.conv1x1_reward = torch.nn.Conv2d(
            num_channels - 1, reduced_channels_reward, 1
        )
        self.block_output_size_reward = block_output_size_reward  # Output size for rewards
        # Fully connected layer for reward prediction
        self.fc = mlp(
            self.block_output_size_reward,
            fc_reward_layers,
            full_support_size,
        )

    def forward(self, x):
        """
        Forward pass through the dynamics network.

        Args:
        - x: Input current state and action.

        Returns:
        - state: The updated (next) state.
        - reward: The predicted reward based on the current state and action.
        """
        x = self.conv(x)  # Apply convolution
        x = self.bn(x)  # Apply batch normalization
        x = torch.nn.functional.relu(x)  # ReLU activation

        for block in self.resblocks:
            x = block(x)  # Pass through residual blocks
        state = x  # Capture the state after processing

        x = self.conv1x1_reward(x)  # Reduce dimensions for reward prediction
        # Flatten for input to the fully connected layer
        x = x.view(-1, self.block_output_size_reward)
        reward = self.fc(x)  # Predict reward
        return state, reward  # Return both the state and reward


class ResPredictionNetwork(torch.nn.Module):
    def __init__(
        self,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
    ):
        """
        Initializes the Prediction Network.

        This network outputs the value and policy based on the given state.

        Args:
        - action_space_size: Size of the action space.
        - num_blocks: Number of residual blocks.
        - num_channels: Number of input channels.
        - reduced_channels_value: Reduced number of channels for value prediction.
        - reduced_channels_policy: Reduced number of channels for policy prediction.
        - fc_value_layers: Structure of the fully connected layers for value prediction.
        - fc_policy_layers: Structure of the fully connected layers for policy prediction.
        - full_support_size: Size of the support for the value distribution.
        - block_output_size_value: Size of the output for value prediction.
        - block_output_size_policy: Size of the output for policy prediction.
        """
        super().__init__()
        # Residual blocks
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        # 1x1 convolutions to project to reduced channels for value and policy
        self.conv1x1_value = torch.nn.Conv2d(
            num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = torch.nn.Conv2d(
            num_channels, reduced_channels_policy, 1)

        self.block_output_size_value = block_output_size_value  # Output size for value
        self.block_output_size_policy = block_output_size_policy  # Output size for policy

        # Fully connected layers for outputting the final value and policy
        self.fc_value = mlp(
            self.block_output_size_value, fc_value_layers, full_support_size
        )
        self.fc_policy = mlp(
            self.block_output_size_policy,
            fc_policy_layers,
            action_space_size,
        )

    def forward(self, x):
        """
        Forward pass through the prediction network.

        Args:
        - x: Input state representation.

        Returns:
        - policy: The predicted policy logits.
        - value: The predicted state value.
        """
        for block in self.resblocks:
            x = block(x)  # Pass through residual blocks

        value = self.conv1x1_value(x)  # Compute value
        policy = self.conv1x1_policy(x)  # Compute policy

        # Flatten outputs for fully connected layers
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)

        value = self.fc_value(value)  # Get final value predictions
        policy = self.fc_policy(policy)  # Get final policy predictions
        return policy, value  # Return both predictions
