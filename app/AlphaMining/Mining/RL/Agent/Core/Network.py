import math
import torch
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Tuple, Any

# from Mining.Config import *

from Mining.RL.Agent.Core.Util.Network_Util import *


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def initial_inference(self, observation: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ 
        Performs initial inference on the observation.

        Args:
        - observation (Tensor): Input observation, shape: (batch_size, (stacked_observations*2+1)*channels, height, width)

        Returns:
        - Tuple[Tensor, Tensor, Tensor, Tensor]: 
            - Encoded state (shape: (batch_size, encoding_size))
            - Policy logits (shape: (batch_size, action_space_size))
            - value logits (shape: (batch_size, full_support_size))
            - reward logits (shape: (batch_size, full_support_size))
        """
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ 
        Performs recurrent inference for subsequent timesteps.

        Args:
        - encoded_state (Tensor): Current encoded state, shape: (batch_size, encoding_size)
        - action (Tensor): Action taken, shape: (batch_size,)

        Returns:
        - Tuple[Tensor, Tensor, Tensor, Tensor]: 
            - Dynamic(Next encoded) state (shape: (batch_size, encoding_size))
            - Policy logits (shape: (batch_size, action_space_size))
            - value logits (shape: (batch_size, full_support_size))
            - reward logits (shape: (batch_size, full_support_size))
        """
        pass

    def get_weights(self) -> dict:
        """ Returns the model's weights as a CPU dictionary. """
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights: dict) -> None:
        """ Sets the model's weights from a CPU dictionary. """
        if weights is not None:
            self.load_state_dict(weights)


class Network:
    """
    Factory class to create policy networks for representation, dynamics, reward, value, and policy.
    Depending on the architecture specified in `config`, returns either an MLP or ResNet.
    """
    def __new__(cls, config: Any) -> AbstractNetwork:
        if config.arch == "mlp":
            return MLPNetwork(config)
        elif config.arch == "resnet":
            return ResNetwork(config)
        else:
            raise NotImplementedError(
                f'The network architecture {config.arch} is not implemented.'
            )


class MLPNetwork(AbstractNetwork):
    def __init__(self, config: Any) -> None:
        super().__init__()

        self.config = config

        # Size of action space
        # e.g., number of possible actions
        self.action_space_size: int = len(self.config.action_space)
        # e.g., full support for reward distribution
        self.full_support_size: int = 2 * self.config.support_size + 1

        # NOTE: when using torch.nn(e.g. mlp) to create model(Tensor), it is automatically created with requires_grad=True
        #       which means that a computation graph is established and latter gradient results from loss
        #       can be directly sent to model using backward()

        # Representation network input size explained below
        self.representation_network = torch.nn.DataParallel(
            mlp(
                self.config.observation_shape[0] *
                self.config.observation_shape[1] *
                self.config.observation_shape[2] *
                (self.config.stacked_observations + 1) +
                # Input features size
                self.config.observation_shape[1] *
                self.config.observation_shape[2] *
                self.config.stacked_observations,
                self.config.mlp_representation_layers,  # Definition of hidden layers in MLP
                self.config.encoding_size,  # Output encoding size
            )
        )

        # Dynamics network: Predicts next state encoding from current state and action
        self.dynamics_encoded_state_network = torch.nn.DataParallel(
            mlp(
                # Input size combines current state encodings with action
                self.config.encoding_size + self.action_space_size,
                self.config.mlp_dynamics_layers,  # Architecture of dynamics network
                self.config.encoding_size  # Output encoding size
            )
        )

        # Reward network predicts the reward based on the next encoded state
        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(self.config.encoding_size,
                # Full support output for reward
                self.config.mlp_reward_layers, self.full_support_size)
        )

        # Policy network predicts action probabilities
        self.prediction_policy_network = torch.nn.DataParallel(
            mlp(self.config.encoding_size,
                # Outputs action probabilities
                self.config.mlp_policy_layers, self.action_space_size)
        )

        # Value network predicts the estimated value of the current encoded state
        self.prediction_value_network = torch.nn.DataParallel(
            mlp(self.config.encoding_size,
                # Full support output for estimated values
                self.config.mlp_value_layers, self.full_support_size)
        )

    def prediction(self, encoded_state: Tensor) -> Tuple[Tensor, Tensor]:
        """ 
        Given an encoded state, outputs predicted policy logits and value.

        Args:
        - encoded_state (Tensor): The latent representation of the state, shape: (batch_size, encoding_size)

        Returns:
        - Tuple[Tensor, Tensor]: 
            - Policy logits (shape: (batch_size, action_space_size))
            - Value estimate (shape: (batch_size, full_support_size))
        """
        policy_logits: Tensor = self.prediction_policy_network(encoded_state)
        value_logits: Tensor = self.prediction_value_network(encoded_state)
        return policy_logits, value_logits

    def representation(self, observation: Tensor) -> Tensor:
        """ 
        Encodes the observation into a latent state representation.

        Args:
        - observation (Tensor): Input observation, shape: (batch_size, (stacked_observations*2+1)*channels, height, width)

        Returns:
        - Tensor: Normalized encoded representation of the state, shape: (batch_size, encoding_size)
        """
        encoded_state: Tensor = self.representation_network(
            # Flatten input observation
            observation.view(observation.shape[0], -1)
        )

        # Normalize the encoded state for stability
        min_encoded_state: Tensor = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state: Tensor = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state: Tensor = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state <1e-5] += \
            1e-5  # Avoid division by zero
        encoded_state_normalized: Tensor = (
            encoded_state - min_encoded_state) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """ 
        Predicts the next state and reward given the current encoded state and action.

        Args:
        - encoded_state (Tensor): Current encoded state, shape: (batch_size, encoding_size)
        - action (Tensor): Action taken, shape: (batch_size,)

        Returns:
        - Tuple[Tensor, Tensor]: 
            - Next encoded state (normalized) (shape: (batch_size, encoding_size))
            - Predicted reward (shape: (batch_size, full_support_size))
        """
        action_one_hot: Tensor = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()  # Create a one-hot tensor for actions
        )
        # One-hot encode the action
        action_one_hot.scatter_(1, action.long(), 1.0)
        # Concatenate encoded state and action
        x: Tensor = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state: Tensor = self.dynamics_encoded_state_network(
            x)  # Get next encoded state
        reward: Tensor = self.dynamics_reward_network(
            next_encoded_state)  # Get reward

        # Normalize the next encoded state for stability
        min_next_encoded_state: Tensor = next_encoded_state.min(1, keepdim=True)[
            0]
        max_next_encoded_state: Tensor = next_encoded_state.max(1, keepdim=True)[
            0]
        scale_next_encoded_state: Tensor = max_next_encoded_state - min_next_encoded_state
        # Avoid division by zero
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized: Tensor = (
            next_encoded_state - min_next_encoded_state) / scale_next_encoded_state
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ 
        Performs initial inference on the observation.

        Args:
        - observation (Tensor): Input observation, shape: (batch_size, (stacked_observations*2+1)*channels, height, width)

        Returns:
        - Tuple[Tensor, Tensor, Tensor, Tensor]: 
            - Encoded state (shape: (batch_size, encoding_size))
            - Policy logits (shape: (batch_size, action_space_size))
            - value logits (shape: (batch_size, full_support_size))
            - reward logits (shape: (batch_size, full_support_size))
        """
        encoded_state: Tensor = self.representation(
            observation)  # Encodes the observation
        policy_logits, value_logits = self.prediction(
            encoded_state)  # Gets policy and value predictions

        # Initialize reward distribution centered around zero
        # [..., -inf, -inf, 0., -inf, -inf, ...]
        reward_logits: Tensor = torch.log(
            (torch.zeros(1, self.full_support_size)
             .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
             .repeat(len(observation), 1)
             .to(observation.device)) + 1e-5
        )
        return encoded_state, policy_logits, value_logits, reward_logits

    def recurrent_inference(self, encoded_state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ 
        Performs recurrent inference for subsequent timesteps.

        Args:
        - encoded_state (Tensor): Current encoded state, shape: (batch_size, encoding_size)
        - action (Tensor): Action taken, shape: (batch_size,)

        Returns:
        - Tuple[Tensor, Tensor, Tensor, Tensor]: 
            - Dynamic(Next encoded) state (shape: (batch_size, encoding_size))
            - Policy logits (shape: (batch_size, action_space_size))
            - value logits (shape: (batch_size, full_support_size))
            - reward logits (shape: (batch_size, full_support_size))
        """
        next_encoded_state, reward_logits = self.dynamics(
            encoded_state, action)  # Get next state and reward
        policy_logits, value_logits = self.prediction(
            next_encoded_state)  # Get predictions for next state
        return next_encoded_state, policy_logits, value_logits, reward_logits


class ResNetwork(AbstractNetwork):
    def __init__(self, config: Any) -> None:
        """ Initializes the Residual Network.

        This network handles representation, dynamics, and predictions for reinforcement learning tasks.
        """
        super().__init__()

        self.config = config

        # Size of action space
        # e.g., number of available actions
        self.action_space_size: int = len(self.config.action_space)
        # Full support size for reward distribution
        self.full_support_size: int = 2 * self.config.support_size + \
            1  # e.g., full support for reward distribution

        # Calculate output sizes for reward, value, and policy networks based on whether downsampling is performed
        block_output_size_reward: int = (
            (self.config.reduced_channels_reward *
             math.ceil(self.config.observation_shape[1] / 16) *
             math.ceil(self.config.observation_shape[2] / 16))
            if self.config.downsample
            else (self.config.reduced_channels_reward * self.config.observation_shape[1] *
                  self.config.observation_shape[2])  # when no downsampling
        )

        block_output_size_value: int = (
            (self.config.reduced_channels_value *
             math.ceil(self.config.observation_shape[1] / 16) *
             math.ceil(self.config.observation_shape[2] / 16))
            if self.config.downsample
            else (self.config.reduced_channels_value * self.config.observation_shape[1] *
                  self.config.observation_shape[2])  # when no downsampling
        )

        block_output_size_policy: int = (
            (self.config.reduced_channels_policy *
             math.ceil(self.config.observation_shape[1] / 16) *
             math.ceil(self.config.observation_shape[2] / 16))
            if self.config.downsample
            else (self.config.reduced_channels_policy * self.config.observation_shape[1] *
                  self.config.observation_shape[2])  # when no downsampling
        )

        # Initialize the representation network
        self.representation_network = torch.nn.DataParallel(
            ResRepresentationNetwork(
                self.config.observation_shape,  # Shape of input observations
                self.config.stacked_observations,  # Number of stacked observations
                self.config.num_blocks,  # Number of residual blocks
                self.config.num_channels,  # Number of channels for convolutional layers
                self.config.downsample,  # Downsampling option
            )
        )

        # Initialize dynamics network
        self.dynamics_network = torch.nn.DataParallel(
            ResDynamicsNetwork(
                self.config.num_blocks,  # Number of residual blocks
                self.config.num_channels + 1,  # Channels for input to dynamics network
                self.config.reduced_channels_reward,  # Reduced channels for reward prediction
                self.config.res_reward_layers,  # MLP layers for reward prediction
                self.full_support_size,  # Full support size for rewards
                block_output_size_reward,  # Output size for reward predictions
            )
        )

        # Initialize prediction network
        self.prediction_network = torch.nn.DataParallel(
            ResPredictionNetwork(
                self.action_space_size,  # Size of action space
                self.config.num_blocks,  # Number of residual blocks
                self.config.num_channels,  # Input channels
                self.config.reduced_channels_value,  # Channels for value output
                self.config.reduced_channels_policy,  # Channels for policy output
                self.config.res_value_layers,  # MLP layers for value predictions
                self.config.res_policy_layers,  # MLP layers for policy predictions
                self.full_support_size,  # Full support size for values
                block_output_size_value,  # Output size for value predictions
                block_output_size_policy,  # Output size for policy predictions
            )
        )

    def prediction(self, encoded_state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Generates policy and value predictions from the encoded state.

        Args:
        - encoded_state (Tensor): Input state representation, shape: (batch_size, (stacked_observations*2+1)*channels, height, width)

        Returns:
        - Tuple[Tensor, Tensor]: 
            - Predicted policy logits (shape: (batch_size, action_space_size))
            - Predicted value (shape: (batch_size, full_support_size))
        """
        policy_logits, value_logits = self.prediction_network(encoded_state)
        return policy_logits, value_logits

    def representation(self, observation: Tensor) -> Tensor:
        """
        Encodes the observation into a latent state.

        Args:
        - observation (Tensor): Input observation, shape: (batch_size, (stacked_observations*2+1)*channels, height, width)

        Returns:
        - Tensor: Normalized encoded state representation, shape: (batch_size, (stacked_observations*2+1)*channels, height, width)
        """
        encoded_state: Tensor = self.representation_network(
            observation)  # Pass observation through representation network

        # Scale encoded state between [0, 1]
        min_encoded_state: Tensor = (
            encoded_state.view(-1, encoded_state.shape[1],
                               encoded_state.shape[2] * encoded_state.shape[3]).min(2, keepdim=True)[0].unsqueeze(-1)
        )
        max_encoded_state: Tensor = (
            encoded_state.view(-1, encoded_state.shape[1],
                               encoded_state.shape[2] * encoded_state.shape[3]).max(2, keepdim=True)[0].unsqueeze(-1)
        )
        scale_encoded_state: Tensor = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state <
                            1e-5] += 1e-5  # Avoid division by zero
        encoded_state_normalized: Tensor = (
            encoded_state - min_encoded_state) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Models the dynamics of the environment, predicting the next state and reward.

        Args:
        - encoded_state (Tensor): Current encoded state, shape: (batch_size, (stacked_observations*2+1)*channels, height, width)
        - action (Tensor): Action taken, shape: (batch_size,)

        Returns:
        - Tuple[Tensor, Tensor]: 
            - Next encoded state (normalized) (shape: (batch_size, (stacked_observations*2+1)*channels, height, width))
            - Predicted reward (shapes depend on the architecture)
        """
        # Stack encoded_state with a one-hot encoded representation of the action
        action_one_hot: Tensor = (
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
        action_one_hot = action[:, :, None, None] * \
            action_one_hot / self.action_space_size

        # Concatenate state and action
        x: Tensor = torch.cat((encoded_state, action_one_hot), dim=1)
        # Predict next state and reward
        next_encoded_state, reward = self.dynamics_network(x)

        # Scale next encoded state between [0, 1]
        # Scale next encoded state between [0, 1] for normalization
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        # Avoid division by zero
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state  # Normalize the next encoded state

        # Return both the normalized next state and the reward
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Performs initial inference from the observation.

        Args:
        - observation (Tensor): Input observation, shape: (batch_size, (stacked_observations*2+1)*channels, height, width)

        Returns:
        - Tuple[Tensor, Tensor, Tensor, Tensor]:
            - Encoded state (shape: (batch_size, (stacked_observations*2+1)*channels, height, width))
            - Policy logits (shape: (batch_size, action_space_size))
            - value logits (shape: (batch_size, full_support_size))
            - reward logits (shape: (batch_size, full_support_size))
        """
        encoded_state: Tensor = self.representation(
            observation)  # Encode the observation
        policy_logits, value_logits = self.prediction(
            encoded_state)  # Get policy logits and value
        # Reward is initialized for consistency
        reward_logits: Tensor = torch.log(
            (torch.zeros(1, self.full_support_size)
             .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
             .repeat(len(observation), 1)
             .to(observation.device))
        )
        return encoded_state, policy_logits, value_logits, reward_logits

    def recurrent_inference(self, encoded_state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Performs recurrent inference given the encoded state and action.

        Args:
        - encoded_state (Tensor): Current encoded state, shape: (batch_size, (stacked_observations*2+1)*channels, height, width)
        - action (Tensor): Action taken, shape: (batch_size,)

        Returns:
        - Tuple[Tensor, Tensor, Tensor, Tensor]: 
            - Dynamic(Next encoded) state (shape: (batch_size, (stacked_observations*2+1)*channels, height, width))
            - Policy logits (shape: (batch_size, action_space_size))
            - value logits (shape: (batch_size, full_support_size))
            - reward logits (shape: (batch_size, full_support_size))
        """
        next_encoded_state, reward_logits = self.dynamics(
            encoded_state, action)  # Predict next state and reward
        # Get predictions for the next state
        policy_logits, value_logits = self.prediction(next_encoded_state)
        return next_encoded_state, policy_logits, value_logits, reward_logits


class ResRepresentationNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape: Tuple[int, int, int],
        stacked_observations: int,
        num_blocks: int,
        num_channels: int,
        downsample: str,
    ) -> None:
        """ 
        Initializes the Representation Network.

        This network encodes observations into a latent representation.

        Args:
        - observation_shape (Tuple[int, int, int]): Shape of the input observations ((stacked_observations*2+1)*channels, height, width).
        - stacked_observations (int): Number of stacked observations.
        - num_blocks (int): Number of residual blocks in the network.
        - num_channels (int): Number of channels for convolutional layers.
        - downsample (str): Whether to downsample the input, options include 'resnet' or 'CNN'.
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

        # Initial convolution layer followed by normalization
        self.conv = conv3x3(
            observation_shape[0] *
            (stacked_observations + 1) + stacked_observations,
            num_channels,
        )
        # Batch normalization after convolution
        self.bn = torch.nn.BatchNorm2d(num_channels)
        # Residual blocks for deeper representation learning
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """ 
        Forward pass through the representation network.

        Args:
        - x (Tensor): Input observations to encode, shape: (batch_size, (stacked_observations*2+1)*channels, height, width)

        Returns:
        - Tensor: The encoded representation, shape: (batch_size, (stacked_observations*2+1)*channels, height, width)
        """
        if self.downsample:
            x = self.downsample_net(x)  # Apply downsampling
        else:
            # If not downsampling, apply convolution and normalization
            x = self.conv(x)
            x = self.bn(x)
            x = torch.nn.functional.relu(x)

        # Pass through all residual blocks
        for block in self.resblocks:
            x = block(x)
        return x  # Return the encoded representation


class ResDynamicsNetwork(torch.nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_channels: int,
        reduced_channels_reward: int,
        fc_reward_layers: int,
        full_support_size: int,
        block_output_size_reward: int,
    ) -> None:
        """ 
        Initializes the Dynamics Network.

        This network predicts the next state and reward from the current state and action.

        Args:
        - num_blocks (int): Number of residual blocks.
        - num_channels (int): Number of input channels.
        - reduced_channels_reward (int): Number of output channels for the reward prediction.
        - fc_reward_layers (int): Structure of fully connected layers for reward prediction.
        - full_support_size (int): Size of the support for the reward distribution.
        - block_output_size_reward (int): Size of the output for reward prediction.
        """
        super().__init__()
        # Convolutional layer followed by batch normalization
        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = torch.nn.BatchNorm2d(num_channels - 1)
        # Residual blocks
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels - 1) for _ in range(num_blocks)]
        )

        # 1x1 convolution to reduce output channels for the reward prediction
        self.conv1x1_reward = torch.nn.Conv2d(
            num_channels - 1, reduced_channels_reward, 1
        )
        self.block_output_size_reward: int = block_output_size_reward  # Output size for rewards
        # Fully connected layer for reward prediction
        self.fc = mlp(
            self.block_output_size_reward,
            fc_reward_layers,
            full_support_size,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """ 
        Forward pass through the dynamics network.

        Args:
        - x (Tensor): Input current state and action, shape: (batch_size, (stacked_observations*2+1)*channels, height, width)

        Returns:
        - Tuple[Tensor, Tensor]: 
            - Updated (next) state (shape: (batch_size, (stacked_observations*2+1)*channels, height, width))
            - Predicted reward (shape: (batch_size, full_support_size))
        """
        x = self.conv(x)  # Apply convolution
        x = self.bn(x)  # Batch normalization
        x = torch.nn.functional.relu(x)  # Activation function

        for block in self.resblocks:
            x = block(x)  # Pass through residual blocks
        state: Tensor = x  # Capture the state after processing

        x = self.conv1x1_reward(x)  # Reduce dimensions for reward prediction
        # Flatten for input to the fully connected layer
        x = x.view(-1, self.block_output_size_reward)
        reward: Tensor = self.fc(x)  # Predict reward
        return state, reward  # Return both the state and reward


class ResPredictionNetwork(torch.nn.Module):
    def __init__(
        self,
        action_space_size: int,
        num_blocks: int,
        num_channels: int,
        reduced_channels_value: int,
        reduced_channels_policy: int,
        fc_value_layers: int,
        fc_policy_layers: int,
        full_support_size: int,
        block_output_size_value: int,
        block_output_size_policy: int,
    ) -> None:
        """ 
        Initializes the Prediction Network.

        This network outputs the value and policy based on the given state.

        Args:
        - action_space_size (int): Size of the action space.
        - num_blocks (int): Number of residual blocks.
        - num_channels (int): Number of input channels.
        - reduced_channels_value (int): Reduced number of channels for value prediction.
        - reduced_channels_policy (int): Reduced number of channels for policy prediction.
        - fc_value_layers (int): Structure of the fully connected layers for value prediction.
        - fc_policy_layers (int): Structure of the fully connected layers for policy prediction.
        - full_support_size (int): Size of the support for the value distribution.
        - block_output_size_value (int): Size of the output for value prediction.
        - block_output_size_policy (int): Size of the output for policy prediction.
        """
        super().__init__()
        # Residual blocks for network depth
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        # 1x1 convolutions to project to reduced channels for value and policy
        self.conv1x1_value: torch.nn.Conv2d = torch.nn.Conv2d(
            num_channels, reduced_channels_value, 1)
        self.conv1x1_policy: torch.nn.Conv2d = torch.nn.Conv2d(
            num_channels, reduced_channels_policy, 1)

        self.block_output_size_value: int = block_output_size_value  # Output size for value
        self.block_output_size_policy: int = block_output_size_policy  # Output size for policy

        # Fully connected layers for outputting the final value and policy
        self.fc_value = mlp(
            self.block_output_size_value, fc_value_layers, full_support_size
        )
        self.fc_policy = mlp(
            self.block_output_size_policy,
            fc_policy_layers,
            action_space_size,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """ 
        Forward pass through the prediction network.

        Args:
        - x (Tensor): Input state representation, shape: (batch_size, (stacked_observations*2+1)*channels, height, width)

        Returns:
        - Tuple[Tensor, Tensor]: 
            - Predicted policy logits (shape: (batch_size, action_space_size))
            - Predicted value (shape: (batch_size, full_support_size))
        """
        for block in self.resblocks:
            x = block(x)  # Pass through residual blocks

        value: Tensor = self.conv1x1_value(x)  # Compute value
        policy: Tensor = self.conv1x1_policy(x)  # Compute policy

        # Flatten outputs for fully connected layers
        # Flatten for value predictions
        value = value.view(-1, self.block_output_size_value)
        # Flatten for policy predictions
        policy = policy.view(-1, self.block_output_size_policy)

        value = self.fc_value(value)  # Get final value predictions
        policy = self.fc_policy(policy)  # Get final policy predictions
        return policy, value  # Return both predictions
