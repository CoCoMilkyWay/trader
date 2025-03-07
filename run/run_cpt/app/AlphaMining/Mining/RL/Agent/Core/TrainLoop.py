import ray
import copy
import time
import torch
import numpy as np
from torch import Tensor
from numpy.typing import NDArray

from typing import Any, Tuple, Dict, List

# from Mining.Config import *
from dotmap import DotMap

from Mining.RL.Agent.Core.Util.Network_Util import \
    scalar_to_support, support_to_scalar, dict_to_cpu
from Mining.RL.Agent.Core.Network import Network  # Unified model: R, D, O, V, π


@ray.remote
class TrainLoop:
    """
    Trainer runs in a dedicated thread (or process) to update the unified model’s parameters.

    It continuously:
      - Samples batches (Trajectory) from the replay buffer.
      - Unrolls the model for K steps.
      - Computes the loss (reward, value, and policy losses).
      - Updates the model weights (using gradient descent with the chosen optimizer).
      - Saves updated weights and optimizer state to the shared checkpoint storage.
    """

    def __init__(self, config, initial_checkpoint):
        self.config = DotMap(config)

        # Fix random generator seeds for reproducibility.
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the unified model (which includes R, D, O, V, and π).
        self.model = Network(self.config)
        # Load initial weights from the checkpoint.
        init_weights = copy.deepcopy(
            ray.get(initial_checkpoint.get_info.remote("weights")))
        self.model.set_weights(init_weights)  # type: ignore
        device = torch.device("cuda" if self.config.train_on_gpu else "cpu")
        self.model.to(device)
        self.model.train()  # Set model to training mode.

        # Track the number of training steps performed.
        self.trained_steps: int = ray.get(initial_checkpoint.get_info.remote(
            "num_trained_steps"))

        # # Warn if not training on GPU.
        # if "cuda" not in str(next(self.model.parameters()).device):
        #     print("You are not training on GPU.\n")

        # Initialize the optimizer (using either SGD or Adam as specified in config).
        if self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr_init,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. Change the optimizer manually in trainer.py."
            )

        # Optionally load previous optimizer state from checkpoint.
        optimizer_state = ray.get(
            initial_checkpoint.get_info.remote("optimizer_state"))
        if optimizer_state is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(copy.deepcopy(optimizer_state[0]))

        print(f"TrainLoop inited on {device}")

    def train_loop(self, checkpoint, replay_buffer):
        """
        Main training loop.

        This loop waits until the replay buffer has enough data, then continuously:
          - Samples a batch of Trajectory.
          - Updates the learning rate.
          - Performs a training step (train_step).
          - Optionally updates priorities for prioritized experience replay.
          - Saves the model and optimizer state to the checkpoint storage.
          - Enforces a training-to-self-play ratio.

        Set learning_rate
        Set number_of_epochs
        Set batch_size

        Load training_data
        Load validation_data

        for epoch in range(number_of_epochs):
            Shuffle training_data
            for batch in get_batches(training_data, batch_size):
                inputs, targets = batch

                // Forward pass
                predictions = neural_network.forward(inputs)

                // Calculate loss
                loss = calculate_loss(predictions, targets)

                // Backward pass
                gradients = neural_network.backward(loss)

                // Update weights
                for layer in neural_network.layers:
                    layer.weights -= learning_rate * layer.gradients
                    layer.biases -= learning_rate * layer.bias_gradients

            // Optionally validate the model
            if epoch % validation_interval == 0:
                val_loss, val_accuracy = validate(
                    neural_network, validation_data)
                Print "Epoch:", epoch, "Validation Loss:", val_loss, "Validation Accuracy:", val_accuracy
        """
        # Wait until at least one game has been played.
        while True:
            num_played_games = int(
                ray.get(checkpoint.get_info.remote("num_played_games")))
            if num_played_games < 1:
                time.sleep(0.1)
            else:
                break

        next_batch = replay_buffer.get_batch.remote()
        # Continue training until reaching the maximum training steps or termination signal.
        while self.trained_steps < self.config.max_training_steps and not ray.get(checkpoint.get_info.remote("terminate")):
            print('train new batch')
            index_batch, batch = ray.get(next_batch)
            self.update_learning_rate(self.trained_steps)

            # Perform one training step using the sampled batch.
            total_loss, policy_loss, value_loss, reward_loss = \
                self.train_step(batch)

            print(f"                                                    "
                  f"{policy_loss:.2f}, "
                  f"{value_loss:.2f}, "
                  f"{reward_loss:.2f}, ")

            # Save training statistics to the checkpoint.
            checkpoint.set_info.remote({
                "num_trained_steps": self.trained_steps,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "total_loss": total_loss,
                "value_loss": value_loss,
                "reward_loss": reward_loss,
                "policy_loss": policy_loss,
            })

            # Save updated model weights and optimizer state periodically.
            if self.trained_steps % self.config.update_interval == 0:
                checkpoint.set_info.remote({
                    "weights": copy.deepcopy(self.model.get_weights()),
                    "optimizer_state": copy.deepcopy(dict_to_cpu(self.optimizer.state_dict())),
                })
                if self.config.save_model:
                    checkpoint.save_checkpoint.remote()

            # Manage the self-play / training ratio.
            # normally you should always be training
            # but training the same data from replay buffer too many times could be a problem
            while True:
                num_played_steps = int(
                    ray.get(checkpoint.get_info.remote("num_played_steps")))
                terminate = ray.get(
                    checkpoint.get_info.remote("terminate"))

                if self.trained_steps < num_played_steps * self.config.ratio_train_play or \
                        self.trained_steps > self.config.max_training_steps or terminate:
                    break

                time.sleep(0.5)

    def train_step(self, batch: Tuple[List[NDArray], List[List[List[float]]], List[List[float]], List[List[int]], List[List[float]], List[float], List[List[float]]]) -> Tuple[float, float, float, float]:
        """
        Perform one training step (one gradient update).

        This method includes:
          - Unpacking the batch (observations, policies, values, actions, rewards).
          - Converting targets to the support representation.
          - Unrolling the model for K steps (initial inference plus recurrent unrolls).
          - Computing losses (value, reward, policy losses).
          - Backpropagating and updating model weights.

        NOTE: why we redo inference here?
            1. it is not a full MCTS, it is fast
            2. just in case the model params changed, which would make gradient calculation inaccurate
        """
        # Unpack the batch
        (
            observation_feature,  # [batch_size, arr[chans, height, width]]
            policy_target,        # [batch_size, unroll_steps+1, num_actions]
            value_target,         # [batch_size, unroll_steps+1]
            action_feature,       # [batch_size, unroll_steps+1]
            reward_target,        # [batch_size, unroll_steps+1]
            weight_batch,         # [batch_size] (if using PER)
            gradient_scale_batch,  # [batch_size, unroll_steps+1]
        ) = batch

        # Move to device
        device = next(self.model.parameters()).device

        # Convert to Tensors
        observation_feature = torch.tensor(  # converting List[NDArray] to Tensor is extremely slow
            np.array(observation_feature), dtype=torch.float32).to(device)
        policy_target = torch.tensor(
            policy_target, dtype=torch.float32).to(device)
        value_target = torch.tensor(
            value_target, dtype=torch.float32).to(device)
        action_feature = torch.tensor(
            action_feature, dtype=torch.int).to(device).unsqueeze(-1)
        reward_target = torch.tensor(
            reward_target, dtype=torch.float32).to(device)
        # if self.config.PER:
        weight_batch = torch.tensor(
            weight_batch, dtype=torch.float32).to(device)
        gradient_scale_batch = torch.tensor(
            gradient_scale_batch, dtype=torch.float32).to(device)

        # Convert scalar targets to support representation
        value_target = scalar_to_support(
            value_target, self.config.support_size)
        reward_target = scalar_to_support(
            reward_target, self.config.support_size)

        # print(observation_feature.shape,policy_target.shape,value_target.shape,action_feature.shape,reward_target.shape,weight_batch.shape,gradient_scale_batch.shape)

        # observation_feature,  shape: [batch_size, array[channels, height, width]]
        # policy_target,        shape: [batch_size, num_unroll_steps + 1, num_actions]
        # value_target,         shape: [batch_size, num_unroll_steps + 1, 2 * support_size + 1]
        # action_feature,       shape: [batch_size, num_unroll_steps + 1, 1] (unsqueeze)
        # reward_target,        shape: [batch_size, num_unroll_steps + 1, 2 * support_size + 1]
        # weight_batch,         shape: [batch_size] (if using PER)
        # gradient_scale_batch, shape: [batch_size, num_unroll_steps + 1]

        # Generate predictions by unrolling the model
        repr_logits, policy_logits, value_logits, reward_logits = \
            self.model.initial_inference(observation_feature)
        predictions = [(policy_logits, value_logits, reward_logits)]
        for i in range(1, action_feature.shape[1]):
            repr_logits, policy_logits, value_logits, reward_logits = self.model.recurrent_inference(
                repr_logits, action_feature[:, i])
            # Scale the gradient at the start of the dynamics function (See paper appendix Training)
            repr_logits.register_hook(lambda grad: grad * 0.5)
            predictions.append((policy_logits, value_logits, reward_logits))

        # Compute losses over the unrolled steps
        policy_loss, value_loss, reward_loss = self.compute_losses(
            predictions, policy_target, value_target, reward_target, gradient_scale_batch)

        # Scale the overall loss (recommended scaling factor)
        loss = (policy_loss +
                value_loss * float(self.config.value_loss_weight) +
                reward_loss)
        if self.config.PER:
            loss *= weight_batch
        loss = loss.mean()

        # NOTE: even we are only calculating gradient from 3 target network outputs,
        #       we have actually already established computation graphs to all 5 models and are training them all at once
        #
        #   representation:   θ_R ← θ_R - α * ∇_θ_R L_total
        #   dynamic:          θ_D ← θ_D - α * ∇_θ_D L_total
        #   policy:           θ_π ← θ_π - α * ∇_θ_π L_policy
        #   value:            θ_V ← θ_V - α * ∇_θ_V L_value
        #   reward:           θ_O ← θ_O - α * ∇_θ_O L_reward

        # --- Backpropagation and optimization ---
        self.optimizer.zero_grad()
        loss.backward()  # dy/dx (y is scalar) sent to tensor x in model (using its memorized computation graph)
        self.optimizer.step()  # optimize x using loaded new gradient
        self.trained_steps += 1

        return loss.item(), policy_loss.mean().item(), value_loss.mean().item(), reward_loss.mean().item()

    def compute_losses(self, predictions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                       target_policy: torch.Tensor,
                       target_value: torch.Tensor,
                       target_reward: torch.Tensor,
                       gradient_scale_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute value, reward, and policy losses from the model predictions.

        This helper function calculates individual losses for each step and accumulates them.

        Returns:
            Tuple containing total value loss, total reward loss, and total policy loss.
        """
        # Compute loss and priorities for remaining unroll steps
        for i in range(0, len(predictions)):
            policy_logits, value_logits, reward_logits = predictions[i]
            current_policy_loss = self.compute_loss(
                policy_logits, target_policy[:, i])
            current_value_loss = self.compute_loss(
                value_logits, target_value[:, i])
            current_reward_loss = self.compute_loss(
                reward_logits, target_reward[:, i])

            if i != 0:
                # Scale losses by the gradient scale for this step
                current_value_loss.register_hook(
                    lambda grad: grad * gradient_scale_batch[:, i])
                current_reward_loss.register_hook(
                    lambda grad: grad * gradient_scale_batch[:, i])
                current_policy_loss.register_hook(
                    lambda grad: grad * gradient_scale_batch[:, i])

            if i == 0:
                policy_loss = current_policy_loss
                value_loss = current_value_loss
                reward_loss = current_reward_loss
            else:
                policy_loss += current_policy_loss
                value_loss += current_value_loss
                reward_loss += current_reward_loss

        return policy_loss, value_loss, reward_loss

    @staticmethod
    def compute_loss(prediction: Tensor, target: Tensor):
        """
        Compute the loss for a training step.

        For each unroll step, losses are computed as cross-entropy between the predictions
        and their corresponding targets.

        Returns:
            loss (each as tensors over the batch).
        """
        # Cross-entropy loss is used (better convergence than MSE in many cases).
        loss = (-target * torch.nn.LogSoftmax(dim=1)(prediction)).sum(1)
        return loss

    def update_learning_rate(self, trained_steps: int):
        """
        Dynamically update the learning rate based on training progress.
        Uses an exponential decay schedule:
          learning_rate = lr_init * (lr_decay_rate)^(trained_steps / lr_decay_steps)
        """
        learning_rate = self.config.lr_init * self.config.lr_decay_rate ** \
            (trained_steps / self.config.lr_decay_steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate
