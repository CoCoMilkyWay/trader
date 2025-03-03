import copy
import time

import numpy as np
import ray
import torch

from Mining.Config import *

from Mining.RL.Agent.Core.Util.Network_Util import \
    scalar_to_support, support_to_scalar, dict_to_cpu
from Mining.RL.Agent.Core.Network import Network  # Unified model: R, D, O, V, π


@ray.remote
class TrainLoop:
    """
    Trainer runs in a dedicated thread (or process) to update the unified model’s parameters.

    It continuously:
      - Samples batches (trajectories) from the replay buffer.
      - Unrolls the model for K steps.
      - Computes the loss (reward, value, and policy losses).
      - Updates the model weights (using gradient descent with the chosen optimizer).
      - Saves updated weights and optimizer state to the shared checkpoint storage.
    """

    def __init__(self, initial_checkpoint):
        # Fix random generator seeds for reproducibility.
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the unified model (which includes R, D, O, V, and π).
        self.model = Network()
        # Load initial weights from the checkpoint.
        init_weights = copy.deepcopy(
            initial_checkpoint.get_info.remote("weights"))
        self.model.set_weights(init_weights)
        device = torch.device("cuda" if train_on_gpu else "cpu")
        self.model.to(device)
        self.model.train()  # Set model to training mode.

        # Track the number of training steps performed.
        self.trained_steps = initial_checkpoint.get_info.remote(
            "num_trained_steps")

        # Warn if not training on GPU.
        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on GPU.\n")

        # Initialize the optimizer (using either SGD or Adam as specified in config).
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr_init,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr_init,
                weight_decay=weight_decay,
            )
        else:
            raise NotImplementedError(
                f"{optimizer} is not implemented. Change the optimizer manually in trainer.py."
            )

        # Optionally load previous optimizer state from checkpoint.
        optimizer_state = initial_checkpoint.get_info.remote("optimizer_state")
        if optimizer_state is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(copy.deepcopy(optimizer_state))

    def train_loop(self, checkpoint, replay_buffer):
        """
        Main training loop.

        This loop waits until the replay buffer has enough data, then continuously:
          - Samples a batch of trajectories.
          - Updates the learning rate.
          - Performs a training step (train_step).
          - Optionally updates priorities for prioritized experience replay.
          - Saves the model and optimizer state to the checkpoint storage.
          - Enforces a training-to-self-play ratio.
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
        while self.trained_steps < max_training_steps and not ray.get(checkpoint.get_info.remote("terminate")):
            index_batch, batch = ray.get(next_batch)
            # Prepare the next batch.
            next_batch = replay_buffer.get_batch.remote()
            self.update_learning_rate()

            # Perform one training step using the sampled batch.
            priorities, total_loss, value_loss, reward_loss, policy_loss = self.train_step(
                batch)

            if PER:
                # Update priorities in the replay buffer based on TD error (per Prioritized Experience Replay).
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save updated model weights and optimizer state periodically.
            if self.trained_steps % update_interval == 0:
                checkpoint.set_info.remote({
                    "weights": copy.deepcopy(self.model.get_weights()),
                    "optimizer_state": copy.deepcopy(dict_to_cpu(self.optimizer.state_dict())),
                })
                if save_model:
                    checkpoint.save_checkpoint.remote()

            # Save training statistics to the checkpoint.
            checkpoint.set_info.remote({
                "num_trained_steps": self.trained_steps,
                "learning_rate": self.optimizer.param_groups[0]["learning_rate"],
                "total_loss": total_loss,
                "value_loss": value_loss,
                "reward_loss": reward_loss,
                "policy_loss": policy_loss,
            })

            # Manage the self-play / training ratio.
            if ratio_train_play:
                while True:
                    num_played_steps = int(
                        ray.get(checkpoint.get_info.remote("num_played_steps")))
                    terminate = ray.get(
                        checkpoint.get_info.remote("terminate"))

                    if self.trained_steps/max(1, num_played_steps) < ratio_train_play or \
                            self.trained_steps > max_training_steps or terminate:
                        break

                    time.sleep(0.5)

    def train_step(self, batch):
        """
        Perform one training step (one gradient update).

        This corresponds to:
          - Unpacking the batch (observations, actions, target values, rewards, and target policies).
          - Converting targets to the support representation.
          - Unrolling the model for K steps (initial inference plus recurrent unrolls).
          - Computing loss (value, reward, policy losses).
          - Backpropagating and updating the model weights.
          - Computing priorities for prioritized replay.
        """
        (observation_batch, action_batch, target_value, target_reward, target_policy,
         weight_batch, gradient_scale_batch) = batch

        # Convert target values to scalars for computing priorities.
        target_value_scalar = np.array(target_value, dtype="float32")
        priorities = np.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device
        if PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)
        observation_batch = torch.tensor(
            np.array(observation_batch)).float().to(device)
        action_batch = torch.tensor(
            action_batch).long().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        gradient_scale_batch = torch.tensor(
            gradient_scale_batch).float().to(device)
        # Shapes:
        # observation_batch: [batch, channels, height, width]
        # action_batch: [batch, num_unroll_steps+1, 1]
        # target_value, target_reward: [batch, num_unroll_steps+1]
        # target_policy: [batch, num_unroll_steps+1, len(action_space)]
        # gradient_scale_batch: [batch, num_unroll_steps+1]

        # Convert scalar targets to support representation.
        target_value = scalar_to_support(
            target_value, support_size)
        target_reward = scalar_to_support(
            target_reward, support_size)
        # Now, target_value and target_reward have shape: [batch, num_unroll_steps+1, 2*support_size+1]

        # --- Generate predictions by unrolling the model ---
        # Initial inference from the observation (corresponds to using the representation model R).
        value, reward, policy_logits, hidden_state = self.model.initial_inference(
            observation_batch)
        predictions = [(value, reward, policy_logits)]
        # Unroll for the remaining steps (using the dynamics model D plus reward O, value V, policy π).
        for i in range(1, action_batch.shape[1]):
            value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                hidden_state, action_batch[:, i]
            )
            # Scale gradient at the start of the dynamics function (see paper appendix "Training").
            hidden_state.register_hook(lambda grad: grad * 0.5)
            predictions.append((value, reward, policy_logits))
        # predictions: List of length (num_unroll_steps+1), each element is (value, reward, policy_logits)

        # --- Compute losses over the unrolled steps ---
        value_loss, reward_loss, policy_loss = 0, 0, 0
        value, reward, policy_logits = predictions[0]
        # For the first step, ignore reward loss (no previous action reward).
        current_value_loss, _, current_policy_loss = self.compute_loss(
            value.squeeze(-1), reward.squeeze(-1), policy_logits,
            target_value[:, 0], target_reward[:, 0], target_policy[:, 0]
        )
        value_loss += current_value_loss
        policy_loss += current_policy_loss

        # Compute priorities for the first step.
        pred_value_scalar = support_to_scalar(
            value, support_size).detach().cpu().numpy().squeeze()
        priorities[:, 0] = (
            np.abs(pred_value_scalar - target_value_scalar[:, 0]) ** PER_alpha)

        # Compute loss and priorities for remaining unroll steps.
        for i in range(1, len(predictions)):
            value, reward, policy_logits = predictions[i]
            (current_value_loss, current_reward_loss, current_policy_loss) = self.compute_loss(
                value.squeeze(-1), reward.squeeze(-1), policy_logits,
                target_value[:, i], target_reward[:, i], target_policy[:, i]
            )
            # Scale gradients by the gradient_scale_batch (see paper appendix "Training").
            current_value_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i])
            current_reward_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i])
            current_policy_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i])

            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss

            # Update priorities for this unroll step.
            pred_value_scalar = support_to_scalar(
                value, support_size).detach().cpu().numpy().squeeze()
            priorities[:, i] = (
                np.abs(pred_value_scalar - target_value_scalar[:, i]) ** PER_alpha)

        # Scale the value loss (paper recommends scaling by 0.25).
        loss = value_loss * value_loss_weight + reward_loss + policy_loss
        if PER:
            # Apply importance-sampling weights to correct bias in prioritized replay.
            loss *= weight_batch
        loss = loss.mean()  # Mean loss over the batch.

        # --- Backpropagation and optimization ---
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.trained_steps += 1

        return (priorities,
                loss.item(),  # Total loss for logging.
                value_loss.mean().item(),
                reward_loss.mean().item(),  # type: ignore
                policy_loss.mean().item())

    def update_learning_rate(self):
        """
        Dynamically update the learning rate based on training progress.

        Uses an exponential decay schedule:
          learning_rate = lr_init * (lr_decay_rate)^(trained_steps / lr_decay_steps)
        """
        learning_rate = lr_init * \
            lr_decay_rate ** (self.trained_steps /
                                          lr_decay_steps)
        for param_group in self.optimizer.param_groups:
            param_group["learning_rate"] = learning_rate

    @staticmethod
    def compute_loss(value, reward, policy_logits, target_value, target_reward, target_policy):
        """
        Compute the loss for a training step.

        For each unroll step, losses are computed as cross-entropy between the predictions
        (value, reward, policy) and their corresponding targets.

        Returns:
            value_loss, reward_loss, policy_loss (each as tensors over the batch).
        """
        # Cross-entropy loss is used (better convergence than MSE in many cases).
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        reward_loss = (-target_reward *
                       torch.nn.LogSoftmax(dim=1)(reward)).sum(1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=1)
                       (policy_logits)).sum(1)
        return value_loss, reward_loss, policy_loss
