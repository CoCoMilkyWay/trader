import os
import sys
import time
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from Mining.RL.Agent.Core.TrainLoop import TrainLoop
from Mining.RL.Agent.Core.PlayLoop import PlayLoop
from Mining.RL.Agent.Core.ReplayBuffer import ReplayBuffer
from Mining.RL.Agent.Core.CheckPoint import CheckPoint
from Mining.RL.Agent.Core.Game import AbstractGame, Game

from Mining.RL.Env.TokenGenEnv import TokenGenEnv
from Mining.Expression.Operand import Operand
from Mining.Config import DATA, CAPACITY

from Mining.Config import *
from Mining.Util.RNG_Util import set_seed

ENV = TokenGenEnv

# ---------------------------------------------------------------------+
#        Env <-> MCTS <-> Replay-Buffer <-> Trainer                    |
#         ^|                                                           |
#         v|    v----------------------------------------------------+ |
#        Builder(Gym)           Parser             Calculator        | |
# Token(str) ^  v Expression(str) ^ v Operator/Operand ^ v AlphaPool ^ |
#                           Data ---^                                  |
# ---------------------------------------------------------------------+


class Miner:
    def __init__(self):
        # set_seed(1000)

        self.init_miner_env()
        self.init_miner_agent()

        self.run()

    def init_miner_env(self):
        self.env = TokenGenEnv()

    def init_miner_agent(self):
        self.config = AgentConfig()
        self.game = Game(self.env)
        self.replay_buffer = {}

    def remove_ray_temp(self, temp_dir):
        # Remove temporary files in the RAY_TMP_DIR
        print("Cleaning temporary directory...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")
        else:
            print(f"No temporary directory found at {temp_dir}")

    def run(self):

        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        ray_tmp_dir = os.path.join(script_dir, "../../misc/ray_tmp")
        self.remove_ray_temp(ray_tmp_dir)

        app_path = os.path.join(script_dir, ".")
        exc_path = []  # ["/Example/"]  # dont send data

        import ray
        ray.init(
            runtime_env={"working_dir": app_path, "excludes": exc_path},
            _temp_dir=ray_tmp_dir,
        )

        nodes = ray.nodes()
        for node in nodes:
            print("Node IP:", node["NodeManagerAddress"])
            print("Resources:", node["Resources"])

        config = self.config.to_dict()

        self.checkpoint = CheckPoint.options(num_cpus=1, num_gpus=0).\
            remote(config)

        # Workers
        # SelfPlay with Env/Game
        self.replay_buffer_worker = ReplayBuffer.options(num_cpus=1).\
            remote(config, self.checkpoint, self.replay_buffer)
        self.playing_workers = [
            PlayLoop.options(num_cpus=1).
            remote(config, self.checkpoint, self.game)
            for _ in range(self.config.num_workers)]
        self.training_worker = TrainLoop.options(num_cpus=1).\
            remote(config, self.checkpoint)

        for playing_worker in self.playing_workers:
            playing_worker.play_loop.remote(  # type: ignore
                self.checkpoint, self.replay_buffer_worker)

        self.training_worker.train_loop.remote(  # type: ignore
            self.checkpoint, self.replay_buffer_worker)

        while True:
            time.sleep(1)

        ray.shutdown()


if __name__ == '__main__':
    M = Miner()

    """
    A MuZero-Based MDP algo with MCTS mechanism(V value) designed specifically for alpha-mining
    https://arxiv.org/abs/2402.07080 (RiskMiner)
    https://arxiv.org/abs/2306.12964 (AlphaGen)

    MCTS (Monte Carlo Tree Search) with:
        Selection, Expansion, Rollout, Back-propagation
    MDP (Markov Decision Process) with:
        1. model-free
        2. discrete action space, discrete state space
        3. off-policy learning:
            a. behavior policy: balance exploration/exploitation
            b. target policy: use replay Buffer (like DQN/DDPG/...)
        4. policy-based (Actor-Critic specifically):
            a. utilize policy networks (like A3C/PPO/...) to implement policy gradient (Actor)
            b. with quantile optimization to emphasize upper quantile returns
            c. has critic part to minimize variance and improve sample efficiency of policy networks, 
        5. temporal structure: Learn based on previous actions
        6. exploration: risk-seeking like A3C/PPO/TD2 with (e.g. entropy bonuses, action noise, ...)

    ===========================================================================================================
    MuZero Pseudocode

    Inputs:
      - Environment M: Provides the interface to interact with the environment (e.g., step function).
      - Number of episodes N: Total episodes to run for training.
      - Initial state s_0: Starting state for each episode.
      - Learning rate α: Step size for updating model parameters.
      - Discount factor γ: Discount factor for future rewards in value estimation.
      - Number of MCTS simulations k: Number of simulations to run in MCTS for each action selection.
      - Number of unroll steps K: Number of steps to unroll the model during training.
      - Batch size B: Number of trajectories to sample for each training step.

    Initialize models:
      - Representation model R: θ_R.                                   (non-Actor, non-Critic, non-Behavior, non-Target, Muzero Specific)
        - Input: Raw state s.
        - Output: Latent representation rep.
      - Dynamics model D: θ_D.                                         (non-Actor, non-Critic, MCTS specific, Behavior-Policy(part of MCTS))
        - Input: Latent state rep, action a.
        - Output: Next latent state rep_pred.
      - Reward model O: θ_O.                                           (Critic, Behavior-Policy(part of MCTS))
        - Input: Latent state rep, action a.
        - Output: Predicted reward r_pred.
      - Value model V: θ_V. (modified reward function for RL needs)    (Critic, Target-Policy(influenced by Behavior(MCTS)))
        - Input: Latent state rep.
        - Output: Predicted value v_pred.
      - Policy model π: θ_policy.                                      (Actor, Target-Policy(influenced by Behavior(MCTS))
        - Input: Latent state rep.
        - Output: Predicted policy p_pred (action probabilities).

    NOTE: Q value means expected cumulative reward from time t onwards given state and action at time t
          V value here may not directly equal to Q value
          e.g. if Q value = V value + reward, then V value means expected cumulative future reward after t

    Initialize replay buffer B: Stores trajectories for training.

    Main training loop
    FOR episode = 1 TO N:
        # Step 1: Initialize episode variables
        s ← s_0  # Current state starts at initial state
        done ← False  # Episode termination flag
        trajectory ← []  # Empty list to store transitions: (state, action, reward, MCTS policy, MCTS value)

        # Step 2: Collect experience via environment interaction
        WHILE NOT done:
            # Step 2.1: Convert current state to latent representation
            rep ← R(s; θ_R)  # rep = latent representation of s using parameters θ_R

            # Step 2.2: Run MCTS in latent space to get action probabilities and value estimate
            # MCTS uses dynamics (D), reward (O), value (V), and policy (π) models for k simulations
            p_MCTS, v_MCTS ← MCTS(rep, θ_D, θ_O, θ_V, θ_π, k)
            # p_MCTS: action probabilities from MCTS, v_MCTS: value estimate from MCTS

            # Step 2.3: Sample action from MCTS policy
            a ← sample_action(p_MCTS)  # Sample action a ~ p_MCTS

            # Step 2.4: Execute action in the environment
            s', r, done ← M.step(s, a)  # Get next state, reward, and done flag

            # Step 2.5: Store transition in trajectory
            trajectory.append((s, a, r, p_MCTS, v_MCTS))

            # Step 2.6: Update current state
            s ← s'

        # Step 3: Store completed trajectory in replay buffer
        B.add(trajectory)

        # Step 4: Sample batch of trajectories for training
        batch ← B.sample(B)  # Sample B trajectories, each with T steps

        # Step 5: Train models using sampled batch
        FOR trajectory IN batch:
            # Step 5.1: Unpack trajectory into components
            (s_0, s_1, ..., s_T), (a_0, a_1, ..., a_{T-1}), (r_0, r_1, ..., r_{T-1}),
            (p_0^MCTS, p_1^MCTS, ..., p_{T-1}^MCTS), (v_0^MCTS, v_1^MCTS, ..., v_{T-1}^MCTS) ← trajectory

            # Step 5.2: Initialize latent state for trajectory starting point
            rep_0 ← R(s_0; θ_R)  # Initial latent representation

            # Step 5.3: Initialize storage for predictions and targets across K unroll steps
            r_preds ← []  # Predicted latent states over unrolls
            reward_preds ← []  # Predicted rewards
            value_preds ← []  # Predicted values
            policy_preds ← []  # Predicted policies
            reward_targets ← []  # Actual rewards from environment
            value_targets ← []  # MCTS value estimates
            policy_targets ← []  # MCTS policy estimates

            # Step 5.4: Unroll model for K steps starting at each timestep t
            FOR t = 0 TO T-1:
                # Step 5.4.1: Set current latent state
                IF t = 0:
                    rep_t ← rep_0  # Use initial latent state
                ELSE:
                    rep_t ← D(rep_{t-1}, a_{t-1}; θ_D)  # Predict next latent state using dynamics model

                # Step 5.4.2: Generate predictions at timestep t
                r_pred_t ← O(rep_t, a_t; θ_O)  # Predicted reward
                v_pred_t ← V(rep_t; θ_V)  # Predicted value
                p_pred_t ← π(rep_t; θ_π)  # Predicted policy

                # Step 5.4.3: Store predictions
                reward_preds.append(r_pred_t)
                value_preds.append(v_pred_t)
                policy_preds.append(p_pred_t)

                # Step 5.4.4: Store targets from trajectory
                reward_targets.append(r_t)  # Actual reward from environment
                value_targets.append(v_t^MCTS)  # MCTS value estimate
                policy_targets.append(p_t^MCTS)  # MCTS policy estimate

                # Step 5.4.5: Unroll for K additional steps (if K > 1)
                rep_current ← rep_t
                FOR k = 1 TO K-1:
                    IF t + k < T:
                        # Use actual action and reward from trajectory if within bounds
                        rep_current ← D(rep_current, a_{t+k}; θ_D)
                        r_pred_k ← O(rep_current, a_{t+k}; θ_O)
                        v_pred_k ← V(rep_current; θ_V)
                        p_pred_k ← π(rep_current; θ_π)
                        reward_preds.append(r_pred_k)
                        value_preds.append(v_pred_k)
                        policy_preds.append(p_pred_k)
                        reward_targets.append(r_{t+k})
                        value_targets.append(v_{t+k}^MCTS)
                        policy_targets.append(p_{t+k}^MCTS)
                    ELSE:
                        # Simulate beyond trajectory using model predictions
                        a_sim ← sample_action(p_pred_{k-1})
                        rep_current ← D(rep_current, a_sim; θ_D)
                        r_pred_k ← O(rep_current, a_sim; θ_O)
                        v_pred_k ← V(rep_current; θ_V)
                        p_pred_k ← π(rep_current; θ_π)
                        reward_preds.append(r_pred_k)
                        value_preds.append(v_pred_k)
                        policy_preds.append(p_pred_k)
                        # For targets beyond T, use bootstrapped estimates (simplified here)
                        reward_targets.append(r_pred_k)  # Self-consistency for reward
                        value_targets.append(v_pred_k)   # Bootstrap value
                        policy_targets.append(p_pred_k)  # Bootstrap policy

            # Step 6: Compute losses over all predictions
            # Step 6.1: Reward loss (mean squared error)
            L_reward ← (1 / |reward_preds|) * SUM_{i} (reward_targets[i] - reward_preds[i])^2

            # Step 6.2: Value loss (mean squared error)
            L_value ← (1 / |value_preds|) * SUM_{i} (value_targets[i] - value_preds[i])^2

            # Step 6.3: Policy loss (cross-entropy)
            L_policy ← (1 / |policy_preds|) * SUM_{i} cross_entropy(policy_targets[i], policy_preds[i])

            # Step 6.4: Total loss
            L_total ← L_reward + L_value + L_policy

            # Step 7: Update each network with gradients from total loss
            # Step 7.1: Update Reward Model (O)
            ∇_θ_O L_total ← compute_gradient(L_reward, θ_O)  # Gradient of L_total w.r.t. θ_O
            θ_O ← θ_O - α * ∇_θ_O L_total  # Gradient descent update

            # Step 7.2: Update Value Model (V)
            ∇_θ_V L_total ← compute_gradient(L_value, θ_V)  # Gradient of L_total w.r.t. θ_V
            θ_V ← θ_V - α * ∇_θ_V L_total  # Gradient descent update

            # Step 7.3: Update Policy Model (π)
            ∇_θ_π L_total ← compute_gradient(L_policy, θ_π)  # Gradient of L_total w.r.t. θ_π
            θ_π ← θ_π - α * ∇_θ_π L_total  # Gradient descent update

            # Step 7.4: Update Dynamics Model (D)
            ∇_θ_D L_total ← compute_gradient(L_total, θ_D)  # Gradient of L_total w.r.t. θ_D
            θ_D ← θ_D - α * ∇_θ_D L_total  # Gradient descent update

            # Step 7.5: Update Representation Model (R)
            ∇_θ_R L_total ← compute_gradient(L_total, θ_R)  # Gradient of L_total w.r.t. θ_R
            θ_R ← θ_R - α * ∇_θ_R L_total  # Gradient descent update

    Helper functions (assumed implemented):
    - MCTS(rep, θ_D, θ_O, θ_V, θ_π, k): Runs k MCTS simulations in latent space, returns policy and value.
    - sample_action(p): Samples an action from probability distribution p.
    - compute_gradient(L, θ): Computes ∂L/∂θ using backpropagation.
    - cross_entropy(target, pred): Computes cross-entropy loss between target and predicted probabilities.
    """
