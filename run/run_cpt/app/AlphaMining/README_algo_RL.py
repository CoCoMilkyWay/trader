# | Algorithm      | Type       | Action Space    | State Space   | Policy Type                     | Convergence Rate | Exploration Strategy                | Sample Efficiency | Stability | Computational Cost  | Implementation Complexity | Scalability | Suitability for Sparse Rewards| Key Features                             | Key Strengths                        | Key Weaknesses                 | Applications                          |
# |----------------|------------|-----------------|---------------|---------------------------------|------------------|-------------------------------------|-------------------|-----------|---------------------|---------------------------|-------------|-------------------------------|------------------------------------------|--------------------------------------|--------------------------------|---------------------------------------|
# | Q-Learning     | Model-free | Discrete        | Discrete      | Value-based                     | Moderate         | ε-greedy                            | Low               | Low       | Low                 | Simple                    | Limited     | Moderate                      | Off-policy learning with Q-values        | Simple, guarantees convergence       | Struggles with large spaces    | Gridworlds, simple control tasks      |
# | SARSA          | Model-free | Discrete        | Discrete      | Value-based                     | Moderate         | ε-greedy                            | Low               | Moderate  | Low                 | Simple                    | Limited     | Moderate                      | On-policy Q-value learning               | Conservative, reduces overestimation | Less efficient than Q-Learning | Exploration-focused tasks             |
# | DQN            | Model-free | Discrete        | Discrete      | Policy-based w/ value approx    | Slow-Moderate    | ε-greedy + experience replay        | Low               | Moderate  | High                | Moderate                  | Moderate    | Good                          | Q-learning + deep neural networks        | Handles large state spaces           | Needs substantial data         | Atari, games, robotics                |
# | DDQN           | Model-free | Discrete        | Discrete      | Value-based w/ two Q-nets       | Moderate         | ε-greedy + experience replay        | Moderate          | High      | High                | Moderate                  | Moderate    | Good                          | Reduces Q-value overestimation           | More stable than DQN                 | Computationally expensive      | Video games, robotics                 |
# | Dueling DQN    | Model-free | Discrete        | Discrete      | Value-based w/ sep. streams     | Moderate         | ε-greedy + experience replay        | Moderate          | High      | High                | High                      | Moderate    | Good                          | Separate state-value & advantage streams | Reduced variance                     | Increased complexity           | Video games, robot control            |
# | A3C            | Model-free | Discrete/Cont.  | Continuous    | Actor-Critic                    | Fast             | Stochastic policies + multi-workers | Moderate          | High      | High                | Complex                   | Very High   | Moderate                      | Asynchronous multi-agent training        | Scalable, parallelized training      | High computational needs       | Large-scale training, real-time robots|
# | A2C            | Model-free | Discrete/Cont.  | Continuous    | Actor-Critic                    | Moderate         | Stochastic policies                 | Moderate          | High      | Moderate            | Simple                    | Moderate    | Moderate                      | Single-threaded A3C                      | Faster convergence in some cases     | Slower in large environments   | Control tasks, robotics               |
# | PPO            | Model-free | Discrete/Cont.  | Continuous    | Policy-gradient                 | Fast             | Clipped objective                   | High              | Very High | Moderate            | Moderate                  | High        | Excellent                     | Combines stability & sample efficiency   | Simplifies TRPO complexity           | Sensitive to hyperparameters   | Robotics, autonomous driving, game AI |
# | TRPO           | Model-free | Discrete/Cont.  | Continuous    | Policy-gradient + trust region  | Slow             | Trust region                        | Moderate          | Very High | Very High           | Complex                   | Moderate    | Good                          | Guarantees monotonic improvement         | Strong theoretical foundation        | Computationally expensive      | High-dim tasks, robotics              |
# | DDPG           | Model-free | Continuous      | Continuous    | Actor-Critic (deterministic)    | Slow             | Deterministic policy                | Moderate          | Moderate  | High                | High                      | Good        | Good                          | Continuous action space handling         | Efficient in continuous tasks        | Sample inefficient, unstable   | Robotic control, continuous envs      |
# | TD3            | Model-free | Continuous      | Continuous    | Actor-Critic                    | Moderate         | Deterministic + target smoothing    | High              | Very High | High                | High                      | High        | Good                          | Stable DDPG variant                      | Better exploration & stability       | Expensive, needs tuning        | Continuous control, robotics          |
# | SAC            | Model-free | Continuous      | Continuous    | Actor-Critic + entropy          | Fast             | Stochastic + entropy max            | High              | Very High | High                | Moderate                  | Very High   | Excellent                     | Maximum entropy optimization             | Stable & sample efficient            | Entropy term tuning required   | Robotics, high-dim control            |
# | Rainbow DQN    | Model-free | Discrete        | Discrete      | Value-based + multi-improvements| Moderate         | ε-greedy + prioritized replay       | High              | Very High | High                | High                      | Moderate    | Excellent                     | Combines 6+ DQN improvements             | State-of-the-art performance         | Very computationally expensive | Video games, high-dim tasks           |
# | HER            | Model-free | Discrete/Cont.  | Discrete/Cont.| Value-based + hindsight         | High             | Hindsight replay                    | Very High         | High      | Moderate            | High                      | Excellent   | Excellent                     | Goal redefinition for sparse rewards     | Improves sparse reward learning      | Computationally expensive      | Robotics, sparse-reward tasks         |
# | Meta-RL        | Model-free | Discrete/Cont.  | Discrete/Cont.| Policy-gradient/Actor-Critic    | Fast (transfer)  | Task-invariant strategies           | High              | High      | High                | Very High                 | Excellent   | Excellent                     | Fast adaptation to new tasks             | Flexible & transferable              | Complex setup, demanding       | Few-shot learning, adaptive systems   |
# | Deep TAMER     | Model-free | Discrete/Cont.  | Discrete/Cont.| Policy-gradient + human feedback| Moderate         | Human feedback                      | High              | High      | High                | Moderate                  | Moderate    | Excellent                     | Human-in-the-loop learning               | Efficient with human guidance        | Dependent on feedback quality  | Human-robot interaction               |
# | MARL           | Both       | Discrete/Cont.  | Discrete/Cont.| Varies                          | Slow             | Communication protocols             | Low-Moderate      | Moderate  | Very High           | High                      | Low         | Good                          | Multi-agent interaction handling         | Models complex agent relationships   | Coordination challenges        | Game AI, multi-robot systems          |
# | MCTS           | Model-free | Discrete        | Discrete      | Planning-based                  | Fast (decisions) | Random sampling                     | Low               | High      | High                | Moderate                  | Poor        | Poor                          | Tree search for decision-making          | Effective in branching spaces        | Poor for continuous spaces     | Board games, planning problems        |

# +---------------------+--------------------------+---------------------------+-------------------------+-----------------------------+
# | **Category**        | **Value-Based**          | **Policy Gradient/**      | **Model-Based**         | **Meta-RL**                 |
# |                     |                          | **Actor-Critic**          |                         |                             |
# +=====================+==========================+===========================+=========================+=============================+
# | **Key Algorithms**  | Q-Learning, DQN,         | REINFORCE, PPO, SAC,      | Dyna-Q, MCTS,           | MAML (Model-Agnostic        |
# |                     | SARSA, DDQN              | A3C, DDPG, TD3            | AlphaZero, PILCO        | Meta-Learning), RL²         |
# +---------------------+--------------------------+---------------------------+-------------------------+-----------------------------+
# | **Optimization**    | Q-value function         | Policy (directly          | Model of environment +  | Adaptation mechanism        |
# | **Focus**           | approximation            | parametrized policy)      | policy/value function   | for fast task adaptation    |
# +---------------------+--------------------------+---------------------------+-------------------------+-----------------------------+
# | **Action Space**    | Discrete                 | Discrete/Continuous       | Discrete/Continuous     | Discrete/Continuous         |
# +---------------------+--------------------------+---------------------------+-------------------------+-----------------------------+
# | **Sample**          | Low to Moderate          | Moderate to High          | **Very High** (uses     | Low during meta-training    |
# | **Efficiency**      |                          | (Actor-Critic improves    | simulated data)         | but High during adaptation  |
# |                     |                          | efficiency)               |                         |                             |
# +---------------------+--------------------------+---------------------------+-------------------------+-----------------------------+
# | **Stability**       | Moderate (risk of        | Low to Moderate           | High (if model is       | Low to Moderate             |
# |                     | Q-value overestimation)  | (high variance in PG,     | accurate)               | (depends on task similarity)|
# |                     |                          | stabilized in AC)         |                         |                             |
# +---------------------+--------------------------+---------------------------+-------------------------+-----------------------------+
# | **Exploration**     | ε-greedy, UCB            | Entropy regularization,   | Uncertainty-aware       | Task-invariant              |
# | **Strategy**        |                          | stochastic policies       | planning (e.g., MCTS)   | exploration strategies      |
# +---------------------+--------------------------+---------------------------+-------------------------+-----------------------------+
# | **Convergence**     | Guaranteed in tabular    | No guarantees,            | Fast if model is        | No guarantees,              |
# |                     | settings                 | local optima risks        | well-specified          | meta-optimization challenges|
# +---------------------+--------------------------+---------------------------+-------------------------+-----------------------------+
# | **Computational**   | Low (tabular) to         | Moderate to High          | **Very High** (requires | **Extremely High**          |
# | **Cost**            | Moderate (DQN)           | (needs policy gradients)  | model learning/planning)| (meta-training overhead)    |
# +---------------------+--------------------------+---------------------------+-------------------------+-----------------------------+
# | **Implementation**  | Simple                   | Moderate (Actor-Critic)   | Complex (model          | **Very Complex**            |
# | **Complexity**      |                          | to Complex (SAC/DDPG)     | learning + planning)    | (nested optimization)       |
# +---------------------+--------------------------+---------------------------+-------------------------+-----------------------------+
# | **Handles**         | Discrete actions,        | Continuous actions,       | Structured environments | Non-stationary              |
# | **Non-Stationarity**| Low-dimensional states   | high-dimensional states   | with learnable dynamics | environments, multi-task    |
# +---------------------+--------------------------+---------------------------+-------------------------+-----------------------------+
# | **Key Strengths**   | - Clear convergence      | - Handles continuous      | - Sample-efficient      | - Adapts to new tasks       |
# |                     |   guarantees (tabular)   |   action spaces           | - Combines planning +   |   quickly                   |
# |                     | - Simple to implement    | - Direct policy control   |   learning              | - Generalizes across        |
# |                     |                          | - Better exploration      | - Robust to sparse      |   task distributions        |
# |                     |                          |   via stochastic policies |   rewards if model is   |                             |
# |                     |                          |                           |   accurate              |                             |
# +---------------------+--------------------------+---------------------------+-------------------------+-----------------------------+
# | **Key Weaknesses**  | - Poor scalability to    | - High variance           | - Model bias/error      | - Requires massive          |
# |                     |   continuous actions     | - Hyperparameter-sensitive|   propagates to policy  |   meta-training data        |
# |                     | - Overestimation bias    | - Local optima risks      | - Computationally heavy | - Complex optimization      |
# |                     |   (e.g., DQN)            |                           |                         |   (two-level learning)      |
# +---------------------+--------------------------+---------------------------+-------------------------+-----------------------------+
# | **Best Use Cases**  | - Discrete control       | - Robotics, continuous    | - Games with perfect    | - Few-shot learning         |
# |                     |   (e.g., Atari games)    |   control tasks           |   simulators (e.g., Go) | - Rapidly changing          |
# |                     | - Simple environments    | - High-dimensional        | - Industrial control    |   environments              |
# |                     |                          |   state spaces            |   systems               | - Multi-task RL             |
# +---------------------+--------------------------+---------------------------+-------------------------+-----------------------------+

# Markov->qlearning->DQN->PG->AC->ppo
# +---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+--------------------+
# | **Component**       | **MDP**             | **Q-Learning**      | **DQN**             | **Policy Gradient** | **Actor-Critic**    | **PPO**            |
# +=====================+=====================+=====================+=====================+=====================+=====================+====================+
# | **Core Idea**       | Theoretical         | Model-free,         | Q-Learning +        | Directly optimize   | Combines PG +       | Actor-Critic +     |
# |                     | framework for RL    | value-based,        | neural networks     | policy parameters   | value function      | clipped objective  |
# |                     | (states, actions,   | updates Q-table     | (approximate Q)     | via gradients       | (critic reduces     | for stable updates |
# |                     | transitions, γ)     |                     |                     |                     | variance)           |                    |
# +---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+--------------------+
# | **Key Features**    | - No learning       | - Discrete actions  | - Handles high-dim  | - Continuous/       | - Lower variance    | - Robust to        |
# |                     | - Defines           | - ε-greedy          |   states (e.g.,     |   discrete actions  |   than PG           |   hyperparameters  |
# |                     | environment dynamics| - Off-policy        |   images)           | - High variance     | - On-policy/        | - Clipped          |
# |                     |                     | - Tabular updates   | - Experience replay |                     |   off-policy hybrid |   surrogate        |
# |                     |                     |                     | - Target network    |                     |                     |   objective        |
# +---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+--------------------+
# | **Exploration**     | N/A                 | ε-greedy            | ε-greedy + noisy    | Stochastic policy   | Stochastic policy   | Adaptive entropy   |
# |                     |                     |                     | nets                | sampling            | + value-guided      | regularization     |
# +---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+--------------------+
# | **Sample**          | N/A                 | Low                 | Moderate            | Low                 | Moderate            | **High**           |
# | **Efficiency**      |                     | (tabular)           | (replay buffer)     | (high variance)     | (critic guidance)   | (clipped updates)  |
# +---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+--------------------+
# | **Stability**       | N/A                 | Stable in tabular   | Unstable due to     | Unstable            | More stable than PG | **Very stable**    |
# |                     |                     | settings            | moving targets      | (gradient noise)    | (critic smooths     | (clipping avoids   |
# |                     |                     |                     |                     |                     | updates)            | drastic changes)   |
# +---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+--------------------+
# | **Limitations**     | Not practical       | - Scalability       | - Discrete actions  | - High variance     | - Hyperparameter    | - Requires careful |
# | **Addressed Next**  | (no learning)       | - No generalization | - Overestimation    | - Sample            |   sensitivity       |   tuning           |
# |                     |                     |                     |                     |   inefficiency      | - Catastrophic      |                    |
# |                     |                     |                     |                     |                     |   forgetting        |                    |
# +---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+--------------------+
# | **Key Innovations** | Foundation for RL   | Temporal Difference | Function approx.    | Gradient-based      | Decoupling policy   | Trust region       |
# |                     |                     | (TD) learning       | for Q-values        | policy optimization | and value updates   | approximation via  |
# |                     |                     |                     |                     |                     |                     | clipping           |
# +---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+--------------------+
# | **Mathematical**    | Bellman equation:   | Q-table update:     | DQN loss:           | REINFORCE gradient: | Actor gradient:     | Clipped objective: |
# | **Formulation**     | V*(s) = max_a[R(s,a)| Q(s,a) ← Q(s,a) + α | L(θ) = E_{(s,a,r,s')| ∇θ J(θ) = E_π [Σ_t  | ∇θ J(θ) = E [∇θ log | L^CLIP(θ) = E_t    |
# |                     | + γ E[V*(s')]]      | [r + γ max_a' Q(s',a| ~ D [(r + γ max_a'  | ∇θ log π(a_t|s_t; θ)| π(a|s; θ) A(s,a)]   | [min( ratio_t A_t, |
# |                     |                     | ') - Q(s,a)]        | Q(s',a'; θ⁻) -      | Q^π(s_t,a_t) ]      | Critic: A(s,a) =    | clip(ratio_t, 1±ε) |
# |                     |                     |                     | Q(s,a;θ))^2 ]       |                     | Q(s,a) - V(s)       | A_t ) ]            |
# +---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+--------------------+
# 
# To problem with vast exploration space and extremely sparse rewards:
# +------------------------------------------+-------------------------------------------------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
# |                Method                    |                         Key Idea                            |                        Advantages                             |                         Disadvantages                              |
# +------------------------------------------+-------------------------------------------------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
# | Intrinsic Motivation & Curiosity-Driven  | Augment extrinsic rewards with intrinsic signals            | - Provides continual feedback in sparse-reward settings       | - Balancing intrinsic vs. extrinsic rewards is challenging         |
# | Methods                                  | (e.g., prediction error, state novelty)                     | - Encourages exploration in high-dimensional spaces           | - Risk of “gaming” the intrinsic signal                            |
# +------------------------------------------+-------------------------------------------------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
# | Count-Based / Pseudo-Count Methods       | Reward novelty based on state visit counts or pseudo-counts | - Principled measure of novelty                               | - Direct counting isn’t feasible in large spaces                   |
# |                                          | (or approximations thereof)                                 | - Explicitly encourages exploration                           | - Density estimation and hashing can be computationally demanding  |
# +------------------------------------------+-------------------------------------------------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
# | Hierarchical Reinforcement Learning      | Decompose tasks into sub-goals or options to simplify       | - Reduces effective exploration space                         | - Requires design/discovery of effective hierarchies               |
# | (HRL)                                    | exploration and improve long-horizon credit assignment      | - Can improve sample efficiency if sub-goals are well chosen  | - Poor sub-goal selection can hinder learning                      |
# +------------------------------------------+-------------------------------------------------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
# | Maximum Entropy & Uncertainty-Aware      | Incorporate entropy bonuses or uncertainty estimates        | - Balances exploration and exploitation                       | - Entropy bonus tuning can be delicate                             |
# | Approaches                               | into policy updates                                         | - Provides robust performance in stochastic environments      | - Estimating uncertainty in high-dimensional spaces is challenging |
# +------------------------------------------+-------------------------------------------------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
# | Model-Based Approaches                   | Learn a model of the environment to simulate future outcomes| - Enables planning and simulated rollouts to guide exploration| - Learning accurate models in complex domains is difficult         |
# |                                          | and guide exploration                                       | - Can improve sample efficiency by “imagining” trajectories   | - Model errors may misguide exploration in rarely-visited areas    |
# +------------------------------------------+-------------------------------------------------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
# 
