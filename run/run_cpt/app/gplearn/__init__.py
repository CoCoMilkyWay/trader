# There are 2 GP types:
# Koza's original GP approach(GPlearn uses this) and Ferreira's GEP (geppy(Deap))

# +-------------------+-------------------------------+----------------------------------+--------------------------------+
# |     Component     |           Input               |            Output                |           Purpose              |
# +-------------------+-------------------------------+----------------------------------+--------------------------------+
# | SymbolicRegressor | Raw features:                 | One formula:                     | For direct value prediction:   |
# |                   | X0 (price)                    | return = 2*X0 + X1^2/X3          | - Stock returns                |
# |                   | X1 (volume)                   | (Single numeric prediction)      | - Price targets                |
# |                   | X2 (volatility)               |                                  | - Continuous metrics           |
# +-------------------+-------------------------------+----------------------------------+--------------------------------+
# | SymbolicClassifier| Same raw features:            | One formula:                     | For classification tasks:      |
# |                   | X0, X1, X2                    | if (X0/X1 + 2*X2 > 0.5)          | - Buy/Sell signals             |
# |                   |                               | (Binary/multi-class output)      | - Market regime detection      |
# |                   |                               |                                  | - Trend direction              |
# +-------------------+-------------------------------+----------------------------------+--------------------------------+
# | SymbolicTransform | Same raw features:            | Multiple formulas:               | For feature engineering:       |
# |                   | X0, X1, X2                    | Feature1 = X0 * log(X1)          | - Creating alpha factors       |
# |                   |                               | Feature2 = sqrt(X2)/X0           | - Technical indicators         |
# |                   |                               | Feature3 = X1^2 + X0/X2          | - Input for other models       |
# +-------------------+-------------------------------+----------------------------------+--------------------------------+

# It's possible that alpha factors (predictive signals) exist even when ensemble trees fail to capture them. Here's why:
# 1. The splitting criteria (like Gini impurity or information gain) might not be able to identify meaningful splits when the signal-to-noise ratio is very low
# 2. Non-linear Relationships: Trees partition the space using axis-parallel splits. Some alpha factors might involve complex non-linear combinations of noisy features that trees can't efficiently approximate with rectangular partitions
# 3. Temporal Dependencies: If the signal in noisy features depends on specific temporal patterns or regime changes, trees might struggle to capture these dynamics
# 4. Conditional Relationships: The signal might only exist under certain complex conditions that are hard for trees to discover through recursive binary splitting

# [INITIALIZATION]────────────────────────────────[Parameters & Constraints]
#         |                                       - init_depth: (2-6)
#         |                                       - population_size: 100-1000
#         |                                       - init_method: 
#         |                                         * 'grow' (random, asymmetric)
#         |                                         * 'full' (functions until max depth)
#         |                                         * 'half and half' (50/50 mix)
#         v
# [CREATE INITIAL POPULATION]───────────────────── Tree Constraints:
#         |                                       - Max Depth (prevent bloat)
#         |                                       - Min Depth (ensure computation)
#         |                                       - Type Safety
#         |                                       - Operation Arity
#         v
# [FITNESS EVALUATION]────────────────────────────[Metrics]
#         |                                       Regression:
#         |                                       - 'mean absolute error'
#         |                                       - 'mse'
#         |                                       - 'rmse'
#         |                                       Transformer:
#         |                                       - 'pearson'
#         |                                       - 'spearman'
#         v
# [TOURNAMENT SELECTION]──────────────────────────[Parameters]
#         |                                       - tournament_size
#         |                                       - parsimony_coefficient
#         |                                       - max_samples (subsampling)
#         v
# [GENETIC OPERATIONS]───────────────────────────[Operation Probabilities]
#         |
#         |─────[CROSSOVER]───────────────────── p_crossover
#         |          - Requires two parents
#         |          - Swaps subtrees
#         |
#         |─────[SUBTREE MUTATION]─────────────── p_subtree_mutation
#         |          - Replaces subtree with random
#         |
#         |─────[HOIST MUTATION]──────────────── p_hoist_mutation
#         |          - Reduces tree size
#         |          - Fights bloat
#         |
#         |─────[POINT MUTATION]──────────────── p_point_mutation
#         |          - Replaces nodes
#         |          - Maintains arity
#         |
#         └─────[REPRODUCTION]───────────────────(1 - sum of other probs)
#                  - Direct copy
#         |
#         v                                    
# [CONSTRAINT CHECKING]──────────────────────────[Constraints]
#         |                                      - Type Constraints
#         |                                      - Operation Constraints
#         |                                      - Protected Operations:
#         |                                        * Division (|x| > 0.001)
#         |                                        * Log (abs value)
#         |                                        * Sqrt (abs value)
#         |                                      - Value Range
#         |                                      - Tree Size
#         |                                      - Resource Usage
#         v
# [TERMINATION CHECK]─────────────────────────── Conditions:
#         |                                      - generations reached
#         |                                      - stopping_criteria met
#         |                                      - perfect score found
#         |
#         └───────[NO]────► Back to FITNESS EVALUATION
#         |
#         v [YES]
# [FINAL PROGRAM]