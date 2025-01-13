# ╔═══════════════════╦════════════════════════════════╦═══════════════════════════════════╦═════════════════════════════════╦════════════════════════════════╗
# ║     Category      ║        Algorithms/Variants     ║         Core Processes            ║         Input Formats           ║         Use Cases              ║
# ╠═══════════════════╬════════════════════════════════╬═══════════════════════════════════╬═════════════════════════════════╬════════════════════════════════╣
# ║ Traditional EA    ║ Simple EA:                     ║ Common EA Processes:              ║ • Fixed-length real vectors     ║ • Parameter optimization       ║
# ║ (Evolutionary/    ║ • eaSimple                     ║ 1. Population Initialization      ║ • Binary strings                ║ • Resource allocation          ║
# ║  Genetic          ║ • eaGenerateUpdate             ║ 2. Fitness Evaluation             ║ • Permutations                  ║ • Production scheduling        ║
# ║  Algorithms)      ║                                ║ 3. Selection                      ║ • Integer vectors               ║ • Network design               ║
# ║                   ║ Evolution Strategy:            ║ 4. Variation                      ║ • Mixed integer-real vectors    ║ • Circuit design               ║
# ║                   ║ • eaMuPlusLambda (μ+λ)         ║ 5. Replacement                    ║                                 ║ • Feature selection            ║
# ║                   ║ • eaMuCommaLambda (μ,λ)        ║                                   ║                                 ║ • Portfolio optimization       ║
# ║                   ║ • CMA-ES                       ║ Special ES Processes:             ║                                 ║                                ║
# ║                   ║ • sep-CMA-ES                   ║ • Strategy Parameter Update       ║                                 ║                                ║
# ║                   ║                                ║ • Covariance Matrix Adaptation    ║                                 ║                                ║
# ╟───────────────────╫────────────────────────────────╫───────────────────────────────────╫─────────────────────────────────╫────────────────────────────────╢
# ║ Genetic           ║ Tree-based GP:                 ║ Common GP Processes:              ║ • Function sets                 ║ • Symbolic regression          ║
# ║ Programming       ║ • Standard GP (Koza's)         ║ 1. Primitive/Terminal Definition  ║ • Terminal sets                 ║ • Boolean function synthesis   ║
# ║                   ║ • STGP (Strongly Typed)        ║ 2. Tree Generation                ║ • Grammar rules                 ║ • Controller design            ║
# ║                   ║ • DEAP PushGP                  ║ 3. Expression Evaluation          ║ • Type hierarchies              ║ • Feature engineering          ║
# ║                   ║ • Grammar Guided GP            ║ 4. Genetic Operations             ║ • Custom primitives             ║ • Program synthesis            ║
# ║                   ║                                ║                                   ║                                 ║ • Rule discovery               ║
# ║                   ║ Linear GP:                     ║ Special GP Features:              ║                                 ║ • Signal processing            ║
# ║                   ║ • Linear GP                    ║ • Type Constraints                ║                                 ║ • Financial modeling           ║
# ║                   ║ • Multi Expression Prog        ║ • Bloat Control                   ║                                 ║                                ║
# ║                   ║ • Cartesian GP                 ║ • Semantic Backpropagation        ║                                 ║                                ║
# ║                   ║ • GEP(gene expr) (Ferreira's)  ║                                   ║                                 ║                                ║
# ╟───────────────────╫────────────────────────────────╫───────────────────────────────────╫─────────────────────────────────╫────────────────────────────────╢
# ║ Differential      ║ Classic DE:                    ║ Common DE Processes:              ║ • Real-valued vectors           ║ • Neural network training      ║
# ║ Evolution         ║ • DE/rand/1/bin                ║ 1. Population Initialization      ║ • Mixed integer-real vectors    ║ • Engineering design           ║
# ║                   ║ • DE/best/1/bin                ║ 2. Mutation Vector Generation     ║ • Permutations                  ║ • Chemical process opt.        ║
# ║                   ║ • DE/rand/2/bin                ║ 3. Crossover                      ║ • Binary vectors                ║ • Image processing             ║
# ║                   ║ • DE/best/2/bin                ║ 4. Selection                      ║                                 ║ • Pattern recognition          ║
# ║                   ║ • DE/current-to-best/1         ║                                   ║                                 ║ • Financial modeling           ║
# ║                   ║                                ║ Special DE Features:              ║                                 ║ • Antenna design               ║
# ║                   ║ Adaptive DE:                   ║ • Self-Adaptive Parameters        ║                                 ║                                ║
# ║                   ║ • JADE                         ║ • Population Size Reduction       ║                                 ║                                ║
# ║                   ║ • jDE                          ║                                   ║                                 ║                                ║
# ╟───────────────────╫────────────────────────────────╫───────────────────────────────────╫─────────────────────────────────╫────────────────────────────────╢
# ║ Multi-Objective   ║ Pareto-based:                  ║ Common MOEA Processes:            ║ • Real-valued vectors           ║ • Engineering design           ║
# ║ Evolution         ║ • NSGA-II                      ║ 1. Non-dominated Sorting          ║ • Binary strings                ║ • Portfolio optimization       ║
# ║                   ║ • NSGA-III                     ║ 2. Diversity Preservation         ║ • Mixed representations         ║ • Supply chain opt.            ║
# ║                   ║ • SPEA2                        ║ 3. Elite Preservation             ║ • Problem-specific encodings    ║ • Environmental management     ║
# ║                   ║ • IBEA                         ║ 4. Reference Point Methods        ║ • Multiple objective functions  ║ • Network design               ║
# ║                   ║                                ║                                   ║                                 ║ • Resource allocation          ║
# ║                   ║ Decomposition-based:           ║ Special MOEA Features:            ║                                 ║ • Chemical process opt.        ║
# ║                   ║ • MOEA/D                       ║ • Reference Points                ║                                 ║                                ║
# ║                   ║ • MOEA/D-DE                    ║ • Crowding Distance               ║                                 ║                                ║
# ║                   ║                                ║ • Tchebycheff Decomposition       ║                                 ║                                ║
# ╟───────────────────╫────────────────────────────────╫───────────────────────────────────╫─────────────────────────────────╫────────────────────────────────╢
# ║ Swarm             ║ Particle Swarm:                ║ Common SI Processes:              ║ • Real-valued vectors           ║ • Function optimization        ║
# ║ Intelligence      ║ • Standard PSO                 ║ 1. Swarm Initialization           ║ • Binary vectors                ║ • Path planning                ║
# ║                   ║ • Binary PSO                   ║ 2. Position Update                ║ • Discrete positions            ║ • Routing problems             ║
# ║                   ║ • Multi-Objective PSO          ║ 3. Velocity Update                ║ • Path sequences                ║ • Task scheduling              ║
# ║                   ║                                ║ 4. Global/Local Best Update       ║ • Graph representations         ║ • Feature selection            ║
# ║                   ║ Ant Colony:                    ║                                   ║                                 ║ • Pattern recognition          ║
# ║                   ║ • AS (Ant System)              ║ Special SI Features:              ║                                 ║ • Network optimization         ║
# ║                   ║ • ACS (Ant Colony System)      ║ • Pheromone Updates               ║                                 ║                                ║
# ║                   ║ • MMAS (Max-Min AS)            ║ • Neighborhood Topology           ║                                 ║                                ║
# ╟───────────────────╫────────────────────────────────╫───────────────────────────────────╫─────────────────────────────────╫────────────────────────────────╢
# ║ Hybrid/Advanced   ║ Memetic Algorithms:            ║ Common Advanced Processes:        ║ • Problem-specific repr.        ║ • Large-scale optimization     ║
# ║                   ║ • MA with local search         ║ 1. Population Structure           ║ • Multiple populations          ║ • Dynamic problems             ║
# ║                   ║ • Adaptive MA                  ║ 2. Migration                      ║ • Hierarchical structures       ║ • Multi-modal optimization     ║
# ║                   ║                                ║ 3. Local Optimization             ║ • Mixed representations         ║ • Constrained optimization     ║
# ║                   ║ Coevolution:                   ║ 4. Adaptation                     ║ • Adaptive encodings            ║ • Complex system design        ║
# ║                   ║ • Cooperative Coevolution      ║                                   ║                                 ║ • Real-time adaptation         ║
# ║                   ║ • Competitive Coevolution      ║ Special Features:                 ║                                 ║ • Multi-level optimization     ║
# ║                   ║                                ║ • Migration Policies              ║                                 ║                                ║
# ║                   ║ Island Models:                 ║ • Topology                        ║                                 ║                                ║
# ║                   ║ • Synchronous Islands          ║ • Learning Rules                  ║                                 ║                                ║
# ║                   ║ • Asynchronous Islands         ║                                   ║                                 ║                                ║
# ╚═══════════════════╩════════════════════════════════╩═══════════════════════════════════╩═════════════════════════════════╩════════════════════════════════╝

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

# +---------------------+----------------+-----------+------------------------------------------+
# | Parameter           | Normal Range   | Default   | Meaning                                  |
# |---------------------+----------------+-----------+------------------------------------------|
# | population_size     | 1000-10000     | 1000      | Number of programs in each generation    |
# |                     |                |           | Larger = more exploration, slower        |
# |---------------------+----------------+-----------+------------------------------------------|
# | generations         | 10-50          | 20        | Number of evolution cycles               |
# |                     |                |           | More = better results but slower         |
# |---------------------+----------------+-----------+------------------------------------------|
# | tournament_size     | 5-20           | 20        | Number of programs compared in selection |
# |                     |                |           | Larger = higher selection pressure       |
# |---------------------+----------------+-----------+------------------------------------------|
# | p_crossover         | 0.7-0.9        | 0.9       | Probability of crossover between programs|
# |                     |                |           | Higher = more genetic mixing             |
# |---------------------+----------------+-----------+------------------------------------------|
# | p_subtree_mutation  | 0.05-0.2       | 0.01      | Chance to replace subtree with new one   |
# |                     |                |           | Higher = more structural changes         |
# |---------------------+----------------+-----------+------------------------------------------|
# | p_hoist_mutation    | 0.05-0.1       | 0.01      | Chance to replace tree with subtree      |
# |                     |                |           | Higher = more simplification             |
# |---------------------+----------------+-----------+------------------------------------------|
# | p_point_mutation    | 0.05-0.2       | 0.01      | Chance to change single node             |
# |                     |                |           | Higher = more fine-tuning                |
# |---------------------+----------------+-----------+------------------------------------------|
# | parsimony_coef      | 0.001-0.1      | 0.01      | Penalty for program complexity           |
# |                     |                |           | Higher = favors simpler programs         |
# |---------------------+----------------+-----------+------------------------------------------|
# | init_depth          | (2,6)-(3,8)    | (2,6)     | Initial program tree depth range         |
# |                     |                |           | Larger = more complex initial programs   |
# |---------------------+----------------+-----------+------------------------------------------|
# | n_components        | 1-10           | 10        | Number of best programs to return        |
# | (Transformer only)  |                |           | More = more diverse solutions            |
# |---------------------+----------------+-----------+------------------------------------------|
# | metric              |'mse','pearson' | 'pearson' | Fitness metric for evaluation            |
# |                     |                |           | Depends on problem type                  |
# +---------------------+----------------+-----------+------------------------------------------+

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