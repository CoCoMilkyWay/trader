"""
Three-layer evolutionary framework:
1. Base DEAP Framework with parallel processing
2. General GP Framework
3. Specific GP Implementation for financial backtesting
"""
import time
import random
import operator
import warnings
import multiprocessing
import numpy as np
import pandas as pd
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
from deap import base, creator, gp, tools, algorithms
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

class ExecutionMode(Enum):
    """Available execution modes for genetic programming"""
    MP = "multiprocessing"  # DEAP's built-in multiprocessing
    # DEAP's built-in SCOOP(distributed cpu) Run with: python -m scoop script.py
    SCOOP = "scoop"
    GPU = "gpu"             # Custom GPU fitness evaluation


"""
for GP:
    individual = "x0 + sin(x1) * x2"  # Example representation
    population = ["x0 + x1","log(x0) * x2","exp(x1)/x0",# ... more individuals]
    terminals = {'x0': price data,'x1': volume data,'x2': volatility data,# etc...}
"""

class DEAPFramework:
    """
    Base DEAP Framework with parallel processing support
    """

    def __init__(
        self,
        optimization_mode: str = 'maximize',
        n_objectives: int = 1,

        execution_mode=ExecutionMode.MP,
        n_jobs: int = -1,
        batch_size: int = 256,

        verbose: bool = False,
        random_seed: Optional[int] = None
    ):
        self.optimization_mode = optimization_mode
        self.n_objectives = n_objectives
        self.execution_mode = execution_mode
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_seed = random_seed

        # Initialize parallel processor
        self.processor = ParallelProcessor(
            mode=execution_mode,
            n_jobs=n_jobs,
            batch_size=batch_size
        )

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)


class GPFramework(DEAPFramework):
    """General Genetic Programming Framework"""
    # +-------------------+----------------------------------------+----------------------------------------+
    # |    Component      |              Method                    |           Description                  |
    # +-------------------+----------------------------------------+----------------------------------------+
    # | Primitive Set     | 1. Basic PrimitiveSet                  | Standard primitive set                 |
    # |                   | 2. TypedPrimitiveSet                   | Strongly typed GP                      |
    # |                   | 3. PrimitiveTree                       | Manual tree construction               |
    # |                   | 4. addPrimitive                        | Add function with arity                |
    # |                   | 5. addPrimitiveSet                     | Combine primitive sets                 |
    # +-------------------+----------------------------------------+----------------------------------------+
    # | Terminal Options  | 1. addTerminal                         | Basic terminal addition                |
    # |                   | 2. addNamedTerminal                    | Terminal with custom name              |
    # |                   | 3. addTypedTerminal                    | Terminal with type specification       |
    # |                   | 4. addEphemeralConstant                | Random constant generator              |
    # |                   | 5. addADF                              | Automatically defined function         |
    # |                   | 6. addRegressionTerminal               | Specific for symbolic regression       |
    # +-------------------+----------------------------------------+----------------------------------------+
    # | Fitness Creation  | 1. Basic Fitness                       | Standard multi-objective fitness       |
    # |                   | 2. CustomFitness                       | User-defined fitness class             |
    # |                   | 3. ConstrainedFitness                  | Fitness with constraints               |
    # |                   | 4. DeltaPenality                       | Penalize invalid solutions             |
    # |                   | 5. FitnessMax                          | Maximization problems                  |
    # |                   | 6. FitnessMin                          | Minimization problems                  |
    # +-------------------+----------------------------------------+----------------------------------------+
    # | Tree Generation   | 1. genGrow                             | Asymmetrical trees                     |
    # |                   | 2. genFull                             | 'Bushy' trees                          |
    # |                   | 3. genHalfAndHalf                      | Balanced approach                      |
    # |                   | 4. genRamped                           | Ramped generation                      |
    # +-------------------+----------------------------------------+----------------------------------------+
    # | Individual Init   | 1. initIterate                         | Basic initialization                   |
    # |                   | 2. CustomIterator                      | User-defined generator                 |
    # |                   | 3. initCycle                           | Cyclic initialization                  |
    # |                   | 4. initRepeat                          | Repeat initialization                  |
    # +-------------------+----------------------------------------+----------------------------------------+
    # | Population Gen    | 1. initRepeat                          | Basic population generation            |
    # |                   | 2. CustomPopIterator                   | Custom population generator            |
    # |                   | 3. CyclicPopulation                    | Cyclic population generation           |
    # |                   | 4. HallOfFame                          | Store best individuals                 |
    # |                   | 5. ParetoFront                         | Non-dominated solutions                |
    # +-------------------+----------------------------------------+----------------------------------------+
    # | Selection Methods | 1. selTournament                       | Tournament selection                   |
    # |                   | 2. selRoulette                         | Roulette wheel selection               |
    # |                   | 3. selBest                             | Best individual selection              |
    # |                   | 4. selRandom                           | Random selection                       |
    # |                   | 5. selNSGA2                            | NSGA-II selection                      |
    # |                   | 6. selSPEA2                            | SPEA2 selection                        |
    # |                   | 7. selLexicase                         | Lexicase selection                     |
    # |                   | 8. selDoubleTournament                 | Size + fitness tournament              |
    # |                   | 9. selAutomaticEpsilonLexicase         | Epsilon lexicase selection             |
    # |                   | 10. selStochasticUniversalSampling     | SUS selection                          |
    # |                   | 11. selTournamentDCD                   | Crowding distance tournament           |
    # +-------------------+----------------------------------------+----------------------------------------+
    # | Crossover Methods | 1. cxOnePoint                          | One point crossover                    |
    # |                   | 2. cxUniform                           | Uniform crossover                      |
    # |                   | 3. cxSemantic                          | Semantic backpropagation               |
    # |                   | 4. cxSizeFair                          | Size fair crossover                    |
    # |                   | 5. cxRegular                           | Constrained crossover                  |
    # |                   | 6. cxOnePointLeafBiased                | Leaf-biased one point                  |
    # |                   | 7. cxPermutate                         | Permutation crossover                  |
    # |                   | 8. cxBlend                             | Blend crossover                        |
    # |                   | 9. cxESTwoPoint                        | ES two-point crossover                 |
    # |                   | 10. cxSimulatedBinary                  | Simulated binary crossover             |
    # +-------------------+----------------------------------------+----------------------------------------+
    # | Mutation Methods  | 1. mutUniform                          | Uniform mutation                       |
    # |                   | 2. mutNodeReplacement                  | Node replacement                       |
    # |                   | 3. mutInsert                           | Insert mutation                        |
    # |                   | 4. mutShrink                           | Shrink mutation                        |
    # |                   | 5. mutHoist                            | Hoist mutation                         |
    # |                   | 6. mutPermutate                        | Permutation mutation                   |
    # |                   | 7. mutEphemeral                        | Ephemeral mutation                     |
    # |                   | 8. mutSemantic                         | Semantic backpropagation mutation      |
    # |                   | 9. mutGaussian                         | Gaussian mutation                      |
    # |                   | 10. mutPolynomialBounded               | Polynomial bounded mutation            |
    # +-------------------+----------------------------------------+----------------------------------------+
    # | Statistics        | 1. Statistics                          | Basic statistics                       |
    # |                   | 2. MultiStatistics                     | Multiple statistics objects            |
    # |                   | 3. HallOfFame                          | Best individuals tracking              |
    # |                   | 4. Logbook                             | Evolution logging                      |
    # +-------------------+----------------------------------------+----------------------------------------+
    # | Constraints       | 1. DeltaPenalty                        | Penalty function                       |
    # |                   | 2. ConstrainedFitness                  | Constrained optimization               |
    # |                   | 3. InequalityConstraint                | Inequality constraints                 |
    # |                   | 4. EqualityConstraint                  | Equality constraints                   |
    # +-------------------+----------------------------------------+----------------------------------------+
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)


class SymbolicTransformer(GPFramework):
    """
    Specific GP Implementation for Financial Feature Engineering
    """

    def __init__(
        self,
        n_features: int = 5,
        population_size: int = 100,
        n_generations: int = 50,
        tournament_size: int = 3,
        min_depth: int = 2,
        max_depth: int = 6,
        parsimony_coefficient: float = 0.1,
        metric: str = "pearson",
        feature_names: Optional[List[str]] = None,
        hall_of_fame_size: int = 100,
        stopping_criteria: float = 0.0,

        # Original mutation probabilities
        p_crossover: float = 0.65,
        p_subtree_mutation: float = 0.1,
        p_hoist_mutation: float = 0.1,
        p_point_mutation: float = 0.1,
        p_point_replace: float = 0.05,

        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.population_size = population_size
        self.n_generations = n_generations
        self.tournament_size = tournament_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.parsimony_coefficient = parsimony_coefficient
        self.metric = metric
        self.feature_names = feature_names or [
            f"x{i}" for i in range(n_features)]
        self.hall_of_fame_size = hall_of_fame_size
        self.stopping_criteria = stopping_criteria

        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace

        # Setup components
        from .Register import setup_primitives
        self.pset = setup_primitives(feature_names)
        self._setup_fitness()
        self._setup_toolbox()

    def _setup_fitness(self):
        """Setup fitness with parsimony"""
        creator.create("FitnessMax", base.Fitness,
                       weights=(1.0, -self.parsimony_coefficient))
        creator.create("Individual", gp.PrimitiveTree,
                       fitness=creator.FitnessMax)  # type: ignore

    def _setup_toolbox(self):
        """Setup GP toolbox with financial operators"""
        self.toolbox = base.Toolbox()

        # Register map function with DEAP toolbox if not GPU mode
        map_func = self.processor.get_map()
        if map_func is not None:
            self.toolbox.register("map", map_func)

        # Tree generation
        self.toolbox.register("expr", gp.genHalfAndHalf,
                              pset=self.pset,
                              min_=self.min_depth,
                              max_=self.max_depth)

        self.toolbox.register("individual", tools.initIterate,
                              creator.Individual,  # type: ignore
                              self.toolbox.expr)  # type: ignore

        self.toolbox.register("population", tools.initRepeat,
                              list,
                              self.toolbox.individual)  # type: ignore

        # Original genetic operators
        self.toolbox.register("select", tools.selTournament,
                              tournsize=self.tournament_size)
        self.toolbox.register("mate", gp.cxOnePoint)

        # Multiple mutation operators
        """
        • Crossover mutation: A random subtree of the parent tree is replaced by a
        random subtree of the donor tree.
        • Subtree mutation: A random subtree of the parent tree is replaced by a
        random subtree of a randomly generated tree.
        • Hoist mutation: Suppose A is a subtree of the parent tree, B is the subtree
        of A, hoist mutation replaces A with B.
        • Point replace mutation: Any given node will be mutated of the parent tree.
        • Point mutation: A node of the parent tree is replaced by a random node.
        """
        self.toolbox.register(
            "expr_mut", gp.genFull, min_=0, max_=2, pset=self.pset)
        self.toolbox.register(  # Subtree
            "mutate_subtree", gp.mutUniform,
            expr=self.toolbox.expr_mut, pset=self.pset)  # type: ignore
        # mutShrink is not quite the same as hoist, but also good reducer
        # not aggressive as hoist, you might want to increase its prob a bit
        self.toolbox.register(  # Hoist Terminal
            "mutate_hoist", gp.mutShrink)
        self.toolbox.register(  # Point Replace
            "mutate_point", gp.mutNodeReplacement, pset=self.pset)
        self.toolbox.register(  # Insert
            "mutate_insert", gp.mutInsert, pset=self.pset)

        self.toolbox.register("mutate", self._combined_mutation)

    def _combined_mutation(self, individual):
        """Mutation scheme matching gplearn"""
        op_choice = random.random() * (1 - self.p_crossover)
        p_current = 0.0

        # Subtree mutation
        p_current += self.p_subtree_mutation
        if op_choice < p_current:
            return self.toolbox.mutate_subtree(individual)  # type: ignore

        # Hoist mutation
        p_current += self.p_hoist_mutation
        if op_choice < p_current:
            return self.toolbox.mutate_hoist(individual)  # type: ignore

        # Point mutation
        p_current += self.p_point_mutation
        if op_choice < p_current:
            return self.toolbox.mutate_point(individual)  # type: ignore

        # Insert mutation
        p_current += self.p_point_replace
        if op_choice < p_current:
            return self.toolbox.mutate_insert(individual)  # type: ignore

        return individual,  # No mutation

    def _verbose_reporter(self, gen: int, pop, hof, stats, start_time: float):
        """Report progress of the evolution process"""
        if gen == 0:
            print('-' * 130)
            print('    |{:^25}|{:^42}|{:^12}|{:^42}|'.format(
                'Population Average',
                'Best Individual',
                '',
                'Best Expression'))
            print('-' * 130)
            line_format = '{:>4} {:>8} {:>16} {:>8} {:>16} {:>16} {:<12} {:<40}'
            print(line_format.format(
                'Gen', 'Length', 'Fitness', 'Length',
                'Fitness', 'OOB Fitness', 'Time Left', 'Expression'))
            print('-' * 130)
        else:
            # Calculate statistics
            record = stats.compile(pop)
            avg_length = np.mean([len(ind) for ind in pop])
            best_length = len(hof[0])
            
            # Calculate remaining time
            elapsed_time = time.time() - start_time
            remaining_gens = self.n_generations - gen
            time_per_gen = elapsed_time / gen
            remaining_time = remaining_gens * time_per_gen
            
            # Format remaining time
            if remaining_time > 60:
                remaining_time_str = '{:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time_str = '{:.2f}s'.format(remaining_time)
            
            # Get best expression and truncate if needed
            best_expr = str(hof[0])
            if len(best_expr) > 37:
                best_expr = best_expr[:37] + "..."
            
            # Print generation info
            line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:>16} {:<12} {:<40}'
            print(line_format.format(
                gen,
                avg_length,
                record['avg'],
                best_length,
                hof[0].fitness.values[0],
                'N/A',  # OOB Fitness placeholder
                remaining_time_str,
                best_expr
            ))

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit transformer with batched parallel evaluation
        """
        self.X_train_ = X
        self.y_train_ = y

        # Initialize population and components
        pop = self.toolbox.population(n=self.population_size)  # type: ignore
        hof = tools.HallOfFame(self.hall_of_fame_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        start_time = time.time()

        # Track best fitness history for better stopping criteria
        best_fitness_history = []

        # Register statistics
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        if self.verbose:
            self._verbose_reporter(0, pop, hof, stats, start_time)

        # Run evolution with batched evaluation
        for gen in range(self.n_generations):

            # Select offspring (with Elitism to preserve best solutions)
            offspring = self.toolbox.select(  # type: ignore
                pop, len(pop) - len(hof)) # pop none hof
            offspring = list(
                map(self.toolbox.clone, offspring))  # type: ignore
            offspring.extend(map(self.toolbox.clone, hof))  # type: ignore

            # Apply crossover and mutation
            for i in range(1, len(offspring), 2):
                if random.random() < self.p_crossover:
                    offspring[i-1], offspring[i] = self.toolbox.mate(  # type: ignore
                        offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < (1 - self.p_crossover):
                    offspring[i], = self.toolbox.mutate(  # type: ignore
                        offspring[i])
                    del offspring[i].fitness.values
            for ofs in offspring:
                print(str(ofs))

            # Evaluate invalid individuals in batches
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = []

            # Process in batches for GPU mode, use direct map for others
            if self.processor.mode == ExecutionMode.GPU:
                # for i in range(0, len(invalid_ind), self.processor.batch_size):
                #     batch = invalid_ind[i:i + self.processor.batch_size]
                #     batch_results = self.processor.evaluate_batch(batch, self)
                #     fitnesses.extend(batch_results)
                pass
            else:
                # Use DEAP's registered map function (MP or SCOOP)
                for ind in invalid_ind:
                    fitnesses.append(self._evaluate_individual_fitness(ind))
                    
            # Update fitness values
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update population
            pop[:] = offspring

            # Update hall of fame and track best fitness
            hof.update(pop)
            best_fitness = hof[0].fitness.values[0]
            best_fitness_history.append(best_fitness)
            # record = stats.compile(pop)
            if self.verbose:
                self._verbose_reporter(gen + 1, pop, hof, stats, start_time)
                
            # Enhanced stopping criteria
            if len(best_fitness_history) > 10:  # Check last 10 generations
                if (best_fitness >= self.stopping_criteria or
                        # last 10 gen have no impv
                        max(best_fitness_history[-10:]) <= best_fitness_history[-10]):
                    break

        self.population_ = pop
        self.hof_ = hof
        self.best_fitness_history_ = best_fitness_history
        return self

    def _evaluate_individual_fitness(self, individual: gp.PrimitiveTree) -> Tuple[float, float]:
        """
        Evaluate individual with backtesting
        Returns (performance, complexity)
        """
        try:
            # Compile expression
            func = gp.compile(individual, self.pset)

            # Transform features
            feature_arrays = [self.X_train_[col].values
                              for col in self.X_train_.columns]
            transformed = func(*feature_arrays)

            # Run backtesting strategy
            # performance = self._backtest_strategy(transformed)

            # Calculate complexity penalty
            complexity = len(individual)

            # Calculate correlation with target
            correlation = np.corrcoef(
                transformed, self.y_train_.values.astype(float))[0, 1]
            if np.isnan(correlation):
                return 0.0, len(individual)

            return correlation, complexity

        except Exception as e:
            if hasattr(self, 'verbose') and self.verbose:  # type: ignore
                warnings.warn(f"Evaluation failed: {str(e)}")
            return 0.0, len(individual)

    def _backtest_strategy(self, feature: np.ndarray) -> float:
        """
        Backtest trading strategy using transformed feature
        Returns performance metric (e.g., Sharpe ratio)
        """
        # Example implementation - replace with actual strategy
        returns = pd.Series(feature).pct_change().dropna()

        if len(returns) == 0:
            return 0.0

        # Calculate Sharpe ratio
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
        return float(sharpe) if not np.isnan(sharpe) else 0.0

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """Transform data using best individual"""
        if not hasattr(self, 'hof_'):
            raise ValueError("Transformer not fitted")

        # Compile best individual
        func = gp.compile(self.hof_[0], self.pset)

        # Transform features
        feature_arrays = [X[col].values for col in X.columns]
        transformed = func(*feature_arrays)

        return pd.Series(transformed, index=X.index)

    def get_expressions(self) -> list:
        """Get symbolic expression of best individual"""
        if not hasattr(self, 'hof_'):
            raise ValueError("Transformer not fitted")
        return [str(exp) for exp in self.hof_[:10]]

    def _check_stopping_criteria(self, best_fitness: float) -> bool:
        """Check if evolution should be stopped"""
        return (best_fitness >= self.stopping_criteria
                if self.stopping_criteria > 0 else False)

try:
    import cupy as cp  # type: ignore
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
try:
    from scoop import futures  # type: ignore
    HAS_SCOOP = True
except ImportError:
    HAS_SCOOP = False

class ParallelProcessor:
    """Unified parallel processing handler for all execution modes"""

    def __init__(
        self,
        mode: ExecutionMode,
        n_jobs: int = -1,
        batch_size: int = 100
    ):
        self.mode = mode
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self.batch_size = batch_size
        self._validate_mode()
        self._setup_processor()

    def _validate_mode(self):
        """Validate execution mode and handle fallbacks"""
        if self.mode == ExecutionMode.GPU and not HAS_CUPY:
            warnings.warn(
                "CuPy not available. Falling back to multiprocessing.")
            self.mode = ExecutionMode.MP
        elif self.mode == ExecutionMode.SCOOP and not HAS_SCOOP:
            warnings.warn(
                "SCOOP not available. Falling back to multiprocessing.")
            self.mode = ExecutionMode.MP

    def _setup_processor(self):
        """Setup appropriate parallel processing method"""
        if self.mode == ExecutionMode.MP:
            self.pool = multiprocessing.Pool(self.n_jobs)
            self.map_func = self.pool.map
        elif self.mode == ExecutionMode.SCOOP:
            self.map_func = futures.map
        elif self.mode == ExecutionMode.GPU:
            if not HAS_CUPY:
                raise RuntimeError("CuPy is required for GPU mode")
            self.stream = cp.cuda.Stream()
            # Note: No map_func for GPU mode as we handle batches directly

    def get_map(self) -> Optional[Callable]:
        """Get map function for DEAP toolbox (MP and SCOOP modes only)"""
        return None if self.mode == ExecutionMode.GPU else self.map_func

    # def evaluate_batch(self, batch: List[Any], evaluator) -> List[Tuple[float, float]]:
    #     """Evaluate a batch of individuals using appropriate method"""
    #     if self.mode == ExecutionMode.GPU:
    #         return self._gpu_evaluate_batch(batch, evaluator)
    #     else:
    #         raise RuntimeError("GPU mode is required for batch processing")
    #         return []
    # 
    # def _gpu_evaluate_batch(self, batch: List[Any], evaluator) -> List[Tuple[float, float]]:
    #     """Evaluate a batch of individuals on GPU"""
    #     with self.stream:
    #         try:
    #             # Compile expressions
    #             funcs = [gp.compile(ind, evaluator.pset) for ind in batch]
    # 
    #             # Transfer feature data to GPU
    #             feature_arrays_gpu = [
    #                 cp.array(evaluator.X_train_[col].values)
    #                 for col in evaluator.X_train_.columns
    #             ]
    #             y_gpu = cp.array(evaluator.y_train_.values.astype(float))
    # 
    #             # Process each function in batch
    #             batch_results = []
    #             for func, ind in zip(funcs, batch):
    #                 # Apply function to features
    #                 transformed = func(*feature_arrays_gpu)
    # 
    #                 # Calculate correlation
    #                 correlation = self._calculate_correlation(
    #                     transformed, y_gpu)
    #                 complexity = len(ind)
    # 
    #                 batch_results.append((float(correlation), complexity))
    # 
    #             return batch_results
    # 
    #         except Exception as e:
    #             warnings.warn(f"GPU batch processing failed: {str(e)}")
    #             # Fallback to CPU for this batch
    #             return [evaluator._evaluate_individual(ind) for ind in batch]
    # 
    # def _calculate_correlation(self, x: cp.ndarray, y: cp.ndarray) -> float:
    #     """Calculate correlation coefficient on GPU"""
    #     try:
    #         x_mean = cp.mean(x)
    #         y_mean = cp.mean(y)
    # 
    #         numerator = cp.sum((x - x_mean) * (y - y_mean))
    #         denominator = cp.sqrt(cp.sum((x - x_mean)**2)
    #                               * cp.sum((y - y_mean)**2))
    # 
    #         correlation = numerator / denominator
    #         return float(correlation)
    #     except Exception:
    #         return 0.0
    # 
    # def close(self):
    #     """Cleanup resources"""
    #     if hasattr(self, 'pool'):
    #         self.pool.close()
    #         self.pool.join()


# Example usage:
"""
# Create transformer with GPU acceleration
transformer = SymbolicTransformer(
    n_features=len(X.columns),
    population_size=100,
    n_generations=50,
    parallel_method='cupy',  # or 'scoop' for distributed CPU
    n_jobs=-1,
    batch_size=50,
    feature_names=X.columns.tolist()
)

# Fit transformer
transformer.fit(X_train, y_train)

# Get best expression
print(transformer.get_expression())

# Transform new data
X_transformed = transformer.transform(X_test)
"""
