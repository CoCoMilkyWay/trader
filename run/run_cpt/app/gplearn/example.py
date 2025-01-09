import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../")) # app/

import numpy as np
from gplearn.genetic import SymbolicRegressor
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
n_samples = 1000

# Input variables
x1 = np.random.uniform(-3, 3, n_samples)
x2 = np.random.uniform(-3, 3, n_samples)
noise = np.random.uniform(-0.5, 0.5, n_samples)

# Create a complex target equation: y = sin(x1) + x2^2
y_true = np.sin(x1) + x2**2 + noise

# Add some noise
noise = np.random.normal(0, 0.1, n_samples)
y = y_true + noise

# Combine input variables
X = np.column_stack((x1, x2))

# Configure symbolic regressor
est_gp = SymbolicRegressor(
    population_size=5000,       # Bigger population for better exploration
    generations=10,             # Number of evolution iterations
    tournament_size=20,         # Size of tournament selection
    function_set=               # Available functions
    ('add', 'sub', 'mul', 'div', 'sin', 'cos', 'sqrt'), 
    metric='mse',               # Mean squared error as fitness metric
    init_depth=(2, 6),          # Initial solutions' tree depth
    parsimony_coefficient=0.1,  # Penalize long solutions
    p_crossover=0.7,            # Probability of crossover
    p_subtree_mutation=0.1,     # Probability of subtree mutation
    p_point_mutation=0.1,       # Probability of point mutation
    p_hoist_mutation=0.05,      # Probability of hoist mutation
    verbose=1,                  # Print progress
    n_jobs=5,
    random_state=42             # For reproducibility
)

# Fit the model
print("Training symbolic regression model...")
est_gp.fit(X, y)

# Print the best found solution
print("\nBest program:", est_gp._program)
print("Raw fitness (MSE):", est_gp._program.raw_fitness_)

# Make predictions
y_pred = est_gp.predict(X)

# Print some metrics
from sklearn.metrics import mean_squared_error, r2_score
print("\nMean squared error:", mean_squared_error(y, y_pred))
print("R² score:", r2_score(y, y_pred))
