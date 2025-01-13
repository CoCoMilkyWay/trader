import numpy as np
import pandas as pd
from typing import List, Tuple
from deap import base, creator, gp, tools

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # app/
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))  # run/
from config.cfg_cpt import cfg_cpt
from GeneticProgram.Deap import ExecutionMode, SymbolicTransformer


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic financial data with features and target variable
    """
    np.random.seed(seed)

    # Generate dates
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # Generate price with trend and noise
    price = np.cumsum(np.random.normal(0.001, 0.02, n_samples))
    price = np.exp(price) * 100  # Convert to exponential growth

    # Generate volume with correlation to price changes
    volume = np.random.normal(1000000, 200000, n_samples)
    volume += np.abs(np.diff(price, prepend=price[0])) * 100000

    # Generate volatility as rolling standard deviation
    returns = np.diff(price, prepend=price[0]) / price
    volatility = pd.Series(returns).rolling(20).std().fillna(0.0).values
    
    # Create features DataFrame
    X = pd.DataFrame({
        'price': price,
        'volume': volume,
        'volatility': volatility
    }, index=dates)

    # Generate target (future returns, shifted for prediction)
    future_returns = pd.Series(returns).shift(-1).fillna(1.0).values  # Next day's returns
    y = pd.Series(future_returns, index=dates)

    return X, y

def main():
    # train factors for timeseries and cross-section
    
    for file in os.listdir(f'{cfg_cpt.ML_MODEL_DIR}/data'):
        if file.endswith('.parquet'):
            file_path = os.path.join(f'{cfg_cpt.ML_MODEL_DIR}/data', file)
            df = pd.read_parquet(file_path)
            break
    df.describe()
    df.info()
    
    label_cols = [col for col in df.columns if 'label' in col]
    y = df[label_cols[0]].copy()
    X = df.drop(columns=label_cols)
    
    # X, y = generate_synthetic_data(n_samples=1000)
    print("\nGenerated synthetic data:")
    print(X.head())
    print("\nTarget variable head:")
    print(y.head())
    
    # Create and fit transformer
    transformer = SymbolicTransformer(
        execution_mode=ExecutionMode.MP,
        n_jobs=10,
        # batch_size=256, # for GPU ops
        
        population_size=100,
        n_generations=1,
        tournament_size=10,
        min_depth=2,
        max_depth=3,
        parsimony_coefficient=0.1,
        metric="pearson",
        
        n_features=len(X.columns),
        feature_names=X.columns.tolist(),
        
        verbose=True,
        random_seed=0000,
    )
    
    print("\nFitting transformer...")
    transformer.fit(X, y)
    
    # Get best expression
    expressions = transformer.get_expressions()
    print("\nBest expression found:")
    for exp in expressions:
        print(exp)
    
    # Transform data
    X_transformed = transformer.transform(X)
    print("\nTransformed data head:")
    print(X_transformed.head())
    
    # Calculate correlation with target
    correlation = X_transformed.corr(y)
    print(f"\nCorrelation with target: {correlation:.4f}")
    
if __name__ == "__main__":
    profile = False
    if profile:
        import cProfile
        cProfile.run('main()', 'run_bt.prof')
        # snakeviz run_bt.prof
    else:
        main()