# PyTorch imports
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.jit

# Other ensemble methods
import xgboost as xgb
import lightgbm as lgb

# Data handling and processing
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import (
    KFold, 
    TimeSeriesSplit, 
    train_test_split,
    GroupKFold,
    StratifiedKFold
)
from scipy import stats

# Utilities and helpers
import math
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional, Any, TypeVar, Iterator
from dataclasses import dataclass
from enum import Enum

class ScalingMethod(Enum):
    """Available scaling methods for features."""
    STANDARD = "standard"   # Zero mean, unit variance
    MINMAX = "minmax"       # Scale to range [0,1]
    ROBUST = "robust"       # Robust to outliers
    NONE = "none"           # No scaling

class SplitMethod(Enum):
    """Combined data splitting and cross-validation methods."""
    """
    1. RANDOM (random 80-20 split)
    Data: [1,2,3,4,5,6,7,8,9,10]
    T T V T T T V T T V  (Randomly distributed)

    2. STRATIFIED (preserves class distribution)
    Data: [1A,2A,3B,4B,5A,6B,7A,8B,9A,10B]  (A,B are classes)
    T T V T T T V T T V  (Each split has same A:B ratio)

    3. TIMESERIES (respects time order)
    Data: [1,2,3,4,5,6,7,8,9,10]  (time order →)
    T T T T T T T V V V  (Later points in validation)

    4. GROUP (keeps related samples together)
    Data: [1g1,2g1,3g2,4g2,5g3,6g3,7g4,8g4,9g5,10g5]  (g1-g5 are groups)
    T T T T V V T T V V  (Entire groups go together)

    5. KFOLD_CV (k=3)
    Fold 1:  V V V T T T T T T T
    Fold 2:  T T T V V V T T T T
    Fold 3:  T T T T T T V V V V

    6. STRATIFIED_KFOLD_CV (k=3, with classes A,B)
    Fold 1:  V V V T T T T T T T  (Each fold maintains A:B ratio)
    Fold 2:  T T T V V V T T T T
    Fold 3:  T T T T T T V V V V

    7. TIMESERIES_CV (expanding window)
    Split 1: T T T V
    Split 2: T T T T V
    Split 3: T T T T T V

    8. GROUP_KFOLD_CV (k=3, with groups g1-g5)
    Data:    [g1,g1,g2,g2,g3,g3,g4,g4,g5,g5]
    Fold 1:  V V T T T T T T T T  (g1 together)
    Fold 2:  T T V V T T T T T T  (g2 together)
    Fold 3:  T T T T V V T T T T  (g3 together)
    """
    
    # Basic splits without CV
    RANDOM = "random"               # Random train-test split
    STRATIFIED = "stratified"       # Stratified split (preserves label distribution)
    TIMESERIES = "timeseries"       # Temporal order preserved
    GROUP = "group"                 # Group-based split
    
    # Cross-validation methods
    KFOLD_CV = "kfold_cv"          # K-fold cross validation
    STRATIFIED_KFOLD_CV = "stratified_kfold_cv"  # Stratified k-fold
    TIMESERIES_CV = "timeseries_cv"  # Time series cross validation
    GROUP_KFOLD_CV = "group_kfold_cv" # Group k-fold cross validation

@dataclass
class DataCheckResult:
    """Stores the results of data quality checks."""
    feature_name: str
    distribution_type: str
    mean: float
    variance: float
    skewness: float
    kurtosis: float
    has_outliers: bool
    cardinality: int
    range_bound: Tuple[float, float]
    suitable_for_nn: bool
    suitable_for_ensemble: bool
    warnings: List[str]
    
    def __post_init__(self) -> None:
        """Convert numeric values to float after initialization."""
        try:
            self.mean = float(self.mean)
            self.variance = float(self.variance)
            self.skewness = float(self.skewness)
            self.kurtosis = float(self.kurtosis)
            if not isinstance(self.range_bound, tuple) or len(self.range_bound) != 2:
                raise ValueError("range_bound must be a tuple of two numbers")
            self.range_bound = (float(self.range_bound[0]), float(self.range_bound[1]))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to convert numeric values to float: {str(e)}")

class ModelType(Enum):
    """Available model types."""
    MLP = "mlp"
    CNN = "cnn"
    LSTM = "lstm"
    GRU = "gru"
    BILSTM = "bilstm"
    BIGRU = "bigru"
    TRANSFORMER = "transformer"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

class CNN(nn.Module):
    """Convolutional Neural Network implementation."""
    def __init__(self, 
                input_dim: int, 
                hidden_dims: List[int], 
                output_dim: int,
                kernel_sizes: List[int],
                use_batch_norm: bool = True,
                dropout: float = 0.3,
                activation: str = 'relu'):
        super().__init__()
        
        if len(hidden_dims) != len(kernel_sizes):
            raise ValueError("Number of hidden dimensions must match number of kernel sizes")
            
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build CNN layers
        self.layers = nn.ModuleList()
        in_channels = input_dim
        
        for hidden_dim, kernel_size in zip(hidden_dims, kernel_sizes):
            conv_block = []
            
            # Conv layer
            conv_block.append(nn.Conv1d(
                in_channels, hidden_dim, kernel_size, 
                padding=kernel_size//2
            ))
            
            # Batch norm
            if use_batch_norm:
                conv_block.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == 'relu':
                conv_block.append(nn.ReLU())
            elif activation == 'gelu':
                conv_block.append(nn.GELU())
            
            # Dropout
            conv_block.append(nn.Dropout(dropout))
            
            self.layers.append(nn.Sequential(*conv_block))
            in_channels = hidden_dim
        
        # Global pooling and output layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, channels, sequence_length)
        for layer in self.layers:
            x = layer(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)

class Recurrent(nn.Module):
    """Recurrent Neural Network implementation (LSTM/GRU) with simplified typing."""
    def __init__(self,
                input_size: int,
                hidden_size: int,
                output_size: int,
                model_type: str = 'lstm',
                num_layers: int = 2,
                bidirectional: bool = False,
                dropout: float = 0.3):
        super().__init__()
        
        self.model_type = model_type.lower()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Select RNN type
        if self.model_type == 'lstm':
            rnn_cell = nn.LSTM
        elif self.model_type == 'gru':
            rnn_cell = nn.GRU
        else:
            raise ValueError(f"Unsupported RNN type: {model_type}")
        
        # Create RNN
        self.rnn = rnn_cell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer - account for bidirectional
        fc_input_size = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with simplified typing for TorchScript compatibility.
        Args:
            x: Input tensor of shape [batch_size, features]
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        # Add sequence dimension [batch_size, 1, features]
        x = x.unsqueeze(1)
        
        # Forward pass through RNN - let PyTorch handle hidden state
        outputs, _ = self.rnn(x)
        
        # Get output for each sample in batch
        last_output = outputs[:, -1, :]
        
        # Apply dropout and final linear layer
        dropped = self.dropout(last_output)
        return self.fc(dropped)

class Transformer(nn.Module):
    """Transformer model implementation."""
    def __init__(self,
                input_size: int,
                d_model: int,
                nhead: int,
                num_layers: int,
                output_size: int,
                dim_feedforward: int = 2048,
                dropout: float = 0.1,
                activation: str = "relu"):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Position encoding
        self.register_buffer('pe', self._create_positional_encoding())
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, output_size)
        
    def _create_positional_encoding(self, max_len: int = 5000) -> torch.Tensor:
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def _create_causal_mask(self, size: int) -> torch.Tensor:
        return torch.triu(torch.ones(size, size), diagonal=1).bool()
    
    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input shape: (batch_size, sequence_length, input_size)
        x = self.input_proj(x)
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        
        # Create mask if not provided
        if mask is None:
            mask = self._create_causal_mask(x.size(1)).to(x.device)
            
        x = self.transformer(x, mask=mask)
        x = x.mean(dim=1)  # Global pooling
        return self.output_proj(x)

class Ensemble:
    """Wrapper class for ensemble models (XGBoost/LightGBM)."""
    def __init__(self,
                model_type: str = 'xgboost',
                **params):
        self.model_type = model_type.lower()
        
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**{
                'n_estimators': params.get('n_estimators', 100),
                'learning_rate': params.get('learning_rate', 0.1),
                'max_depth': params.get('max_depth', 6),
                'subsample': params.get('subsample', 0.8),
                'colsample_bytree': params.get('colsample_bytree', 0.8),
                'min_child_weight': params.get('min_child_weight', 1),
                'objective': params.get('objective', 'reg:squarederror'),
                'tree_method': params.get('tree_method', 'hist'),
                'booster': params.get('booster', 'gbtree')
            })
        
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(**{
                'n_estimators': params.get('n_estimators', 100),
                'learning_rate': params.get('learning_rate', 0.1),
                'max_depth': params.get('max_depth', 6),
                'num_leaves': params.get('num_leaves', 31),
                'subsample': params.get('subsample', 0.8),
                'colsample_bytree': params.get('colsample_bytree', 0.8),
                'min_child_samples': params.get('min_child_samples', 20),
                'objective': params.get('objective', 'regression'),
                'boosting_type': params.get('boosting_type', 'gbdt')
            })
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def fit(self, X, y, **kwargs):
        """Fit the ensemble model."""
        return self.model.fit(X, y, **kwargs) # type: ignore
        
    def predict(self, X):
        """Make predictions with the ensemble model."""
        return self.model.predict(X)
        
    def save_model(self, path: str):
        """Save the ensemble model."""
        if self.model_type == 'lightgbm':
            # Save LightGBM model using model's save_model method
            if hasattr(self.model, 'booster_'):
                self.model.booster_.save_model(path) # type: ignore
            else:
                lgb.Booster(model_file=path)
        elif self.model_type == 'xgboost':
            self.model.save_model(path) # type: ignore
        else:
            # Fallback for models without direct save_model method
            import joblib
            joblib.dump(self.model, path)
            
    def load_model(self, path: str):
        """Load the ensemble model."""
        if self.model_type == 'lightgbm':
            # Load LightGBM model
            self.model = lgb.Booster(model_file=path)
        elif self.model_type == 'xgboost':
            self.model.load_model(path) # type: ignore
        else:
            # Fallback for models without direct load_model method
            import joblib
            self.model = joblib.load(path)

class GeneralizedModel:
    """
    A generalized model framework supporting multiple neural network architectures 
    and ensemble methods, with comprehensive data checking and cross-validation.
    
    Features:
    - Support for various NN architectures (MLP, CNN, RNN, Transformer)
    - Support for ensemble methods (XGBoost, LightGBM)
    - Flexible scaling options per feature
    - Comprehensive data quality checks
    - Multiple split and cross-validation schemes
    - Real-time single-step prediction capability
    - Model persistence and loading
    
    'model_type': 
        Available model types:
        - 'mlp': Multi-layer Perceptron for general purpose regression/classification
        - 'cnn': Convolutional Neural Network for spatial data
        - 'rnn': Recurrent Neural Network for sequential data
        - 'ensemble': Ensemble methods (Random Forest, XGBoost, etc.)
    'scaling_methods': 
        Available scaling methods:
        - STANDARD: Zero mean, unit variance scaling (best for normal distributions)
        - MINMAX: Scale to range [0,1] (best for bounded data)
        - ROBUST: Robust to outliers (best for data with outliers)
        - NONE: No scaling (use when data is already preprocessed)
    'split_method': 
        Available split methods:
        - RANDOM: Simple random split (default)
        - STRATIFIED: Maintains label distribution (for classification)
        - TIMESERIES: Respects temporal order (for time series data)
        - GROUP: Group-based splitting (when data has natural groups)
    'cv_method': 
        Available cross-validation methods:
        - KFOLD: K-fold cross validation (default)
        - STRATIFIED_KFOLD: Stratified k-fold (for classification)
        - TIME_SERIES: Time series cross validation
        - GROUP_KFOLD: Group k-fold cross validation
    """

    def __init__(self, 
                 model_type: str,
                 input_dims: List[int],
                 output_dims: List[int],
                 scaling_methods: Dict[str, ScalingMethod],
                 split_method: SplitMethod = SplitMethod.KFOLD_CV,
                 n_splits: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 **model_params: Any) -> None:
        """Initialize the generalized model."""
        self.model_type = ModelType(model_type)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.scaling_methods = scaling_methods
        self.split_method = split_method
        self.n_splits = n_splits
        self.device = device
        self.model_params = model_params
        
        self.model: Optional[Union[nn.Module, Ensemble]] = None
        self.scalers: Dict[str, Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]]] = {}
        self.data_check_results: Dict[str, Any] = {}
        self.training_history: List[Dict[str, float]] = []
        self.best_params: Dict[str, Any] = {}
        self.compiled_model: Optional[torch.jit.ScriptModule] = None
        self.is_compiled = False
        
        # tracking best model
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_model_fold = None
        
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
        self.logger = logging.getLogger(__name__)

    def create_model(self) -> Union[nn.Module, Ensemble]:
        """Factory method to create model instances based on model type."""
        input_size = sum(self.input_dims)
        output_size = sum(self.output_dims)
        
        if self.model_type == ModelType.CNN:
            return CNN(
                input_dim=input_size,
                hidden_dims=self.model_params.get('hidden_dims', [64, 128, 256]),
                output_dim=output_size,
                kernel_sizes=self.model_params.get('kernel_sizes', [3, 3, 3])
            )

        elif self.model_type in [ModelType.LSTM, ModelType.GRU, ModelType.BILSTM, ModelType.BIGRU]:
            return Recurrent(
                input_size=input_size,
                hidden_size=self.model_params.get('hidden_size', 128),
                output_size=output_size,
                model_type='lstm' if self.model_type in [ModelType.LSTM, ModelType.BILSTM] else 'gru',
                bidirectional=self.model_type in [ModelType.BILSTM, ModelType.BIGRU]
            )

        elif self.model_type == ModelType.TRANSFORMER:
            return Transformer(
                input_size=input_size,
                d_model=self.model_params.get('d_model', 512),
                nhead=self.model_params.get('nhead', 8),
                num_layers=self.model_params.get('num_layers', 6),
                output_size=output_size
            )

        elif self.model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
            return Ensemble(
                model_type=self.model_type.value,
                **self.model_params
            )
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def check_data(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality checks on all features and targets.
        Args:
            X: Feature DataFrame
            y: Target DataFrame
        """
        def _check_data_quality(data: pd.DataFrame, feature_name: str) -> DataCheckResult:
            """Perform comprehensive data quality checks on a feature."""
            series = data[feature_name]

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                # Basic statistics
                mean = series.mean()
                variance = float(series.var()) # type: ignore
                skewness = float(stats.skew(series))
                kurtosis = float(stats.kurtosis(series))

                # Distribution analysis
                distribution_type = _analyze_distribution_shape(series)

            # Outlier detection
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1 # IQR is the range within which the middle 50% of the data lies
            outlier_bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            outlier_mask = (series < outlier_bounds[0]) | (series > outlier_bounds[1])
            outlier_ratio = outlier_mask.mean()  # Proportion of outliers
            middle_data = series[(series >= outlier_bounds[0]) & (series <= outlier_bounds[1])]
            trimmed_mean = middle_data.mean()
            trimmed_std = middle_data.std()
            outlier_values = series[outlier_mask]
            outlier = False
            if len(outlier_values) > 0:
                max_deviation = max(abs((outlier_values - trimmed_mean) / trimmed_std))
                outlier = outlier_ratio > 0.3 or max_deviation > 200
                
            # Cardinality
            cardinality = len(series.unique())

            # Generate warnings and suitability flags
            warnings = []
            suitable_for_nn = True
            suitable_for_ensemble = True
            
            if abs(mean) > 0.1 and abs(variance/mean) < 0.01:
                warnings.append(f"Low variance ({abs(variance/mean):.2f})")
                suitable_for_nn = False

            if abs(skewness) > 2*5: # 2 is bit too conservative for modern ML algo
                warnings.append(f"High skewness ({skewness:.2f})")
                suitable_for_nn = False

            if abs(kurtosis) > 7*5: # 7 is bit too conservative for modern ML algo
                warnings.append(f"High kurtosis ({kurtosis:.2f})")
                suitable_for_nn = False

            if outlier:
                warnings.append(f"Contains >{outlier_ratio*100:.2f}% outliers(extreme dev:{max_deviation:.2f})")
                suitable_for_nn = False

            if distribution_type not in ["normal", "uniform", "students_t", "beta", "gauss_mix", "exponential"]:
                # highly skewed ones (lognormal, exponential, gamma, weibull)
                # heavy tails (cauchy)
                warnings.append("high-skew/heavy-tail distribution")
                suitable_for_nn = False

            return DataCheckResult(
                feature_name=feature_name,
                distribution_type=distribution_type,
                mean=mean,
                variance=variance, # type: ignore
                skewness=skewness,
                kurtosis=kurtosis,
                has_outliers=outlier,
                cardinality=cardinality,
                range_bound=(series.min(), series.max()),
                suitable_for_nn=suitable_for_nn,
                suitable_for_ensemble=suitable_for_ensemble,
                warnings=warnings
            )
            
        def _print_data_check_summary(results: Dict[str, Any]) -> None:
            """Print a formatted summary of data quality checks."""
            print("\n=== Data Quality Check Summary ===")
            print("\n(Problematic)Feature Analysis:")
            for feat, res in results['features'].items():
                if res['suitable_for_nn']:
                    continue
                print(f"\n{feat}:")
                print(f"  Distribution: {res['distribution_type']}, Mean: {res['mean']:.2f}, Variance: {res['variance']:.2f}")
                print(f"  Warnings: {', '.join(res['warnings']) if res['warnings'] else 'None'}")
                
            print(results['bad_features'])
            print(results['good_features'])
            
        def _analyze_distribution_shape(series):
            """
            Analyzes a pandas Series by testing against standard distributions,
            falling back to gaussian mixture for complex cases.
            """
            import numpy as np
            from scipy import stats
            import warnings
            # Convert to numpy array and clean data
            data = series.values
            data = data[~np.isnan(data)]
            if len(data) < 8:
                return "unif"
            # Normalize data for consistent testing
            data_norm = (data - np.mean(data)) / np.std(data)
            # Dictionary to store test results
            fits = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Test standard distributions
                _, fits['normal'] = stats.shapiro(data_norm)
                _, fits['uniform'] = stats.kstest(data, 'uniform', 
                                                args=(data.min(), data.max() - data.min()))
                # Shift data for distributions requiring positive values
                shifted_data = data - min(data) + 1e-10
                _, fits['exponential'] = stats.kstest(shifted_data, 'expon')
                # Test log-normal if data is positive
                if min(data) > 0:
                    _, fits['lognormal'] = stats.kstest(np.log(data), 'norm')
                else:
                    fits['lognormal'] = 0
                _, fits['students_t'] = stats.kstest(data_norm, 't', args=(10,))
                # Scale to [0,1] for beta
                scaled_data = (data - min(data)) / (max(data) - min(data))
                _, fits['beta'] = stats.kstest(scaled_data, 'beta', args=(2, 2))
                _, fits['gamma'] = stats.kstest(shifted_data, 'gamma', args=(2,))
                _, fits['weibull'] = stats.kstest(shifted_data, 'weibull_min', args=(2,))
                _, fits['cauchy'] = stats.kstest(data_norm, 'cauchy')
                # Get best fit among standard distributions
                best_fit = max(fits.items(), key=lambda x: x[1])
                # Check for gaussian mixture if no good standard fit
                if best_fit[1] < 0.05:
                    kde = stats.gaussian_kde(data)
                    x_grid = np.linspace(min(data), max(data), 100)
                    density = kde(x_grid)
                    # Find significant peaks
                    peaks = []
                    for i in range(1, len(density) - 1):
                        if density[i-1] < density[i] > density[i+1]:
                            peaks.append(i)
                    if len(peaks) == 2:
                        return "gauss_mix"
                return best_fit[0]
            
        results = {
            'features': {},
            'targets': {},
            'bad_features': "",
            'good_features': "",
        }
        
        # Check features
        for col in X.columns:
            check_result = _check_data_quality(X, col)
            results['features'][col] = check_result.__dict__
            
        # Check targets
        for col in y.columns:
            check_result = _check_data_quality(y, col)
            results['targets'][col] = check_result.__dict__
            
        # Generate overall recommendation
        good_features = []
        bad_features = []
        for f, r in results['features'].items():
            if r['suitable_for_nn']:
                good_features.append(f)
            else:
                bad_features.append(f)
                
        # results['good_features'] = f"Features okay for NN: \n{good_features} "
        results['bad_features'] = f"Features NOT okay for NN: \n{bad_features} "
            
        # Save results
        self.data_check_results = results
        
        # Print summary
        _print_data_check_summary(results)
        
        return results
            
    def fit(self, 
           X: Union[np.ndarray, pd.DataFrame],
           y: Union[np.ndarray, pd.DataFrame],
           test_split: float = 0.1,
           validation_split: float = 0.2,
           batch_size: int = 32,
           epochs: int = 100,
           early_stopping_patience: int = 10,
           groups: Optional[np.ndarray] = None,
           **training_params) -> Dict[str, Any]:
        """
        Train the model with cross-validation.
        
        Args:
            X: Feature data
            y: Target data
            validation_split: Proportion of data for validation
            batch_size: Batch size for training
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            **training_params: Additional training parameters
            
        Returns:
            Dictionary containing training history and metrics
        """
        
        def _get_data_splitter(
            X: pd.DataFrame, 
            y: Optional[pd.DataFrame] = None,
            groups: Optional[np.ndarray] = None,
            validation_split: float = 0.2) -> \
            Union[Iterator[Tuple[np.ndarray, np.ndarray]], 
            Tuple[np.ndarray, np.ndarray]]:
            """Get appropriate data splitter based on method."""
            if self.split_method == SplitMethod.RANDOM:
                train_idx, val_idx = train_test_split(
                    np.arange(len(X)), 
                    test_size=validation_split, 
                    random_state=42
                )
                return train_idx, val_idx

            elif self.split_method == SplitMethod.STRATIFIED:
                if y is None:
                    raise ValueError("Stratified split requires target values")
                train_idx, val_idx = train_test_split(
                    np.arange(len(X)), 
                    test_size=validation_split,
                    stratify=y,
                    random_state=42
                )
                return train_idx, val_idx

            elif self.split_method == SplitMethod.TIMESERIES:
                split_point = int(len(X) * (1 - validation_split))
                return np.arange(split_point), np.arange(split_point, len(X))

            elif self.split_method == SplitMethod.GROUP:
                if groups is None:
                    raise ValueError("Group-based split requires group labels")
                group_kfold = GroupKFold(n_splits=2)  # 2 splits for train/val
                return next(group_kfold.split(X, groups=groups))

            elif self.split_method == SplitMethod.KFOLD_CV:
                kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
                return kfold.split(X)

            elif self.split_method == SplitMethod.STRATIFIED_KFOLD_CV:
                if y is None:
                    raise ValueError("Stratified k-fold requires target values")
                stratified_kfold = StratifiedKFold(
                    n_splits=self.n_splits, 
                    shuffle=True, 
                    random_state=42
                )
                return stratified_kfold.split(X, y)

            elif self.split_method == SplitMethod.TIMESERIES_CV:
                tscv = TimeSeriesSplit(n_splits=self.n_splits)
                return tscv.split(X)

            elif self.split_method == SplitMethod.GROUP_KFOLD_CV:
                if groups is None:
                    raise ValueError("Group k-fold requires group labels")
                group_kfold = GroupKFold(n_splits=self.n_splits)
                return group_kfold.split(X, groups=groups)

            else:
                raise ValueError(f"Unsupported split method: {self.split_method}")

        def _train_fold(X_train, y_train, X_val, y_val, 
                       batch_size, epochs, early_stopping_patience,
                       fold, **training_params) -> Dict[str, Any]:
            """Train a single fold or split."""
            # Create and initialize model
            self.model = self.create_model()
            if isinstance(self.model, nn.Module):
                self.model = self.model.to(self.device)

            # Train based on model type
            if isinstance(self.model, nn.Module):
                history = self._train_neural_network(
                    X_train, y_train, X_val, y_val,
                    batch_size, epochs, early_stopping_patience,
                    **training_params
                )
            else:  # Ensemble models
                history = self._train_ensemble(
                    X_train, y_train, X_val, y_val,
                    **training_params
                )

            # Calculate validation loss
            val_loss = min(h['val_loss'] for h in history)
            
            # Update best model if current model is better
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_fold = fold
                # Save model state
                if isinstance(self.model, nn.Module):
                    self.best_model_state = self.model.state_dict()
                else:  # Ensemble models
                    self.best_model_state = self.model

            return {
                'fold': fold,
                'history': history,
                'best_val_loss': val_loss
            }

        def _evaluate_on_test_set(X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
            """Evaluate final model on test set with optimized batch prediction."""
            if not getattr(self, '_inference_ready', False):
                self._prepare_inference()

            # Scale the entire test set at once using pandas
            X_test_scaled = X_test.copy()
            for col, scaler in self.scalers.items():
                if scaler is not None:
                    X_test_scaled[col] = scaler.transform(X_test[[col]])

            if isinstance(self.model, nn.Module):
                # Batch predict for compiled neural networks
                X_tensor = torch.FloatTensor(X_test_scaled.values).to(self.device)
                with torch.no_grad():  # Ensure no gradients are computed
                    predictions = self.compiled_model(X_tensor).cpu().numpy() # type: ignore
            else:
                # Batch predict for ensemble models
                predictions = self.model.predict(X_test_scaled.values) # type: ignore

            # Ensure predictions have correct shape and calculate metrics
            if predictions.ndim == 1: # type: ignore
                predictions = predictions.reshape(-1, 1) # type: ignore

            diff = y_test.values - predictions
            mse = (diff ** 2).mean()

            return {
                'test_mse': mse,
                'test_rmse': np.sqrt(mse),
                'test_mae': np.abs(diff).mean(),
                'test_r2': 1 - ((diff ** 2).sum() / 
                               ((y_test.values - y_test.values.mean()) ** 2).sum())
            }

        def _print_cv_summary(results: List[Dict[str, Any]], 
                                      is_cv: bool) -> Dict[str, Any]:
            """Summarize training results for both CV and single split cases."""
            val_losses = [r['best_val_loss'] for r in results]
            summary = {
                'split_method': self.split_method.value,
                'is_cross_validation': is_cv,
                'n_splits': len(results),
                'best_val_loss': min(val_losses),
                'avg_val_loss': np.mean(val_losses),
                'std_val_loss': np.std(val_losses) if is_cv else 0,
                'fold_results': results
            }
            # Print summary
            print("\n=== Training Summary ===")
            print(f"Split method: {summary['split_method']}")
            print(f"Cross-validation: {'Yes' if is_cv else 'No'}")
            print(f"Number of splits: {summary['n_splits']}")
            print(f"Best validation loss: {summary['best_val_loss']:.4f}")
            if is_cv:
                print(f"Average validation loss: {summary['avg_val_loss']:.4f}")
                print(f"Standard deviation: {summary['std_val_loss']:.4f}")
            print("\n")
            return summary

        def _print_final_summary(metrics: Dict[str, Any]) -> None:
            """Print final model performance summary."""
            print("\n" + "="*50)
            print("FINAL MODEL PERFORMANCE")
            print("="*50)

            # Cross-validation results
            if metrics['is_cv']:
                print("\nCross-validation Performance:")
                print(f"Mean CV Loss: {np.mean([r['best_val_loss'] for r in metrics['cv_results']]):.4f}")
                print(f"Best CV Loss: {metrics['best_val_loss']:.4f}")
            else:
                print("\nValidation Performance:")
                print(f"Validation Loss: {metrics['best_val_loss']:.4f}")

            # Test set performance
            print("\nTest Set Performance:")
            for metric, value in metrics['test_metrics'].items():
                print(f"{metric}: {value:.4f}")

            print("\nModel is ready for deployment.")
            print("="*50)

        # Convert inputs to DataFrames if needed
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        y = pd.DataFrame(y) if not isinstance(y, pd.DataFrame) else y.copy()

        # Validate input dimensions
        if len(X.columns) != sum(self.input_dims):
            raise ValueError(f"Expected {sum(self.input_dims)} input features, got {len(X.columns)}")
        if len(y.columns) != sum(self.output_dims):
            raise ValueError(f"Expected {sum(self.output_dims)} output features, got {len(y.columns)}")

        # First split off the holdout test set
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42
        )

        # Scale features using only training data statistics
        self.logger.info("Scaling features...")
        X_train_scaled = self._scale_full(X_train_full)
        # Scale test set using training data statistics
        X_test_scaled = self._scale_full(X_test)


        print("\nX statistics:")
        print(f"X_scaled mean: {X_train_scaled.mean().mean()}")
        print(f"X_scaled std: {X_train_scaled.std().mean()}")
        print(f"X_scaled min: {X_train_scaled.min().min()}")
        print(f"X_scaled max: {X_train_scaled.max().max()}")

        print("\ny statistics:")
        print(f"y mean: {y_train_full.mean().mean()}")
        print(f"y std: {y_train_full.std().mean()}")
        print(f"y min: {y_train_full.min().min()}")
        print(f"y max: {y_train_full.max().max()}")
        print("\n")

        # Perform data quality checks
        self.logger.info("Performing data quality checks...")
        quality_results = self.check_data(X_train_scaled, y_train_full)

        # Get appropriate splitter for CV/validation
        splits = _get_data_splitter(X_train_scaled, y_train_full, groups, validation_split)
        
        # Initialize training results
        cv_results = []
        best_val_loss = float('inf')
        
        # Determine if using CV or single split
        is_cv = self.split_method.value.endswith('_cv')
        
        if is_cv:
            self.logger.info("Cross-validation Enabled")
            # Cross-validation training
            for fold, (train_idx, val_idx) in enumerate(splits):
                print("\n")
                self.logger.info(f"Training fold {fold + 1}/{self.n_splits}")

                # Split data
                X_train, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
                y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

                # Train fold
                fold_results = _train_fold(
                    X_train, y_train, X_val, y_val,
                    batch_size, epochs, early_stopping_patience,
                    fold, **training_params
                )

                cv_results.append(fold_results)

                # Update best parameters if needed
                if fold_results['best_val_loss'] < best_val_loss:
                    best_val_loss = fold_results['best_val_loss']
                    self.best_params = training_params.copy()
                    self.best_model_state = self.model.state_dict() if isinstance(self.model, nn.Module) else self.model

        else:
            # Single validation split
            train_idx, val_idx = splits
            X_train, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
            y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

            fold_results = _train_fold(
                X_train, y_train, X_val, y_val,
                batch_size, epochs, early_stopping_patience,
                0, **training_params
            )

            cv_results.append(fold_results)
            best_val_loss = fold_results['best_val_loss']
            self.best_params = training_params.copy()
            self.best_model_state = self.model.state_dict() if isinstance(self.model, nn.Module) else self.model

        # Print CV summary
        _print_cv_summary(cv_results, is_cv)

        # Train final model on full training dataset using best parameters
        self.logger.info("Training final model on full training dataset with best parameters...")
        self.model = self.create_model()
        if isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)
            final_history = self._train_neural_network(
                X_train_scaled, y_train_full, 
                X_test_scaled, y_test,  # Use test set for validation during final training
                batch_size, epochs, early_stopping_patience,
                **self.best_params
            )
        else:
            final_history = self._train_ensemble(
                X_train_scaled, y_train_full, 
                X_test_scaled, y_test,
                **self.best_params
            )

        # Evaluate final model on test set
        test_metrics = _evaluate_on_test_set(X_test_scaled, y_test)

        # Update best model state with final model
        self.best_model_state = self.model.state_dict() if isinstance(self.model, nn.Module) else self.model

        final_metrics = {
            'cv_results': cv_results,
            'is_cv': is_cv,
            'best_val_loss': best_val_loss,
            'final_loss': final_history[-1]['val_loss'],
            'test_metrics': test_metrics,
            'best_params': self.best_params
        }

        _print_final_summary(final_metrics)
        return final_metrics

    def _scale_full(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale the entire dataset before splitting.

        Args:
            X: Full feature DataFrame

        Returns:
            Scaled DataFrame with original column names
        """
        
        def _get_scaler(scaling_method: ScalingMethod) -> Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]]:
            """Get the appropriate scaler based on scaling method."""
            if scaling_method == ScalingMethod.STANDARD:
                return StandardScaler()
            elif scaling_method == ScalingMethod.MINMAX:
                return MinMaxScaler()
            elif scaling_method == ScalingMethod.ROBUST:
                return RobustScaler()
            elif scaling_method == ScalingMethod.NONE:
                return None
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        X_scaled = X.copy()

        for col in X.columns:
            if col not in self.scalers:
                scaler_type = self.scaling_methods.get(col, ScalingMethod.STANDARD)
                self.scalers[col] = _get_scaler(scaler_type)

            if self.scalers[col] is not None:
                X_scaled[col] = self.scalers[col].fit_transform(X[[col]])  # type: ignore

        return X_scaled
    
    def _scale_single(self, features: np.ndarray) -> np.ndarray:
        """Scale features for single prediction."""
        scaled_features = features.copy()
        for i, (_, scaler) in enumerate(self.scalers.items()):
            if scaler is not None:
                scaled_features[:, i] = scaler.transform(features[:, [i]]).ravel()
        return scaled_features
    
    def _train_neural_network(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        batch_size: int,
        epochs: int,
        early_stopping_patience: int,
        **training_params
    ) -> List[Dict[str, float]]:
        """Train neural network models."""
        if not isinstance(self.model, nn.Module):
            raise TypeError("Model must be a neural network for this training method")
            
        optimizer = Adam(self.model.parameters(), 
                        lr=training_params.get('learning_rate', 0.001))
        criterion = nn.MSELoss()
        
        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train.values),
            torch.FloatTensor(y_train.values)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        history = []
        patience_counter = 0
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_X = torch.FloatTensor(X_val.values).to(self.device)
                val_y = torch.FloatTensor(y_val.values).to(self.device)
                val_outputs = self.model(val_X)
                val_loss = criterion(val_outputs, val_y).item()
            
            # Record metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            history.append(metrics)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
                
            if (epoch + 1) % 1 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )
                
        return history
    
    def _train_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        **training_params
    ) -> List[Dict[str, float]]:
        """Train ensemble models."""
        if not isinstance(self.model, Ensemble):
            raise TypeError("Model must be an Ensemble for this training method")

        # Train the ensemble model
        eval_metric = 'rmse' if self.model_type == ModelType.XGBOOST else 'l2'
        eval_set = [(X_train.values, y_train.values), (X_val.values, y_val.values)]

        history = []
        if self.model_type == ModelType.XGBOOST:
            self.model.fit(
                X_train, 
                y_train,
                eval_set=eval_set,
                eval_metric=eval_metric,
                **training_params
            )

            # Extract training history
            if hasattr(self.model.model, 'evals_result'):
                evals_result = self.model.model.evals_result() # type: ignore
                if evals_result and 'validation_0' in evals_result:
                    metrics = evals_result['validation_0'].get(eval_metric, [])
                    val_metrics = evals_result['validation_1'].get(eval_metric, [])

                    history = [
                        {
                            'epoch': i + 1,
                            'train_loss': train_loss,
                            'val_loss': val_loss
                        }
                        for i, (train_loss, val_loss) in enumerate(zip(metrics, val_metrics))
                    ]

        else:  # lightgbm
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                eval_metric=eval_metric,
                **training_params
            )

            # Extract LightGBM history
            if hasattr(self.model.model, 'evals_result_'):
                evals_result = self.model.model.evals_result_ # type: ignore
                if evals_result and 'training' in evals_result:
                    metrics = evals_result['training'].get(eval_metric, [])
                    val_metrics = evals_result['valid_1'].get(eval_metric, [])

                    history = [
                        {
                            'epoch': i + 1,
                            'train_loss': train_loss,
                            'val_loss': val_loss
                        }
                        for i, (train_loss, val_loss) in enumerate(zip(metrics, val_metrics))
                    ]

        return history
    
    def _prepare_inference(self) -> None:
        """
        One-time preparation for fast inference, including model compilation.
        Called automatically on first prediction.
        """
        if self.model is None:
            raise ValueError("Model must be trained before inference")
            
        # Preallocate arrays for scaling
        self.scaled_features = np.zeros((1, sum(self.input_dims)), dtype=np.float32)
        
        # Set up inference path for neural networks or ensembles
        if isinstance(self.model, nn.Module):
            self.model.eval()  # Set to evaluation mode permanently
            
            # Preallocate tensor for neural network
            self.inference_tensor = torch.zeros(1, sum(self.input_dims), 
                                              dtype=torch.float32, 
                                              device=self.device)
            
            # Force compilation with proper error handling
            try:
                # Test input
                example_input = torch.randn(1, sum(self.input_dims), device=self.device)
                
                # First test forward pass
                _ = self.model(example_input)
                
                # Then try compilation
                self.compiled_model = torch.jit.script(self.model)
                
                # Test compiled model
                _ = self.compiled_model(example_input)
                self.is_compiled = True
                
            except Exception as e:
                raise RuntimeError(f"Model compilation failed: {str(e)}")
                
            if not self.compiled_model:
                raise RuntimeError("Model compilation failed silently")
                
            # Assign compiled prediction path
            self._predict_impl = self._predict_compiled_nn
        else:
            # Assign ensemble prediction path
            self._predict_impl = self._predict_ensemble
            
        self._inference_ready = True
    
    def _scale_features_fast(self, features: np.ndarray) -> None:
        """
        Apply scaling using existing scaler transform functions.
        Updates self.scaled_features in-place.
        Args:
            features: 1D input array to be scaled
        """
        feature_idx = 0
        features_2d = features.reshape(1, -1)
        
        for _, scaler in self.scalers.items():
            if scaler is not None:
                self.scaled_features[0, feature_idx] = scaler.transform(features_2d[:, [feature_idx]])[0]
            else:
                self.scaled_features[0, feature_idx] = features[feature_idx]
            feature_idx += 1
    
    def _predict_compiled_nn(self, features: np.ndarray) -> np.ndarray:
        """Ultra-optimized path for compiled neural network inference."""
        self._scale_features_fast(features)
        self.inference_tensor[0].copy_(torch.from_numpy(self.scaled_features[0]))
        return self.compiled_model(self.inference_tensor)[0].cpu().detach().numpy() # type: ignore
    
    def _predict_ensemble(self, features: np.ndarray) -> np.ndarray:
        """Ultra-optimized path for ensemble model inference."""
        self._scale_features_fast(features)
        return self.model.predict(self.scaled_features)[0] # type: ignore
    
    def predict_single(self, features: np.ndarray) -> np.ndarray:
        """
        Ultra-optimized single prediction method with proper scaling.
        Args:
            features: 1D numpy array of shape (n_features,) with dtype float32
        Returns:
            1D numpy array of predictions
        """
        if not getattr(self, '_inference_ready', False):
            self._prepare_inference()
            
        return self._predict_impl(features)
        
    def save(self, path: Union[str, Path]) -> None:
        """
        Save only the best model and necessary components.
        
        Args:
            path: Path to save the model
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.model is None or self.best_model_state is None:
            raise ValueError("No model to save or no best model state available")

        # Save model architecture and parameters
        model_config = {
            'model_type': self.model_type.value,
            'input_dims': self.input_dims,
            'output_dims': self.output_dims,
            'scaling_methods': {k: v.value for k, v in self.scaling_methods.items()},
            'split_method': self.split_method.value,
            'n_splits': self.n_splits,
            'model_params': self.model_params,
            'device': self.device,
            'is_compiled': self.is_compiled,
            'best_val_loss': self.best_val_loss,
            'best_model_fold': self.best_model_fold
        }

        # Save best model state
        if isinstance(self.model, nn.Module):
            torch.save({
                'model_config': model_config,
                'state_dict': self.best_model_state
            }, save_path / 'best_model.pt')
        else:  # Ensemble models
            with open(save_path / 'model_config.json', 'w') as f:
                json.dump(model_config, f)
            if isinstance(self.best_model_state, Ensemble):
                self.best_model_state.save_model(str(save_path / 'best_model.txt'))

        # Save scalers
        import joblib
        joblib.dump(self.scalers, save_path / 'scalers.pkl')

        # Save training history and metadata
        with open(save_path / 'metadata.json', 'w') as f:
            json.dump({
                'data_check_results': self.data_check_results,
                'training_history': self.training_history,
                'best_params': self.best_params
            }, f, indent=4)

        self.logger.info(f"Best model (fold {self.best_model_fold}) saved to {save_path}")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'GeneralizedModel':
        """
        Load the saved best model and its components.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded GeneralizedModel instance with best model
        """
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Path {load_path} does not exist")

        # Load model configuration
        if (load_path / 'best_model.pt').exists():
            # PyTorch model
            checkpoint = torch.load(load_path / 'best_model.pt')
            model_config = checkpoint['model_config']
        else:
            # Ensemble model
            with open(load_path / 'model_config.json', 'r') as f:
                model_config = json.load(f)

        # Create instance with saved parameters
        instance = cls(
            model_type=model_config['model_type'],
            input_dims=model_config['input_dims'],
            output_dims=model_config['output_dims'],
            scaling_methods={k: ScalingMethod(v) for k, v in model_config['scaling_methods'].items()},
            split_method=SplitMethod(model_config['split_method']),
            n_splits=model_config['n_splits'],
            device=model_config['device'],
            **model_config['model_params']
        )

        # Create and load best model
        instance.model = instance.create_model()
        if isinstance(instance.model, nn.Module):
            instance.model.load_state_dict(checkpoint['state_dict'])
            instance.model.to(instance.device)
        else:  # Ensemble models
            instance.model.load_model(str(load_path / 'best_model.txt'))

        # Load scalers
        import joblib
        instance.scalers = joblib.load(load_path / 'scalers.pkl')

        # Load metadata and set best model information
        with open(load_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
            instance.data_check_results = metadata['data_check_results']
            instance.training_history = metadata['training_history']
            instance.best_params = metadata['best_params']

        instance.best_val_loss = model_config.get('best_val_loss', float('inf'))
        instance.best_model_fold = model_config.get('best_model_fold', None)
        instance.best_model_state = instance.model.state_dict() if isinstance(instance.model, nn.Module) else instance.model

        return instance
    
# ==============================================================================
# ==================================example=====================================
# ==============================================================================

def example_bilstm():
    # Generate sample time series data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    n_outputs = 1

    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=n_samples)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)],
        index=dates
    )
    y = pd.DataFrame(
        np.sin(np.arange(n_samples) * 0.1) + np.random.randn(n_samples) * 0.1,
        columns=['target'],
        index=dates
    )

    # Initialize BiLSTM model
    bilstm_model = GeneralizedModel(
        model_type='bilstm',
        input_dims=[n_features],  # Input dimension matches number of features
        output_dims=[n_outputs],  # Output dimension matches number of targets
        scaling_methods={
            f'feature_{i}': ScalingMethod.STANDARD for i in range(n_features)
        },
        hidden_size=64,          # Size of LSTM hidden state
        num_layers=2,            # Number of LSTM layers
        dropout=0.2             # Dropout rate
    )

    # Train the model
    training_history = bilstm_model.fit(
        X=X,
        y=y,
        batch_size=32,
        epochs=100,
        early_stopping_patience=10,
        learning_rate=0.001
    )

    # Make predictions
    new_data = pd.DataFrame(
        np.random.randn(5, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    predictions = bilstm_model.predict_single(new_data) # type: ignore

    # Save the model
    bilstm_model.save('bilstm_model')

    # Load the model later
    loaded_model = GeneralizedModel.load('bilstm_model')

def example_xgboost():
    # Generate sample tabular data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_outputs = 1

    # Create sample data
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.DataFrame(
        (X.sum(axis=1) > 0).astype(float) + np.random.randn(n_samples) * 0.1,
        columns=['target']
    )

    # Initialize XGBoost model
    xgb_model = GeneralizedModel(
        model_type='xgboost',
        input_dims=[n_features],  # Input dimension matches number of features
        output_dims=[n_outputs],  # Output dimension matches number of targets
        scaling_methods={
            f'feature_{i}': ScalingMethod.STANDARD for i in range(n_features)
        },
        # XGBoost specific parameters
        n_estimators=100,        # Number of boosting rounds
        max_depth=6,            # Maximum tree depth
        learning_rate=0.1,      # Learning rate
        subsample=0.8,          # Subsample ratio of training instances
        colsample_bytree=0.8,   # Subsample ratio of columns when constructing trees
        objective='reg:squarederror'  # Regression objective
    )

    # Train the model
    training_history = xgb_model.fit(
        X=X,
        y=y,
        early_stopping_patience=10,
        eval_metric='rmse'
    )

    # Make predictions
    new_data = pd.DataFrame(
        np.random.randn(5, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    predictions = xgb_model.predict_single(new_data) # type: ignore

    # Feature importance analysis (XGBoost specific)
    if hasattr(xgb_model.model.model, 'feature_importances_'): # type: ignore
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': xgb_model.model.model.feature_importances_ # type: ignore
        }).sort_values('importance', ascending=False)
        print("\nFeature Importances:")
        print(importances)

    # Save the model
    xgb_model.save('xgboost_model')

    # Load the model later
    loaded_model = GeneralizedModel.load('xgboost_model')

if __name__ == "__main__":
    example_bilstm()
    example_xgboost()