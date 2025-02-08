import math
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

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
    suitable: bool
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

def CheckDist(df: pd.DataFrame) -> Dict[str, Any]:
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
        suitable = True
        
        if abs(mean) > 0.1 and abs(variance/mean) < 0.01:
            warnings.append(f"Low variance ({abs(variance/mean):.2f})")
            suitable = False
        if abs(skewness) > 2*5: # 2 is bit too conservative for modern ML algo
            warnings.append(f"High skewness ({skewness:.2f})")
            suitable = False
        if abs(kurtosis) > 7*5: # 7 is bit too conservative for modern ML algo
            warnings.append(f"High kurtosis ({kurtosis:.2f})")
            suitable = False
        if outlier:
            warnings.append(f"Contains >{outlier_ratio*100:.2f}% outliers(extreme dev:{max_deviation:.2f})")
            suitable = False
        if distribution_type not in ["normal", "uniform", "students_t", "beta", "gauss_mix", "exponential"]:
            # highly skewed ones (lognormal, exponential, gamma, weibull)
            # heavy tails (cauchy)
            warnings.append("high-skew/heavy-tail distribution")
            suitable = False
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
            suitable=suitable,
            warnings=warnings
        )
        
    def _print_data_dist_summary(df, results) -> None:
        print("\n=== Data Dist Summary (Before Split) ===")
        # columns summary
        print(f"\nColumns(X) ({df.shape[1]}): {df.shape[0]:,} samples")

        feature_stats = pd.DataFrame({
            'Mean': pd.Series({col: results['columns'][col]['mean'] for col in df.columns}),
            'Std': pd.Series({col: math.sqrt(results['columns'][col]['variance']) for col in df.columns}),
            'Min': df.min(),
            'Max': df.max(),
            'Skewness': pd.Series({col: results['columns'][col]['skewness'] for col in df.columns}),
            'Kurtosis': pd.Series({col: results['columns'][col]['kurtosis'] for col in df.columns}),
            'Unique': df.nunique(),
            'Distribution': pd.Series({col: results['columns'][col]['distribution_type'] for col in df.columns}),
            'Suitable': pd.Series({col: results['columns'][col]['suitable'] for col in df.columns}),
        }).round(4)
        print(feature_stats)

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
        'columns': {},
    }
    
    # Check columns
    for col in df.columns:
        column_result = _check_data_quality(df, col)
        results['columns'][col] = column_result.__dict__

    # Print summary
    _print_data_dist_summary(df, results)
    return results