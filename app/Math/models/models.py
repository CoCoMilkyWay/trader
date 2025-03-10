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
    STRATIFIED = "stratified"       # Stratified split (preserves label distribution) (for classification)
    TIMESERIES = "timeseries"       # Temporal order preserved
    GROUP = "group"                 # Group-based split
    
    # Cross-validation methods
    KFOLD_CV = "kfold_cv"          # K-fold cross validation
    STRATIFIED_KFOLD_CV = "stratified_kfold_cv"  # Stratified k-fold (for classification)
    TIMESERIES_CV = "timeseries_cv"  # Time series cross validation
    GROUP_KFOLD_CV = "group_kfold_cv" # Group k-fold cross validation