from typing import Optional

'''
+-------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------+
| Technique                     | Description                                                                                                                                |
+-------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------+
| Combining or Encoding         | - Lag Features: Use previous values in sequences for trend analysis.                                                                      |
| Sequences                     | - Aggregated Sequence Statistics: Calculate rolling window statistics (mean, max, min) for sequence patterns.                             |
|                               | - Embedding/Encoding Sequences: Create embeddings or aggregate key statistics for sequences to capture trends and amplitude.               |
+-------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------+
| Feature Interactions          | - Polynomial/Pairs: Multiply or square features to capture non-linear relationships.                                                      |
|                               | - Binning & Cross Features: Bin continuous values into categories, cross bins for unique subgroups.                                       |
+-------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------+
| Target Encoding               | - Encode categorical variables by target means to capture category-target relationships.                                                   |
+-------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------+
| Domain-Specific Features      | - Construct Indicators: Use domain-specific indicators (e.g., price volatility) for insight into data.                                    |
|                               | - Aggregate & Count Features: Summarize counts or frequency of repeating elements for predictive strength.                                |
+-------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------+
| Clustering & Dimensionality   | - Clustering: Cluster observations and create a new feature identifying the cluster.                                                      |
| Reduction                     | - Principal Component Analysis (PCA): Use top components to reduce dimensionality while preserving key information.                       |
+-------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------+
| Handling Missing Values       | - Indicator for Missing Values: Add binary flags to represent missing data.                                                               |
| and Outliers                  | - Capping/Transforming Outliers: Cap or transform (log, square root) outliers to prevent distortion.                                      |
+-------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------+

# there are many choices for XGBoost-like algorithm to learn [a,b,c], where a,b,c are real number
# and their sequence matters:
#   1. weighted/polynomial-encoding
#   2. embedding (for NN)
#   3. sinusoidal encoding (from transformer models to capture temporal info)
#   4. fourier/frequency encoding
#   5. auto-encoder-based encoding (LSTM/GRU)
#   6. lagged features

'''

class CFeatures:
    def __init__(self, initFeat=None):
        self.__features = {} if initFeat is None else dict(initFeat)

    def items(self):
        yield from self.__features.items()

    def __getitem__(self, k):
        return self.__features[k]

    def add_feat(self, inp1, inp2: Optional[float] = None):
        if inp2 is None:
            self.__features.update(inp1)
        else:
            self.__features.update({inp1: inp2})
