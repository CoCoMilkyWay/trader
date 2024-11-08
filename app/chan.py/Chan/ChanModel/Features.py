from typing import Optional, Dict, List
import copy

'''
+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Technique                     | Description                                                                                                                               |
+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Combining or Encoding         | - Lag Features: Use previous values in sequences for trend analysis.                                                                      |
|                               | - Aggregated Sequence Statistics: Calculate rolling window statistics (mean, max, min) to analyze sequence patterns.                      |
|                               | - Embedding/Encoding Sequences: Create embeddings or use aggregations to capture trends, capturing amplitude and sequence.                |
+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Feature Interactions          | - Polynomial/Pairs: Multiply or square features to capture non-linear relationships.                                                      |
|                               | - Binning & Cross Features: Bin continuous values, then cross bins to identify unique subgroups.                                          |
+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Target Encoding               | - Encode categorical variables based on target means, capturing the relationship between categories and targets.                          |
+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Domain-Specific Features      | - Construct Indicators: Use domain-specific indicators (e.g., price volatility) to gain insights.                                         |
|                               | - Aggregate & Count Features: Summarize counts or frequency of elements for predictive strength.                                          |
+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Clustering & Dimensionality   | - Clustering: Group observations, creating a new feature for the cluster each observation belongs to.                                     |
| Reduction                     | - Principal Component Analysis (PCA): Reduce dimensionality while preserving essential information through top components.                |
+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Handling Missing Values       | - Indicator for Missing Values: Add binary flags to represent missing data for better interpretability.                                   |
| and Outliers                  | - Capping/Transforming Outliers: Cap or transform outliers using log or square root to prevent data distortion.                           |
+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Sequence Encoding Techniques  | - Weighted/Polynomial Encoding: Weight or apply polynomial functions to encode sequences with meaningful order.                           |
|                               | - Embedding (for NN): Use embeddings in neural networks to represent sequences in lower dimensions.                                       |
|                               | - Sinusoidal Encoding: Borrowed from transformer models, captures temporal information within sequences.                                  |
|                               | - Fourier/Frequency Encoding: Encode sequences based on frequency, capturing cyclic patterns.                                             |
|                               | - Auto-Encoder-Based Encoding (LSTM/GRU): Use recurrent networks to capture dependencies and compress sequence patterns.                  |
|                               | - Lagged Features: Treat lagged values as additional features, capturing shifts over time in sequence.                                    |
one-hot(unique category value), ordinal(with meaningful order), frequency(probability), hash encoding(high-cardinality: ID/names)                                           |
+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
'''

'''
qlib/data/dataset
| Processor             | Use for Learn | Use for Infer | Purpose                                 | Data Type Processed        |
|-----------------------|---------------|---------------|-----------------------------------------|----------------------------|
| `DropnaProcessor`     | No            | Yes           | Drops rows with NaN values              | Any (numeric, text, etc.)  |
| `DropnaLabel`         | No            | No            | Drops rows with NaN in the label column | Label data                 |
| `DropCol`             | No            | Yes           | Drops specified columns                 | Any                        |
| `FilterCol`           | No            | Yes           | Keeps only specified columns            | Any                        |
| `TanhProcess`         | No            | Yes           | Denoises data using the tanh function   | Numeric                    |
| `ProcessInf`          | No            | Yes           | Replaces infinities with mean values    | Numeric                    |
| `Fillna`              | No            | Yes           | Fills NaN values with a specified value | Numeric                    |
| `MinMaxNorm`          | Yes           | Yes           | Normalizes data to [0, 1] range         | Numeric                    |
| `ZScoreNorm`          | Yes           | Yes           | Applies Z-score normalization           | Numeric                    |
| `RobustZScoreNorm`    | Yes           | Yes           | Robust Z-score normalization with MAD   | Numeric                    |
| `CSZScoreNorm`        | No            | Yes           | Cross-sectional Z-score normalization   | Numeric                    |
| `CSRankNorm`          | No            | Yes           | Cross-sectional rank normalization      | Numeric                    |
| `CSZFillna`           | No            | Yes           | Cross-sectional fill of NaN values      | Numeric                    |
| `HashStockFormat`     | No            | Yes           | Converts data to hashed stock format    | Any                        |
| `TimeRangeFlt`        | No            | Yes           | Filters data by a specified time range  | Time-series                |

Inst_Processors:
    Resample1minProcessor
    ResampleNProcessor
'''
'''
simple raw factors:
| Type               | Factors                                                                                         |
|--------------------|-------------------------------------------------------------------------------------------------|
| K-Bar              | KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2                                          |
| Price              | OPEN, HIGH, LOW, VWAP                                                                           |
| Volume             | VOLUME                                                                                          |
| Rolling Operators  | ROC, MA, STD, BETA, RSQR, RESI, MAX, MIN, QTLU, QTLD, RANK, RSV, IMAX, IMIN, IMXD, CORR, CORD,  |
|                    | CNTP, CNTN, CNTD, SUMP, SUMN, SUMD, VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD                        |
'''

def pause():
    import time
    time.sleep(1000)
    return

# feature map
DEFAULT = -10.0
m:Dict[str, List[float]] = {
#       name:str:                    [default_value:? data_consumed:int]
# PA: price_action
    # CP: chart_pattern
        # if a con/diverging nexus/jiatou exist
        'PA_CP_exist_nexus'         :[DEFAULT,  0,],
        # multiple chart patterns exist at the same time
        'PA_CP_exist_nexus_mult'    :[DEFAULT,  0,],
        # nexus type (not using on-hot since it is important)
        'PA_CP_first_entry'         :[DEFAULT,  0,],
        'PA_CP_is_channel'          :[DEFAULT,  0,],
        'PA_CP_is_rect'             :[DEFAULT,  0,],
        'PA_CP_is_meg_sym'          :[DEFAULT,  0,],
        'PA_CP_is_meg_brk_far'      :[DEFAULT,  0,],
        'PA_CP_is_meg_rev_bak'      :[DEFAULT,  0,],
        'PA_CP_is_tri_sym'          :[DEFAULT,  0,],
        'PA_CP_is_tri_brk_far'      :[DEFAULT,  0,],
        'PA_CP_is_tri_rev_bak'      :[DEFAULT,  0,],
        'PA_CP_is_flag'             :[DEFAULT,  0,],
        
        'PA_CP_entry_dir'           :[DEFAULT,  0,],
        'PA_CP_num_vertices'        :[DEFAULT,  0,],
        'PA_CP_far_cons'            :[DEFAULT,  0,],
        'PA_CP_near_cons'           :[DEFAULT,  0,],
        'PA_CP_top_slope'           :[DEFAULT,  0,],
        'PA_CP_bot_slope'           :[DEFAULT,  0,],
        'PA_CP_top_residue'         :[DEFAULT,  0,],
        'PA_CP_bot_residue'         :[DEFAULT,  0,],
        
}
# key_index = list(m.keys()).index('b')  # Output: 1
# value_index = list(m.values()).index(2)  # Output: 1

# python would reserve a virtual space from MMU of CPUs,
# after constantly appending items to an array,
# it would only occasionally req more space from malloc, 
# which is typically 1.125 to 1.5 times its current size

class CFeatures: # Features Snapshot for a potential buy/sell point
    def __init__(self, initFeat=None):
        self.empty_feature_page:Dict[str, float] = dict(zip(m.keys(), map(lambda l: float(l[0]), m.values())))
        self.feature_history:Dict[str, List[float]] = dict(zip(m.keys(), []))
        self.label_history:List[float] = []
        self.num_features_updates:int = 0
        self.num_label_updates:int = 0
        if initFeat is None:
            self.refresh_feature_page()
            # from pprint import pprint
            # pprint(self._features)
        else:
            self._features = dict(initFeat)

    def items(self):
        yield from self._features.items()

    def __getitem__(self, k):
        return self._features[k]

    def refresh_feature_page(self):
        self._features = copy.deepcopy(self.empty_feature_page)

    def add_feat(self, inp1:str, inp2:float):
        # self.__features.update({inp1: inp2})
        self._features[inp1] = inp2

    def update_features_array(self): # for training deep learning algorithm
        # if self.label_updated:
        for key, value in self._features.items():
            self.feature_history.setdefault(key, []).append(round(value, 2)) # TODO
        self.num_features_updates += 1
        #     self.label_updated = False
        # else:
        #     print('Err: failed to update features array')
        #     pause()

    def update_label_list(self, new_label:float): # labels contain future information
        # if self.label_updated:
        #     print('Err: failed to update label list')
        #     pause()
        # else:
        self.label_history.append(round(new_label,2))
        self.num_label_updates += 1
        # self.label_updated = True
        # return True
    
    def get_pending_label_updates(self) -> int:
        return self.num_features_updates - self.num_label_updates
    