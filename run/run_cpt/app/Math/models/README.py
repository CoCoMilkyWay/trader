"""
prompt to generate ML class:

this task is huge, take it slowly, and dedicate, try put in single class, unless there are benefit to split
provide a generalized pytorch model class which able to:
    0.generalized: able to use most NN and ensemble methods, regression/classification, multi input/output
    1.scalar: able to choose scalar scheme for each feature
    2.checking: make sure data are fit for NN/ensemble algo to effectively learn after scaling:
        e.g. bound/distribution(type/tail-heaviness)/Mean/Variance/Skewness/Kurtosis/Outliers/Cardinality/inter-correlation/
        generate a report(json and console print it) to assess the dataset quality(for each feature and label) after scaling, different for NN and ensemble methods(because they have different focus)
        show which check failed for which feature failed for which model
    3.split: able to choose dozen different splitting schemes(into train/val/test)
        enable cross validation by default(different k-fold/timeseries split schemes)
        auto-choosing the best hyperparameter over iterations, provided tested not overfit
        show cross validation report after train/tune
    4.train: for multiple labels, this would be a multi-output model
        also able to monitor training progress as training continues(showing major metrics)
    5.predict: different to training from a large dataset, this model should be able to perform fast single step predict on-fly from a feature list generate for current time
        thus should also support pre-compile before single-step-inference
        no need for batch mode as this is real-time application, only single-step needed
        able to load/compile multiple trained models of different types to do predict on the same features list single-steppedly at the same time
    6.persist: save and load
    7.show comment to above configurable items to help user make decision of what to use
"""


"""
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| Model                           | Pros                                                        | Cons                                                        |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| LSTM (Long Short-Term Memory)   | - Handles long-range dependencies well.                     | - Computationally expensive for long sequences.             |
|                                 | - Mitigates vanishing gradient problem.                     | - Slower training due to more parameters.                   |
|                                 | - Suitable for complex sequences (e.g., time-series, text). |                                                             |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| GRU (Gated Recurrent Unit)      | - Simpler architecture (fewer parameters).                  | - Struggles with very long sequences compared to LSTM.      |
|                                 | - Faster training than LSTM, often similar performance.     | - Fewer gates may limit flexibility for complex tasks.      |
|                                 | - Effective at learning temporal dependencies.              |                                                             |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| RNN (Recurrent Neural Network)  | - Simple and easy to implement.                             | - Struggles with long-range dependencies(vanishing gradient)|
|                                 | - Good for short-term dependencies.                         | - Not suitable for complex tasks without enhancements.      |
|                                 | - Lower computational cost than LSTM/GRU.                   |                                                             |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| BiLSTM (Bidirectional LSTM)     | - Captures both past and future context.                    | - More computationally expensive than unidirectional LSTM.  |
|                                 | - Suitable for tasks where future context is important.     | - Increases train time due to processing in both directions.|
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| 1D CNN (1D Convolutional        | - Efficient for local patterns in sequential data.          | - Limited at capturing long-term deps comparing to RNNs.    |
| Neural Networks)                |                                                             |                                                             |
|                                 | - Faster than RNNs for certain cases.                       | - Requires tuning of kernel size for optimal performance.   |
|                                 | - Can handle high-dimensional inputs.                       |                                                             |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| Attention Mechanisms            | - Highly parallelizable, faster training.                   | - Requires large amounts of data for effective training.    |
| (e.g., Transformer)             |                                                             |                                                             |
|                                 | - Great at capturing long-range dependencies.               | - Computationally intensive with long sequences.            |
|                                 | - Scalable to large datasets and sequences.                 | - Complexity in hyperparameter tuning.                      |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+

+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Normalization Type     | Formula                                          | Best Used For                              | Characteristics                                  |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Z-Score Normalization  | (x - μ) / σ                                      | - Normally distributed data                | - Mean = 0                                       |
| (StandardScaler)       | Transforms to mean 0, std dev 1                  | - Features on similar scales               | - Std Dev = 1                                    |
|                        |                                                  | - When outliers are not extreme            | - Sensitive to outliers                          |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Min-Max Scaling        | (x - min(x)) / (max(x) - min(x))                 | - When you need bounded range              | - Preserves zero values                          |
| (MinMaxScaler)         | Scales to fixed range (default [0,1])            | - Neural network inputs                    | - Sensitive to outliers                          |
|                        |                                                  | - Image processing                         | - Does not handle new min/max in test data       |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Robust Scaling         | (x - median) / IQR                               | - Data with many outliers                  | - Uses median instead of mean                    |
| (RobustScaler)         | Uses median and interquartile range              | - Skewed distributions                     | - Less affected by extreme values                |
|                        |                                                  | - Financial data with extreme values       | - Preserves shape of distribution                |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| MaxAbs Scaling         | x / max(abs(x))                                  | - Sparse data                              | - Preserves sign of original values              |
| (MaxAbsScaler)         | Scales by maximum absolute value                 | - When zero is meaningful                  | - Does not center data                           |
|                        |                                                  | - Machine learning with sparse features    | - Bounded between -1 and 1                       |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Quantile Transformation| Transforms to uniform/normal distribution        | - Non-gaussian distributions               | - Makes data look like normal distribution       |
| (QuantileTransformer)  | Equalizes feature distributions                  | - When feature distributions differ        | - Can handle non-linear transformations          |
|                        |                                                  | - Machine learning with varied features    | - Destroys sparseness                            |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Power Transformation   | x^λ or log(x)                                    | - Right-skewed data                        | - Stabilizes variance                            |
| (PowerTransformer)     | Stabilizes variance and makes data more Gaussian | - Financial ratios                         | - Handles positive values                        |
|                        |                                                  | - Economic indicators                      | - Different methods like Yeo-Johnson, Box-Cox    |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
"""

"""
FEATURE PROPERTIES FOR ENSEMBLE VS NEURAL NETWORK METHODS
+-----------------------+--------------------------------+--------------------------------+
| Property              | Ensemble/Tree-Based Methods    | Neural Networks                |
+-----------------------+--------------------------------+--------------------------------+
| Value Range           | - Raw values acceptable        | - Scale to [-1, 1] or [0, 1]   |
|                       | - No scaling needed            | - Min-max or standard scaling  |
|                       | - Unbounded OK                 | - Remove values beyond ±3σ     |
+-----------------------+--------------------------------+--------------------------------+
| Distribution          | - Any distribution OK          | - Normal/Uniform preferred     |
|                       | - Multimodal OK                | - Transform skewed features    |
|                       | - No transformation needed     | - Log/Box-Cox for heavy tails  |
+-----------------------+--------------------------------+--------------------------------+
| Mean                  | - Any value OK                 | - ~0 after standardization     |
|                       | - No centering needed          | - Center before training       |
+-----------------------+--------------------------------+--------------------------------+
| Variance              | - Non-zero required            | - ~1 after standardization     |
|                       | - No upper bound               | - Similar scale across features|
|                       | - High variance OK             | - Bounded variance preferred   |
+-----------------------+--------------------------------+--------------------------------+
| Skewness              | - Any value acceptable         | - Between -1 and 1             |
|                       | - No transformation needed     | - Transform if outside range   |
|                       | - Handles asymmetry well       | - Symmetric preferred          |
+-----------------------+--------------------------------+--------------------------------+
| Kurtosis              | - Any value acceptable         | - Close to 3 (normal dist)     |
|                       | - Handles heavy tails well     | - Transform heavy tails        |
|                       | - No transformation needed     | - Avoid extreme peaks          |
+-----------------------+--------------------------------+--------------------------------+
| Missing Values        | - Up to 25-30% acceptable      | - Less than 5% preferred       |
|                       | - Handle natively              | - Requires imputation          |
|                       | - Can use as split point       | - Use mean/median/mode         |
+-----------------------+--------------------------------+--------------------------------+
| Outliers              | - Robust to outliers           | - Remove or cap at ±3σ         |
|                       | - Can keep as is               | - Use robust scaling           |
|                       | - May improve splits           | - Consider Winsorization       |
+-----------------------+--------------------------------+--------------------------------+
| Cardinality           | - High cardinality OK          | - Low cardinality preferred    |
| (Categorical)         | - Up to 50-100 levels OK       | - One-hot if < 15 levels       |
|                       | - Can handle rare cats         | - Embedding for high cardinal  |
+-----------------------+--------------------------------+--------------------------------+
| Interactions          | - Auto-detected                | - Need manual engineering      |
|                       | - No preprocessing needed      | - Cross-features helpful       |
|                       | - Handles non-linearity        | - Polynomial features help     |
+-----------------------+--------------------------------+--------------------------------+
| Feature Independence  | - Handles correlation well     | - Independent preferred        |
|                       | - No decorrelation needed      | - PCA/decorrelation helps      |
|                       | - Redundancy OK                | - Remove highly correlated     |
+-----------------------+--------------------------------+--------------------------------+
| Zero/Constant Values  | - Auto-filtered                | - Remove constant features     |
|                       | - Low variance OK              | - Remove near-zero variance    |
|                       | - Sparse data OK               | - Sparse inputs challenging    |
+-----------------------+--------------------------------+--------------------------------+
| Numeric Resolution    | - Any precision OK             | - 32-bit float typical         |
|                       | - Integer or float OK          | - Consistent precision needed  |
|                       | - Binary values OK             | - Normalize if large scale diff|
+-----------------------+--------------------------------+--------------------------------+
| Feature Scale         | - Any range acceptable         | - Similar scales crucial       |
| Relationships         | - No relative scaling needed   | - Normalize across features    |
|                       | - Preserves relationships      | - Balance feature importance   |
+-----------------------+--------------------------------+--------------------------------+
"""

"""
WEAKNESSES OF ENSEMBLE TREE-BASED METHODS
+---------------------------+------------------------------------------------+
| Category                  | Limitations & Issues                           |
+---------------------------+------------------------------------------------+
| Computational Resources   | - Large memory footprint for model storage     |
|                           | - Slow inference with deep trees               |
|                           | - Training time scales poorly (N*logN)         |
|                           | - High RAM usage during training               |
|                           | - Parallel training limited by CPU cores       |
+---------------------------+------------------------------------------------+
| Feature Space Handling    | - Inefficient with sparse high-dim features    |
|                           | - Cannot learn natural linear combinations     |
|                           | - Manual feature crossing often needed         |
|                           | - Many trees needed for simple linear patterns |
|                           | - Poor handling of cyclic/periodic features    |
+---------------------------+------------------------------------------------+
| Optimization & Training   | - Discrete optimization (no smooth gradients)  |
|                           | - No end-to-end training capability            |
|                           | - Hard to incorporate domain constraints       |
|                           | - Limited transfer learning ability            |
|                           | - Cannot freeze/fine-tune partial model        |
+---------------------------+------------------------------------------------+
| Generalization Issues     | - Cannot extrapolate beyond training range     |
|                           | - Poor performance in sparse regions           |
|                           | - Rectangular decision boundaries only         |
|                           | - Limited smooth boundary approximation        |
|                           | - Struggles with continuous symmetries         |
+---------------------------+------------------------------------------------+
| Statistical Limitations   | - No native probability calibration            |
|                           | - Uncertainty requires special methods         |
|                           | - Cannot incorporate Bayesian priors           |
|                           | - Biased importance for correlated features    |
|                           | - No natural confidence intervals              |
+---------------------------+------------------------------------------------+
| Performance Ceiling       | - Plateaus with increased data size            |
|                           | - Exponential data needs with dimensions       |
|                           | - Cannot learn certain function classes        |
|                           | - Limited representation power                 |
|                           | - Inefficient for some pattern types           |
+---------------------------+------------------------------------------------+
| Model Properties          | - Prone to overfitting with deep trees         |
|                           | - Hard to interpret individual predictions     |
|                           | - No natural online learning                   |
|                           | - Model size grows with complexity             |
|                           | - Limited model compression options            |
+---------------------------+------------------------------------------------+
| Deployment Challenges     | - Large deployment package size                |
|                           | - High latency for real-time inference         |
|                           | - Resource-intensive serving                   |
|                           | - Complex versioning with ensembles            |
|                           | - Difficult to deploy on edge devices          |
+---------------------------+------------------------------------------------+
"""
