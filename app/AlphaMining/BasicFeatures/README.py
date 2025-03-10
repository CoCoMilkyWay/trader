# This is basic timeseries features from tsfresh
# +--------------------------------+----------------------------------------+------------------------------------------------------------+
# | Category                       | Function                               | Description                                                |
# +--------------------------------+----------------------------------------+------------------------------------------------------------+
# | Basic Statistics               | mean(x)                                | Returns the mean of x.                                     |
# |                                | median(x)                              | Returns the median of x.                                   |
# |                                | minimum(x)                             | Calculates the lowest value of the time series.            |
# |                                | maximum(x)                             | Calculates the highest value of the time series.           |
# |                                | standard_deviation(x)                  | Returns the standard deviation of x.                       |
# |                                | variance(x)                            | Returns the variance of x.                                 |
# |                                | variation_coefficient(x)               | Returns the variation coefficient (standard error / mean). |
# |                                | kurtosis(x)                            | Returns the kurtosis of x.                                 |
# |                                | skewness(x)                            | Returns the skewness of x.                                 |
# |                                | sum_values(x)                          | Calculates the sum over the time series values.            |
# |                                | length(x)                              | Returns the length of x.                                   |
# +--------------------------------+----------------------------------------+------------------------------------------------------------+
# | Energy & Complexity            | abs_energy(x)                          | Returns the absolute energy (sum of squared values).       |
# |                                | root_mean_square(x)                    | Returns the root mean square (RMS) of x.                   |
# |                                | cid_ce(x, normalize)                   | Estimates time series complexity.                          |
# |                                | lempel_ziv_complexity(x, bins)         | Estimates complexity based on Lempel-Ziv compression.      |
# |                                | sample_entropy(x)                      | Computes sample entropy of x.                              |
# |                                | approximate_entropy(x, m, r)           | Implements Approximate Entropy algorithm.                  |
# |                                | permutation_entropy(x, tau, dimension) | Computes permutation entropy.                              |
# |                                | fourier_entropy(x, bins)               | Computes the binned entropy of the power spectral density. |
# |                                | binned_entropy(x, max_bins)            | Bins values into max_bins and calculates entropy.          |
# +--------------------------------+----------------------------------------+------------------------------------------------------------+
# | Trend & Change Detection       | absolute_sum_of_changes(x)             | Sum of absolute differences between consecutive values.    |
# |                                | mean_abs_change(x)                     | Computes average over first differences.                   |
# |                                | mean_change(x)                         | Computes average over time series differences.             |
# |                                | mean_second_derivative_central(x)      | Computes mean value of the second derivative.              |
# |                                | linear_trend(x, param)                 | Fits a linear least-squares regression.                    |
# |                                | linear_trend_timewise(x, param)        | Similar to linear_trend(x, param).                         |
# |                                | agg_linear_trend(x, param)             | Aggregated linear regression over chunks.                  |
# +--------------------------------+----------------------------------------+------------------------------------------------------------+
# | Peaks & Extreme Values         | absolute_maximum(x)                    | Returns the highest absolute value of x.                   |
# |                                | first_location_of_maximum(x)           | Returns the first index of the max value.                  |
# |                                | last_location_of_maximum(x)            | Returns the last index of the max value.                   |
# |                                | first_location_of_minimum(x)           | Returns the first index of the min value.                  |
# |                                | last_location_of_minimum(x)            | Returns the last index of the min value.                   |
# |                                | number_peaks(x, n)                     | Counts peaks of at least support n.                        |
# |                                | number_cwt_peaks(x, n)                 | Number of peaks found using continuous wavelet transform.  |
# |                                | c3(x, lag)                             | Measures non-linearity in time series.                     |
# +--------------------------------+----------------------------------------+------------------------------------------------------------+
# | Autocorrelation & Frequency Analysis | autocorrelation(x, lag)          | Computes autocorrelation for a given lag.                  |
# |                                | partial_autocorrelation(x, param)      | Computes partial autocorrelation at a given lag.           |
# |                                | agg_autocorrelation(x, param)          | Descriptive statistics of autocorrelation.                 |
# |                                | fft_coefficient(x, param)              | Computes Fourier coefficients.                             |
# |                                | fft_aggregated(x, param)               | Returns spectral centroid, variance, skew, kurtosis of the Fourier spectrum. |
# |                                | spkt_welch_density(x, param)           | Estimates cross power spectral density.                    |
# |                                | time_reversal_asymmetry_statistic(x, lag) | Computes time reversal asymmetry.                       |
# +--------------------------------+----------------------------------------+------------------------------------------------------------+
# | Threshold-based Features       | count_above(x, t)                      | Percentage of values above threshold t.                    |
# |                                | count_below(x, t)                      | Percentage of values below threshold t.                    |
# |                                | count_above_mean(x)                    | Counts values above mean.                                  |
# |                                | count_below_mean(x)                    | Counts values below mean.                                  |
# |                                | range_count(x, min, max)               | Counts values within the interval [min, max).              |
# |                                | ratio_beyond_r_sigma(x, r)             | Ratio of values more than r times std(x) from mean.        |
# |                                | large_standard_deviation(x, r)         | Checks if standard deviation is large.                     |
# +--------------------------------+----------------------------------------+------------------------------------------------------------+
# | Recurrence & Repetition        | has_duplicate(x)                       | Checks if any value appears more than once.                |
# |                                | has_duplicate_max(x)                   | Checks if the maximum value appears more than once.        |
# |                                | has_duplicate_min(x)                   | Checks if the minimum value appears more than once.        |
# |                                | percentage_of_reoccurring_values_to_all_values(x) | Percentage of recurring values in x.            |
# |                                | percentage_of_reoccurring_datapoints_to_all_datapoints(x) | Percentage of non-unique values.        |
# |                                | sum_of_reoccurring_values(x)           | Sum of all reoccurring values.                             |
# |                                | sum_of_reoccurring_data_points(x)      | Sum of all data points that appear more than once.         |
# |                                | ratio_value_number_to_time_series_length(x) | 1 if all values are unique, otherwise below 1.        |
# +--------------------------------+----------------------------------------+------------------------------------------------------------+
# | Anomaly Detection & Unit Root Tests | augmented_dickey_fuller(x, param) | Checks for unit root in the time series.                   |
# |                                | benford_correlation(x)                 | Checks correlation with Benford's law for anomalies.       |
# |                                | symmetry_looking(x, param)             | Checks if distribution looks symmetric.                    |
# +--------------------------------+----------------------------------------+------------------------------------------------------------+
# | Feature Engineering & Miscellaneous | ar_coefficient(x, param)          | Fits an autoregressive AR(k) model.                        |
# |                                | friedrich_coefficients(x, param)       | Computes polynomial coefficients for Langevin model.       |
# |                                | cwt_coefficients(x, param)             | Computes Continuous Wavelet Transform coefficients.        |
# |                                | index_mass_quantile(x, param)          | Computes the relative index where q% of mass lies.         |
# |                                | matrix_profile(x, param)               | Computes 1-D Matrix Profile and returns statistics.        |
# |                                | energy_ratio_by_chunks(x, param)       | Computes sum of squares per chunk.                         |
# |                                | query_similarity_count(x, param)       | Counts occurrences of a given subsequence.                 |
# |                                | value_count(x, value)                  | Counts occurrences of a specific value.                    |
# |                                | number_crossing_m(x, m)                | Counts the number of crossings of x at value m.            |
# |                                | longest_strike_above_mean(x)           | Length of the longest streak above the mean.               |
# |                                | longest_strike_below_mean(x)           | Length of the longest streak below the mean.               |
# |                                | max_langevin_fixed_point(x, r, m)      | Computes the largest fixed point of Langevin dynamics.     |
# |                                | quantile(x, q)                         | Computes the q-th quantile of x.                           |
# |                                | variance_larger_than_standard_deviation(x) | Checks if variance > standard deviation.               |
# +--------------------------------+----------------------------------------+------------------------------------------------------------+
# 