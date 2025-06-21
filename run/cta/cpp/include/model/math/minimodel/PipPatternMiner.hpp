#ifndef PIP_PATTERN_HPP
#define PIP_PATTERN_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

#include <model/define/DataType.hpp>
#include <model/define/CBuffer.hpp>
#include <model/misc/print.hpp>

/**
 * Perceptually Important Point (PIP) Pattern Miner
 *
 * Identifies PIP patterns in regularized price data
 * Preprocessing using VAE, clustering in latent space using GMM
 * Evaluates performance using Martin ratio
 */
template <typename T1, size_t N> // compiler auto derive
class PipPatternMiner
{
public:
    // Constructor
    explicit PipPatternMiner(
        CBuffer<T1, N> *high,
        CBuffer<T1, N> *low)
        : high_(high),
          low_(low) {}

    inline void process()
    {
        price_buffer.push_back(high_->back() + low_->back());
        ++sample_count;

        if (!is_inited) [[unlikely]]
        {
            if (price_buffer.full())
            {
                is_inited = 1;
            }
            return;
        }

        // improve sample efficiency
        if (sample_count % NUM_BARS_PER_PIP != 0)
        {
            return;
        }

        std::array<float, NUM_LOOKBACK> window_backward = price_buffer.to_array<NUM_LOOKBACK>(0);
        std::array<float, NUM_HOLD> window_forward = price_buffer.to_array<NUM_HOLD>(NUM_LOOKBACK);
        std::array<int, NUM_PIPS> pips_x;
        std::array<float, NUM_PIPS> pips_y;
        float label;
        findPips(window_backward, pips_x, pips_y);

        // println(window_backward);
        // println(pips_x);
        // println(pips_y);

        // check duplicates
        bool same = true;
        for (int j = 1; j < NUM_PIPS - 1; ++j)
        {
            if (pips_x[j] != last_pips_x[j]) [[likely]]
            {
                same = false;
                break;
            }
        }
        if (!same) [[likely]]
        {
            label = findMartin(window_forward);
            buffer_pip_indices_.push_back(pips_x);
            buffer_pip_patterns_.push_back(pips_y);
            buffer_pip_labels_.push_back(label);
        }
        last_pips_x = pips_x;
    }

    //// Train on historical price data (log prices)
    // void train(const std::vector<float> &data)
    //{
    //     // store data up to limit
    //     size_t limit = std::min<size_t>(data.size(), 20000);
    //     data_.assign(data.begin(), data.begin() + limit);
    //     test_data_.assign(data_.begin() + std::min<size_t>(data.size(), 10000),
    //                       data.begin() + limit);
    //     findUniquePatterns();
    //     // TODO: integrate clustering and performance
    //     getClusterPerformance();
    // }
    //
    //// Predict signals for test set, returns vector of signals (+1 long, -1 short, 0 neutral)
    // std::vector<int> predict() const
    //{
    //     std::vector<int> signals(test_data_.size(), 0);
    //     int hold = hold_period_ / 2;
    //     for (int i = lookback_ - 1; i + hold < static_cast<int>(test_data_.size()); i += 4)
    //     {
    //         // sliding window
    //         std::vector<float> window(test_data_.begin() + (i - lookback_ + 1), test_data_.begin() + i + 1);
    //         auto [px, py] = findPips(window);
    //         // normalize
    //         float mean = std::accumulate(py.begin(), py.end(), 0.0) / py.size();
    //         float var = 0.0;
    //         for (float v : py)
    //             var += (v - mean) * (v - mean);
    //         var /= py.size();
    //         float std = std::sqrt(var) + 1e-8;
    //         for (float &v : py)
    //             v = (v - mean) / std;
    //         // find closest cluster (stub: no clusters, default neutral)
    //         int signal = 0;
    //         // assign signals
    //         for (int k = 0; k < hold; ++k)
    //         {
    //             signals[i + k] = signal;
    //         }
    //     }
    //     return signals;
    // }

private:
    CBuffer<T1, N> *high_;
    CBuffer<T1, N> *low_;
    CBuffer<float, NUM_LOOKBACK + NUM_HOLD> price_buffer;

    bool is_inited = 0;
    bool is_trained = 0;
    int sample_count = 0;

    // improve sample efficiency
    std::array<int, NUM_PIPS> last_pips_x;

    // Replay Buffer
    CBuffer<std::array<float, NUM_PIPS>, NUM_TRAINING_SAMPLES> buffer_pip_patterns_; // X1~X4
    CBuffer<std::array<int, NUM_PIPS>, NUM_TRAINING_SAMPLES> buffer_pip_indices_;    // X5~X8
    CBuffer<float, NUM_TRAINING_SAMPLES> buffer_pip_labels_;                         // y

    // // Clusters
    // const int num_clusters = 40;
    // CBuffer<std::array<float, NUM_PIPS>, num_clusters> clusters_centers;
    // CBuffer<std::array<float, NUM_PIPS>, num_clusters> clusters_centers;
    // CBuffer<std::array<float, NUM_PIPS>, num_clusters> clusters_centers;

    // Utility: find perceptually important points in window
    void findPips(const std::array<float, NUM_LOOKBACK> &data,
                  std::array<int, NUM_PIPS> &pips_x,
                  std::array<float, NUM_PIPS> &pips_y)
    {
        pips_x[0] = 0;
        pips_x[1] = static_cast<int>(data.size() - 1);
        pips_y[0] = data.front();
        pips_y[1] = data.back();
        int curr_size = 2;

        // Iteratively add one PIP until we have NUM_PIPS
        for (int curr = 2; curr < NUM_PIPS; ++curr)
        {
            float md = 0.0f;     // Maximum distance found so far
            int md_i = -1;       // Index of point with max distance
            int insert_pos = -1; // Where to insert this point in the list
            // Go through each segment defined by two existing PIPs
            for (int k = 0; k < curr_size - 1; ++k)
            {
                // Compute line parameters between two PIPs
                int left = pips_x[k], right = pips_x[k + 1];
                float dy = pips_y[k + 1] - pips_y[k];
                float dx = static_cast<float>(right - left);
                float slope = dy / dx;
                float intercept = pips_y[k] - slope * left;
                // Evaluate points between left and right
                for (int i = left + 1; i < right; ++i)
                {
                    float d = 0.0f;
                    float val = data[i];
                    // 1: Euclidean, 2: Perpendicular, 3: Vertical
                    if (PIP_DIST_MEASURE == 1)
                    {
                        float d1 = std::hypot(i - left, pips_y[k] - val);
                        float d2 = std::hypot(i - right, pips_y[k + 1] - val);
                        d = d1 + d2;
                    }
                    else if (PIP_DIST_MEASURE == 2)
                    {
                        d = std::abs(slope * i + intercept - val) / std::hypot(slope, 1.0f);
                    }
                    else
                    {
                        d = std::abs(slope * i + intercept - val);
                    }
                    if (d > md) // Update max distance if this point is farther
                    {
                        md = d;
                        md_i = i;
                        insert_pos = k + 1;
                    }
                }
            }
            // Insert the point with the maximum distance by shifting elements
            std::copy_backward(pips_x.begin() + insert_pos, pips_x.begin() + curr_size, pips_x.begin() + curr_size + 1);
            pips_x[insert_pos] = md_i;
            std::copy_backward(pips_y.begin() + insert_pos, pips_y.begin() + curr_size, pips_y.begin() + curr_size + 1);
            pips_y[insert_pos] = data[md_i];
            curr_size++;
        }
        normalize_array(pips_y);
    }

    // Utility: compute Martin ratio on future window (Martin Ratio = Total Log Return / Ulcer Index)
    float findMartin(const std::array<float, NUM_HOLD> &price)
    {
        static_assert(NUM_HOLD >= 2, "Need at least 2 prices to compute returns");

        std::array<float, NUM_HOLD - 1> rets{};
        float total = 0.0f;
        float prev_log = std::log(price[0]);

        // Compute log returns and total log return
        for (std::size_t i = 1; i < NUM_HOLD; ++i)
        {
            float curr_log = std::log(price[i]);
            float r = curr_log - prev_log;
            rets[i - 1] = r;
            total += r;
            prev_log = curr_log;
        }

        // Treat as short if total return is negative
        bool short_pos = false;
        if (total < 0.0f)
        {
            for (float &r : rets)
                r = -r;
            total = -total;
            short_pos = true;
        }

        // Compute equity curve, drawdowns, and ulcer index
        float csum = 0.0f;
        float running_max = 0.0f;
        float drawdown_sum_sq = 0.0f;
        for (float r : rets)
        {
            csum += r;
            float eq = std::exp(csum); // cumulative equity
            running_max = std::max(running_max, eq);
            float dd = (eq / running_max) - 1.0f;
            drawdown_sum_sq += dd * dd;
        }

        float ulcer = std::sqrt(drawdown_sum_sq / static_cast<float>(rets.size()));
        float min_ui = std::max(1e-4f, total * 0.05f); // stabilize denominator
        ulcer = std::max(ulcer, min_ui);

        float martin = total / ulcer;
        return short_pos ? -martin : martin;
    }

    template <std::size_t M>
    inline void normalize_array(std::array<float, M> &arr)
    {
        // Take log of each element
        for (float &v : arr)
        {
            v = std::log(v + 1e-8f); // avoid log(0) or negative input
        }

        float mean = 0.0f;
        float m2 = 0.0f; // sum of squares of differences from the mean

        // One-pass numerically stable mean & variance (Welford’s algorithm)
        for (std::size_t i = 0; i < M; ++i)
        {
            float delta = arr[i] - mean;
            mean += delta / (i + 1);
            m2 += delta * (arr[i] - mean); // uses updated mean
        }

        float stddev = std::sqrt(m2 / static_cast<float>(M));
        float denom = stddev + 1e-8f;

        // Normalize in-place
        for (float &v : arr)
        {
            v = (v - mean) / denom;
        }
    }
};

#endif // PIP_PATTERN_HPP
