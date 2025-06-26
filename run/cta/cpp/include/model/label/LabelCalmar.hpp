#ifndef LABEL_CALMAR_HPP
#define LABEL_CALMAR_HPP

// System headers
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

// Project headers
#include <model/define/DataType.hpp>
#include <model/misc/misc.hpp>

class LabelCalmar
{
private:
    // Configuration constants
    static constexpr int H_PER_D = 23;                          // trade hours per day
    static constexpr int P_PER_B = 5;                           // equivalent bar period
    static constexpr int EMA_VOL_SPAN = 60 / P_PER_B * H_PER_D; // span for EMA volatility (daily)
    static constexpr float CUSUM_FACTOR = 0.1f;                 // multiplier for CUSUM threshold
    static constexpr int ATR_WINDOW = 60 / P_PER_B * H_PER_D;   // bars for ATR
    static constexpr size_t MAX_BARS = 60 / P_PER_B * 4;        // maximum lifetime of event
    static constexpr float ALPHA = 2.0f / (EMA_VOL_SPAN + 1);

    const float exchange_fee = 0.25f;     // per round turn
    const float clearing_fee = 0.10f;     // per round turn
    const float nfa_fee = 0.02f;          // per round turn
    const float broker_commission = 1.0f; // per round turn
    const float total_fees = exchange_fee + clearing_fee + nfa_fee + broker_commission;
    const float taker_slippage = 0.25f * 4.0f;
    const float index_value = 25000.0f;              // hypothetical index level
    const float notional_value = 2.0f * index_value; // MNQ multiplier is $2
    const float fee = (total_fees + taker_slippage) / notional_value;

    // State
    float prev_close = 0.0f;
    float ema_mean_pos = 0.0f, ema_var_pos = 0.0f;
    float ema_mean_neg = 0.0f, ema_var_neg = 0.0f;
    std::vector<size_t> best_close_indices;

    struct DirectionResult
    {
        float max_rr; // Maximum Return ratio
        float max_dr; // Maximum drawdown/drawup
        size_t best_idx;
        float calmar;
    };

    DirectionResult calculate_direction(
        int direction, size_t i,
        const std::vector<RunBar> &bars,
        float entry_price,
        size_t max_end)
    {
        DirectionResult result = {};

        // Find exit point (price crosses stop line)
        float stop_line = (direction == 1) ? bars[i].low : bars[i].high;
        size_t end_idx = max_end;
        for (size_t k = i + 1; k <= max_end; ++k)
        {
            if (direction * bars[k].close <= direction * stop_line)
            {
                end_idx = k;
                break;
            }
        }

        // Find best price in the run
        auto it = std::max_element(
            bars.begin() + i,
            bars.begin() + end_idx + 1,
            [direction](auto const &a, auto const &b)
            {
                return direction * a.close < direction * b.close;
            });
        result.best_idx = std::distance(bars.begin(), it);

        // Calculate return ratio
        result.max_rr = (it->close / entry_price) - 1.0f - fee;
        // result.max_rr = std::clamp(result.max_rr, -0.01f, 0.01f);  // cap between ±1%

        // Calculate max drawdown/drawup with early termination
        float running_extreme_l = entry_price + 1e-3f;
        float running_extreme_s = entry_price - 1e-3f;
        result.max_dr = 0.0f;

        for (size_t k = i; k <= result.best_idx && result.max_dr < 1.0f; ++k)
        {
            float price = bars[k].close;
            if (direction == 1)
            {
                running_extreme_l = std::max(running_extreme_l, price);
                float run_up = running_extreme_l - entry_price;
                float dr = (running_extreme_l - price) / run_up;
                result.max_dr = std::max(result.max_dr, dr);
            }
            else
            {
                running_extreme_s = std::min(running_extreme_s, price);
                float run_down = entry_price - running_extreme_s;
                float dr = (price - running_extreme_s) / run_down;
                result.max_dr = std::max(result.max_dr, dr);
            }
        }

        // Final calculations
        result.max_dr = std::min(result.max_dr, 1.0f);
        float raw_calmar = result.max_rr * (1.5f - result.max_dr); // tolerate more drawdowns
        result.calmar = (direction == 1) ? raw_calmar : -raw_calmar;

        return result;
    }

public:
    LabelCalmar() {}

    void process(std::vector<RunBar> &bars)
    {
        for (size_t i = 0; i < bars.size(); ++i)
        {
            process_bar(i, bars);
            misc::print_progress(i, bars.size());
        }
        calculate_uniqueness(bars);
        winsorize_labels(bars);
        discretize_field(bars);
    }

    bool process_bar(size_t i, std::vector<RunBar> &bars)
    {
        RunBar &bar = bars[i];
        if (i == 0) [[unlikely]]
        {
            best_close_indices.assign(bars.size(), 0);
            prev_close = bar.close;
            ema_mean_pos = ema_var_pos = ema_mean_neg = ema_var_neg = 0.0001f;
            bar.label_continuous = 0.0f;
            bar.label_uniqueness = 1.0f;
            return true;
        }

        float return_val = (bar.close - prev_close) / std::max(prev_close, 1e-6f);
        float pos_return = return_val > 0 ? return_val : 0.0f;
        float neg_return = return_val < 0 ? return_val : 0.0f;

        ema_mean_pos = ALPHA * pos_return + (1 - ALPHA) * ema_mean_pos;
        ema_var_pos = ALPHA * (pos_return - ema_mean_pos) * (pos_return - ema_mean_pos) + (1 - ALPHA) * ema_var_pos;

        ema_mean_neg = ALPHA * neg_return + (1 - ALPHA) * ema_mean_neg;
        ema_var_neg = ALPHA * (neg_return - ema_mean_neg) * (neg_return - ema_mean_neg) + (1 - ALPHA) * ema_var_neg;

        float pos_vol = std::sqrt(ema_var_pos);
        float neg_vol = std::sqrt(ema_var_neg);
        float vol = return_val > 0 ? pos_vol : neg_vol;
        vol = std::max(vol, 1e-6f);

        size_t n = bars.size();
        size_t max_end = std::min(i + MAX_BARS, n - 1);

        float entry_price = (bar.high + bar.low) * 0.5f;

        // we want label to be somewhat Gaussian
        auto long_res = calculate_direction(1, i, bars, entry_price, max_end);
        auto short_res = calculate_direction(-1, i, bars, entry_price, max_end);

        if (long_res.calmar > short_res.calmar)
        {
            bar.label_continuous = long_res.calmar;
            best_close_indices[i] = long_res.best_idx;
        }
        else
        {
            bar.label_continuous = -short_res.calmar;
            best_close_indices[i] = short_res.best_idx;
        }

        prev_close = bar.close;

        // if (i % int(bars.size() * 0.05f) == 0 || i == bars.size() - 1) [[unlikely]]
        // {
        //     misc::print_progress(i + 1, bars.size());
        // }

        return true;
    }

    void calculate_uniqueness(std::vector<RunBar> &bars)
    {
        size_t n = bars.size();
        std::vector<int> counts(n, 0);
        for (size_t i = 0; i < n; ++i)
        {
            size_t j = best_close_indices[i];
            for (size_t k = i; k <= j && k < n; ++k)
                counts[k]++;
        }
        for (size_t k = 0; k < n; ++k)
            bars[k].label_uniqueness = (counts[k] > 0) ? 1.0f / counts[k] : 0.0f;
    }

    float interpolate_percentile(std::vector<float> &data, float p)
    {
        assert(!data.empty());
        std::vector<float> sorted = data; // Make a copy for sorting
        std::sort(sorted.begin(), sorted.end());

        float idx = p * (sorted.size() - 1);
        int idx_below = static_cast<int>(std::floor(idx));
        int idx_above = static_cast<int>(std::ceil(idx));
        float weight = idx - idx_below;

        return sorted[idx_below] * (1.0f - weight) + sorted[idx_above] * weight;
    }

    void discretize_field(std::vector<RunBar> &bars)
    {
        if (bars.empty())
            return;

        // Step 1: extract the continuous values
        std::vector<float> values;
        values.reserve(bars.size());
        for (const auto &bar : bars)
            values.push_back(bar.label_continuous);

        // Step 2: copy and sort the values
        std::vector<float> sorted = values;
        std::sort(sorted.begin(), sorted.end());

        auto interpolate_percentile = [&](float p) -> float
        {
            float idx = p * (sorted.size() - 1);
            int idx_below = static_cast<int>(std::floor(idx));
            int idx_above = static_cast<int>(std::ceil(idx));
            float weight = idx - idx_below;
            return sorted[idx_below] * (1.0f - weight) + sorted[idx_above] * weight;
        };

        // Step 3: compute percentile thresholds
        std::vector<float> thresholds = {
            interpolate_percentile(0.15f),
            interpolate_percentile(0.35f),
            interpolate_percentile(0.50f),
            interpolate_percentile(0.65f),
            interpolate_percentile(0.85f)};

        // Step 4: assign bin index to each bar
        for (auto &bar : bars)
        {
            float v = bar.label_continuous;
            int label;
            if (v < thresholds[0])
                label = -3;
            else if (v < thresholds[1])
                label = -2;
            else if (v < thresholds[2])
                label = -1;
            else if (v < thresholds[3])
                label = 1;
            else if (v < thresholds[4])
                label = 2;
            else
                label = 3;
            bar.label_discrete = label;
        }
    }

    void winsorize_labels(std::vector<RunBar> &bars)
    {
        std::vector<float> labels;
        labels.reserve(bars.size());
        for (auto &b : bars)
            labels.push_back(b.label_continuous);

        // Step 1: Compute percentiles
        float lo = interpolate_percentile(labels, 0.05f);
        float hi = interpolate_percentile(labels, 0.95f);
        float range = hi - lo;
        float clip_lo = lo - 0.1f * range;
        float clip_hi = hi + 0.1f * range;

        // Step 2: Clip labels (winsorize soft)
        std::vector<float> clipped;
        clipped.reserve(labels.size());
        for (float v : labels)
        {
            float c = std::clamp(v, clip_lo, clip_hi);
            clipped.push_back(c);
        }

        // Step 3: Compute std of clipped values
        float mean = std::accumulate(clipped.begin(), clipped.end(), 0.0f) / clipped.size();
        float sq_sum = 0.0f;
        for (float v : clipped)
            sq_sum += (v - mean) * (v - mean);
        float std_dev = std::sqrt(sq_sum / clipped.size());
        float scale = std_dev * 1.5f;

        // Step 4: Apply tanh normalization
        for (auto &b : bars)
            b.label_continuous = std::tanh(b.label_continuous / scale);
    }
};

#endif // LABEL_CALMAR_HPP