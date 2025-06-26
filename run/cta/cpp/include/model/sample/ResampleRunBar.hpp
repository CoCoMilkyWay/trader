#pragma once

// System headers
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

// Project headers
#include <model/define/CBuffer.hpp>
#include <model/define/DataType.hpp>

template <typename T1, typename T2, typename T3, size_t N> // compiler auto derive
class ResampleRunBar
{
public:
    explicit ResampleRunBar(
        CBuffer<T1, N> *time,
        CBuffer<T2, N> *timedelta,
        CBuffer<T3, N> *open,
        CBuffer<T3, N> *high,
        CBuffer<T3, N> *low,
        CBuffer<T3, N> *close,
        CBuffer<T3, N> *vwap)
        : time_(time),
          timedelta_(timedelta),
          open_(open),
          high_(high),
          low_(low),
          close_(close),
          vwap_(vwap)
    {
        daily_bars_label.reserve(60 * 24);
        daily_bars_volume.reserve(60 * 24);
    }

    inline bool process(const TimeBar &bar)
    {
        auto [date, time] = extract_datetime(bar.time);

        // ---- Label Calculation ----
        if (prev_close > 0.0f) [[likely]]
        {
            label = bar.close > prev_close;
            buy_vol = static_cast<float>(label) * bar.volume;
            sell_vol = static_cast<float>(!label) * bar.volume;
        }
        prev_close = bar.close;

        // ---- New Day Handling ----
        if (date != prev_date) [[unlikely]]
        {
            if (!daily_bars_label.empty()) [[likely]]
            {
                daily_thresh = find_run_threshold();
                ema_thresh = (ema_thresh < 0.0f) ? daily_thresh : alpha * daily_thresh + (1 - alpha) * ema_thresh;
            }
            daily_bars_label.clear();
            daily_bars_volume.clear();
            prev_date = date;
        }

        // ---- Update OHLC ----
        ohlc_open = (ohlc_open == 0.0f) ? bar.open : ohlc_open;
        ohlc_high = std::max(ohlc_high, bar.high);
        ohlc_low = std::min(ohlc_low, bar.low);
        daily_bars_label.push_back(label);
        daily_bars_volume.push_back(std::abs(bar.volume));
        ++bar_count;

        // ---- Accumulate Volumes ----
        cumm_buy += buy_vol;
        cumm_sell += sell_vol;
        cumm_vol += bar.volume;
        cumm_dollar += bar.volume * (bar.high + bar.low) / 2.0f;

        // ---- Check Bar Formation ----
        float theta = std::max(cumm_buy, cumm_sell);
        float threshold = (__builtin_expect(ema_thresh < 0.0f, 0)) ? 0.0f : ema_thresh;

        if (theta >= threshold) [[unlikely]]
        {
            time_->push_back(bar.time);
            timedelta_->push_back(bar_count);
            open_->push_back(ohlc_open);
            high_->push_back(ohlc_high);
            low_->push_back(ohlc_low);
            close_->push_back(bar.close);
            vwap_->push_back(__builtin_expect(cumm_vol > 0, 1) ? cumm_dollar / cumm_vol : bar.close);

            // Reset state
            ohlc_open = bar.close;
            ohlc_high = -std::numeric_limits<float>::infinity();
            ohlc_low = std::numeric_limits<float>::infinity();
            cumm_buy = cumm_sell = cumm_vol = cumm_dollar = 0.0f;
            bar_count = 0;
            return true;
        }

        return false;
    }

private:
    int p_ori = DATA_BASE_PERIOD;     // original sampling period (min)
    int p_tar = RESAMPLE_BASE_PERIOD; // target bar length (min)

    CBuffer<T1, N> *time_;
    CBuffer<T2, N> *timedelta_;
    CBuffer<T3, N> *open_;
    CBuffer<T3, N> *high_;
    CBuffer<T3, N> *low_;
    CBuffer<T3, N> *close_;
    CBuffer<T3, N> *vwap_;

    bool label;
    float buy_vol = 0.0f;
    float sell_vol = 0.0f;

    const float ema_days = 5 * 4;
    const float alpha = 2.0f / (ema_days + 1);
    float ema_thresh = -1.0f;

    float daily_thresh = 0.0f;
    float prev_close = -1.0f;
    int prev_date = -1;
    int bar_count = 0;

    float cumm_buy = 0.0f;
    float cumm_sell = 0.0f;
    float cumm_vol = 0.0f;
    float cumm_dollar = 0.0f;

    float ohlc_open = 0.0f;
    float ohlc_high = -std::numeric_limits<float>::infinity();
    float ohlc_low = std::numeric_limits<float>::infinity();
    float ohlc_close = 0.0f;

    std::vector<bool> daily_bars_label;
    std::vector<float> daily_bars_volume;
    int expected_bars = 0;

    static inline std::tuple<int, int> extract_datetime(int64_t timestamp)
    {
        int date = static_cast<int>(timestamp / 10000); // YYYYMMDD
        int time = static_cast<int>(timestamp % 10000); // HHMMSS
        return {date, time};
    }

    inline int compute_bar_count(float x)
    {
        float acc_pos = 0.0f;
        float acc_neg = 0.0f;
        int bar_count = 0;
        size_t n = daily_bars_volume.size();

        for (size_t i = 0; i < n; ++i)
        {
            if (daily_bars_label[i])
                acc_pos += daily_bars_volume[i];
            else
                acc_neg += daily_bars_volume[i];

            if (acc_pos >= x || acc_neg >= x)
            {
                ++bar_count;
                acc_pos = 0.0f;
                acc_neg = 0.0f;
            }
        }
        return bar_count;
    }

    inline float find_run_threshold()
    {
        if (daily_bars_label.empty())
            return 0.0f;

        expected_bars = int(daily_bars_label.size() * p_ori / p_tar);

        float x_max = std::accumulate(daily_bars_volume.begin(), daily_bars_volume.end(), 0.0f);
        float x_min = *std::min_element(daily_bars_volume.begin(), daily_bars_volume.end());

        int max_iter = 50;
        float x_mid = 0.0f;
        for (int i = 0; i < max_iter; ++i)
        {
            x_mid = 0.5f * (x_min + x_max);
            int bars = compute_bar_count(x_mid);

            if (std::abs(bars - expected_bars) <= 5 || (x_max - x_min) < 1e-6f)
                return x_mid;

            if (bars > expected_bars)
                x_min = x_mid;
            else
                x_max = x_mid;
        }

        return 0.5f * (x_min + x_max);
    }
};