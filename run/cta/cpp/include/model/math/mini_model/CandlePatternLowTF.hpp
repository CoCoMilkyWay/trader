#pragma once

#include <cmath>
#include <vector>

#include <model/define/DataType.hpp>
#include <model/define/CBuffer.hpp>

// in lower time-frame e.g. min, we only consider candle wick, the candle body doesn't carry much info

template <typename T1, typename T2, size_t N> // compiler auto derive
class CandlePattern
{
public:
    explicit CandlePattern(
        CBuffer<T1, N> *open,
        CBuffer<T1, N> *high,
        CBuffer<T1, N> *low,
        CBuffer<T1, N> *close,
        CBuffer<T2, N> *strength,
        CBuffer<T2, N> *timedelta
        )
        : open_(open),
          high_(high),
          low_(low),
          close_(close),
          strength_(strength),
          timedelta_(timedelta)
    {
    }

    inline void process()
    {
        // TODO: Implement candle pattern logic
    }

private:
    CBuffer<T1, N> *open_;
    CBuffer<T1, N> *high_;
    CBuffer<T1, N> *low_;
    CBuffer<T1, N> *close_;
    CBuffer<T2, N> *strength_;
    CBuffer<T2, N> *timedelta_;

    const int num_candle_bars = NUM_CANDLE_BARS;
    const int num_candle_strength = 9;

    // bar0, bar1, ... barN strength * return (barN+1 strength)
    std::vector<std::vector<int>> stat_history = std::vector<std::vector<int>>(std::pow(num_candle_strength, num_candle_bars), std::vector<int>(num_candle_strength, 0));
};
