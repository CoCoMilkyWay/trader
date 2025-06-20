#pragma once
#include <cstdint>

#pragma pack(push, 1)
struct TimeBar
{
    int64_t time; // e.g., 202406120930
    float open, high, low, close, volume;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct RunBar
{
    int64_t time;
    int timedelta;
    float open, high, low, close;
    float vwap;
    float threshold;
    float label_continuous;
    int label_discrete;
    float label_uniqueness;
    // UMAP embedding (3D)
    float umap_x, umap_y, umap_z;
};
#pragma pack(pop)

// Data Struct
inline constexpr int BLen = 100;

// Resample
inline constexpr int DATA_BASE_PERIOD = 1;                                       // base period of data (e.g. 1min)
inline constexpr int RESAMPLE_BASE_PERIOD = 4;                                   // avoid using 5/10/30 for liquidity reasons
inline constexpr std::array<int, 3> RESAMPLE_MULTITIMEFRAME_PERIODS = {5, 5, 5}; // periods used for multi-time-frame analysis

// Mini-Models
inline constexpr int NUM_CANDLE_BARS = 4;