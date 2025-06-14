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
    float open, high, low, close;
    float vwap;
    float threshold;
    float label_continuous;
    int label_discrete;
    float label_uniqueness;
};
#pragma pack(pop)
