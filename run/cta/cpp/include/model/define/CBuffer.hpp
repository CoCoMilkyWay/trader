#pragma once
#include <array>
#include <span>
#include <cstddef>
#include <stdexcept>

// CBuffer<double, 100> close_prices;
// close_prices.push_back(100.5);
// close_prices.push_back(101.2);
//
// // Access full buffer
// auto full_view = close_prices.span();
// double sum = 0;
// for (double v : full_view.head) sum += v;
// for (double v : full_view.tail) sum += v;
//
// // Access last 20 elements
// auto last20 = close_prices.last(20);
// if (last20.head.size() + last20.tail.size() >= 20) {
//     // Process moving average
// }
//
// // Access slice [10:59]
// auto slice = close_prices.subspan(10, 50);

template <typename T, size_t N>
class CBuffer // CircularBuffer
{
    static_assert(N > 0, "Capacity must be positive");

private:
    std::array<T, N> data_;
    size_t start_ = 0;
    size_t size_ = 0;

public:
    struct SplitSpan
    {
        std::span<const T> head;
        std::span<const T> tail;

        size_t size() const noexcept
        {
            return head.size() + tail.size();
        }
    };

    void push_back(const T &val)
    {
        if (size_ < N) [[unlikely]]
        {
            data_[(start_ + size_) % N] = val;
            ++size_;
        }
        else
        {
            data_[start_] = val;
            start_ = (start_ + 1) % N;
        }
    }

    void push_back(T &&val)
    {
        if (size_ < N) [[unlikely]]
        {
            data_[(start_ + size_) % N] = std::move(val);
            ++size_;
        }
        else
        {
            data_[start_] = std::move(val);
            start_ = (start_ + 1) % N;
        }
    }

    const T &front() const
    {
        if (size_ == 0) [[unlikely]]
        {
            throw std::out_of_range("Buffer is empty");
        }
        return data_[start_];
    }

    const T &back() const
    {
        if (size_ == 0) [[unlikely]]
        {
            throw std::out_of_range("Buffer is empty");
        }
        size_t last_index = (start_ + size_ - 1) % N;
        return data_[last_index];
    }

    SplitSpan span() const noexcept
    {
        return subspan(0, size_);
    }

    SplitSpan last(size_t count) const
    {
        if (count > size_) [[unlikely]]
        {
            throw std::out_of_range("Requested count exceeds buffer size");
        }
        return subspan(size_ - count, count);
    }

    SplitSpan subspan(size_t logical_start, size_t length) const
    {
        if (length == 0) [[unlikely]]
        {
            return SplitSpan{};
        }
        if (logical_start + length > size_) [[unlikely]]
        {
            throw std::out_of_range("Subspan exceeds buffer size");
        }

        const size_t physical_start = (start_ + logical_start) % N;
        const size_t contig_size = N - physical_start;

        if (length <= contig_size) [[likely]]
        {
            return {std::span(data_.data() + physical_start, length), {}};
        }
        return {
            std::span(data_.data() + physical_start, contig_size),
            std::span(data_.data(), length - contig_size)};
    }

    template <size_t M>
    std::array<T, M> to_array(size_t logical_start) const
    {
        auto split = subspan(logical_start, M); // M is known at compile time
        std::array<T, M> arr;
        std::copy(split.head.begin(), split.head.end(), arr.begin());
        std::copy(split.tail.begin(), split.tail.end(), arr.begin() + split.head.size());
        return arr;
    }

    size_t size() const noexcept
    {
        return size_;
    }
    bool full() const noexcept
    {
        return size_ == N;
    }
};