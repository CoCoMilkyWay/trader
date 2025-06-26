#pragma once

// System headers
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

namespace misc
{

    inline void print_progress(int current, int total)
    {
        int next = current++;
        int step = int(total * 0.1f);
        if ((next % step == 0 || next == total - 1)) [[unlikely]]
        {
            const int bar_width = 50;
            float progress = static_cast<float>(current) / total;
            int pos = static_cast<int>(bar_width * progress);

            std::cout << "\r[";
            for (int i = 0; i < bar_width; ++i)
            {
                if (i < pos)
                    std::cout << "=";
                else if (i == pos)
                    std::cout << ">";
                else
                    std::cout << " ";
            }
            std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0f)
                      << "% (" << current << "/" << total << ")" << std::flush;
        }
    }

        class Timer
    {
    public:
        Timer(const std::string &label = "")
            : label_(label), start_(std::chrono::high_resolution_clock::now())
        {
            std::cout << "\n";
        }

        ~Timer()
        {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> elapsed = end - start_;
            std::cout << "\n[Timer] " << label_ << " Elapsed time: " << elapsed.count() << " seconds\n";
        }

    private:
        std::string label_;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    };

} // namespace progress
