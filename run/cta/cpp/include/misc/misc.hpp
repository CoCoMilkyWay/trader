#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace misc
{

    inline void print_progress(int current, int total)
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

    class Timer
    {
    public:
        Timer(const std::string &label = "")
            : label_(label), start_(std::chrono::high_resolution_clock::now()) {}

        ~Timer()
        {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start_;
            std::cout << "\n[Timer] " << label_ << " Elapsed time: " << elapsed.count() << " seconds\n";
        }

    private:
        std::string label_;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    };

    // Helper function to print any number of arguments with a separator
    template <typename... Args>
    void print(const Args &...args)
    {
        const std::string &sep = " ";
        std::ostringstream oss;
        ((oss << args << sep), ...); // Fold expression (C++17)
        std::string result = oss.str();
        if (!result.empty())
        {
            result.erase(result.size() - sep.size()); // Remove trailing separator
        }
        std::cout << result << '\n';
    }

} // namespace progress
