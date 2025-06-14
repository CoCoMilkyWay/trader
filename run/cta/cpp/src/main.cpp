#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <iostream>
#include <cmath>
#include <stdexcept>

#include <misc/misc.hpp>
#include <define/DataType.hpp>
#include <sample/VolumeRunBar.hpp>
#include <label/LabelCalmar.hpp>

namespace py = pybind11;

#define BACKWARD_HAS_DW 1
#include <debug/backward.hpp>
namespace bw = backward;
bw::SignalHandling sh; // Installs handlers for SIGSEGV, etc.

// =====================================
// Main processing function
// =====================================
py::bytes process_bars(
    py::bytes input_data,
    size_t input_rows)
{
    std::srand(42);

    const size_t struct_size = sizeof(TimeBar);
    const size_t expected_size = input_rows * struct_size;

    std::string input_str = input_data;
    const size_t actual_size = input_str.size();

    if (actual_size != expected_size)
    {
        throw std::runtime_error("Input data size mismatch.\n");
    }

    const TimeBar *bars = reinterpret_cast<const TimeBar *>(input_str.data());

    std::vector<RunBar> output_bars;
    output_bars.reserve(input_rows); // Preallocate

    VolumeRunBar run_bar(1, 5);
    RunBar bar_out;

    {
        misc::Timer t("run bar");
        int step = std::max(1, static_cast<int>(input_rows / 20));
        for (size_t i = 0; i < input_rows; ++i)
        {
            // Respond to Ctrl+C / interrupt
            if (i % int(input_rows * 0.05f) && PyErr_CheckSignals() != 0) [[unlikely]]
                throw py::error_already_set();

            if (run_bar.process(bars[i], bar_out))
            {
                output_bars.push_back(bar_out);
            }

            if ((i % step == 0 || i == input_rows - 1)) [[unlikely]]
                misc::print_progress(i + 1, input_rows);
        }
    }

    std::cout << " Output bars: " << output_bars.size() << " (" << (100.0 * output_bars.size() / input_rows) << "%)\n";
    {
        misc::Timer t("label(calmar)");
        LabelCalmar label_calmar;
        label_calmar.process(output_bars);
    }
    // Convert output vector to bytes
    std::string output_data(
        reinterpret_cast<const char *>(output_bars.data()),
        output_bars.size() * sizeof(RunBar));

    return py::bytes(output_data);
}

// =====================================
// Pybind11 module definition
// =====================================
PYBIND11_MODULE(Pipeline, m)
{
    m.def("process_bars", &process_bars,
          py::arg("input_data"),
          py::arg("input_rows"));
}
