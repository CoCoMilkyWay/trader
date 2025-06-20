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

#include <model/misc/misc.hpp>
#include <model/define/DataType.hpp>
#include <model/define/CBuffer.hpp>
#include <model/sample/ResampleRunBar.hpp>
#include <model/label/LabelCalmar.hpp>

#include <model/math/mini_model/CandlePattern.hpp>

#include <umappp/umappp.hpp>

namespace py = pybind11;

#define BACKWARD_HAS_DW 1
#include <model/debug/backward.hpp>
namespace bw = backward;
bw::SignalHandling sh;

// =====================================
// Main processing function
// =====================================
py::bytes process_bars(const py::bytes input_data, const size_t input_rows)
{
    // input/output arrays ====================================================
    std::srand(42);

    const size_t struct_size = sizeof(TimeBar);
    const size_t expected_size = input_rows * struct_size;

    const std::string input_str = input_data;
    const size_t actual_size = input_str.size();

    if (actual_size != expected_size)
    {
        throw std::runtime_error("Input data size mismatch.");
    }

    std::vector<RunBar> output_bars;
    output_bars.reserve(input_rows);

    // data structs and features ==============================================

    CBuffer<int64_t, BLen> time_1;
    CBuffer<int, BLen> timedelta_1;
    CBuffer<float, BLen> open_1;
    CBuffer<float, BLen> high_1;
    CBuffer<float, BLen> low_1;
    CBuffer<float, BLen> close_1;
    CBuffer<float, BLen> vwap_1;
    CBuffer<int, BLen> candlesthregth;

    const TimeBar *bars = reinterpret_cast<const TimeBar *>(input_str.data());
    RunBar bar_out;

    ResampleRunBar ResRunBar(&time_1, &timedelta_1, &open_1, &high_1, &low_1, &close_1, &vwap_1);
    CandlePattern CandPat(&open_1, &high_1, &low_1, &close_1, &candlesthregth, &timedelta_1);
    {
        misc::Timer t("run bar");
        for (size_t i = 0; i < input_rows; ++i)
        {
            if (i % int(input_rows * 0.05f) && PyErr_CheckSignals() != 0) [[unlikely]]
                throw py::error_already_set();

            if (ResRunBar.process(bars[i]))
            {

                CandPat.process();

                bar_out.time = time_1.back();
                bar_out.timedelta = timedelta_1.back();
                bar_out.open = open_1.back();
                bar_out.high = high_1.back();
                bar_out.low = low_1.back();
                bar_out.close = close_1.back();
                bar_out.vwap = vwap_1.back();
                output_bars.push_back(bar_out);
            }

            misc::print_progress(i, input_rows);
        }
        std::cout << "\nOutput bars: " << output_bars.size() << " (" << (100.0 * output_bars.size() / input_rows) << "%)\n";
    }

    {
        misc::Timer t("label(calmar)");
        LabelCalmar label_calmar;
        label_calmar.process(output_bars);
    }

    if (0)
    // UMAP
    {
        misc::Timer t("umap(features)"); //
        static const std::unordered_map<std::string, std::function<double(const RunBar &)>> feature_extractors = {
            {"open", [](const RunBar &ob)
             { return ob.open; }},
            {"high", [](const RunBar &ob)
             { return ob.high; }},
            {"low", [](const RunBar &ob)
             { return ob.low; }},
            {"close", [](const RunBar &ob)
             { return ob.close; }},
            {"vwap", [](const RunBar &ob)
             { return ob.vwap; }},
            {"threshold", [](const RunBar &ob)
             { return ob.threshold; }}}; //
        const int ndim = static_cast<int>(feature_extractors.size());
        const int nobs = static_cast<int>(output_bars.size());
        std::vector<double> input_data(ndim * nobs); // column-major    //
        int i = 0;
        for (const auto &[fname, extractor] : feature_extractors)
        {
            for (int j = 0; j < nobs; ++j)
            {
                input_data[i * nobs + j] = extractor(output_bars[j]);
            }
            ++i;
        } //
        knncolle::VptreeBuilder<int, double, double> vp_builder(
            std::make_shared<knncolle::EuclideanDistance<double, double>>()); //
        size_t out_dim = 3;
        std::vector<double> embedding(nobs * out_dim); //
        umappp::Options opt;
        auto status = umappp::initialize(
            ndim, nobs,
            input_data.data(),
            vp_builder,
            out_dim,
            embedding.data(),
            opt);     //
        status.run(); //
        for (size_t i = 0; i < output_bars.size(); ++i)
        {
            output_bars[i].umap_x = static_cast<float>(embedding[i * out_dim + 0]);
            output_bars[i].umap_y = static_cast<float>(embedding[i * out_dim + 1]);
            output_bars[i].umap_z = static_cast<float>(embedding[i * out_dim + 2]);
        }
    }

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
