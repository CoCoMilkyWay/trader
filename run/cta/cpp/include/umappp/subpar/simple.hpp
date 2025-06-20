#ifndef SUBPAR_SIMPLE_HPP
#define SUBPAR_SIMPLE_HPP

#ifndef SUBPAR_CUSTOM_PARALLELIZE_SIMPLE
#include <vector>
#include <stdexcept>
#include <thread>
#include <type_traits>
#endif

/**
 * @file simple.hpp
 * @brief Parallelize individual tasks across workers.
 */

namespace subpar {

/**
 * @brief Parallelize individual tasks across workers.
 *
 * The aim is to parallelize the execution of tasks across workers, under the assumption that there is a 1:1 mapping between them.
 * This is most relevant when the overall computation has already been split up and assigned to workers outside of **subpar**.
 * In such cases, `parallelize_simple()` is more suitable than `parallelize_range()` as it avoids the unnecessary overhead of partitioning the task interval.
 * 
 * The `SUBPAR_USES_OPENMP_SIMPLE` macro will be defined as 1 if and only if OpenMP was used in the default scheme.
 * Users can define the `SUBPAR_NO_OPENMP_SIMPLE` macro to force `parallelize_simple()` to use `<thread>` even if OpenMP is available.
 * This is occasionally useful when OpenMP cannot be used in some parts of the application, e.g., with POSIX forks.
 *
 * Advanced users can substitute in their own parallelization scheme by defining `SUBPAR_CUSTOM_PARALLELIZE_SIMPLE` before including the **subpar** header.
 * This should be a function-like macro that accepts the same arguments as `parallelize_simple()` or the name of a function that accepts the same arguments as `parallelize_simple()`.
 * If defined, the custom scheme will be used instead of the default scheme whenever `parallelize_simple()` is called.
 * Macro authors should note the expectations on `run_task()`.
 *
 * If `nothrow_ = true`, exception handling is omitted from the default parallelization scheme.
 * This avoids some unnecessary work when the caller knows that `run_task()` will never throw. 
 * For custom schemes, if `SUBPAR_CUSTOM_PARALLELIZE_SIMPLE_NOTHROW` is defined, it will be used if `nothrow_ = true`;
 * otherwise, `SUBPAR_CUSTOM_PARALLELIZE_SIMPLE` will continue to be used.
 *
 * @tparam nothrow_ Whether the `Run_` function cannot throw an exception.
 * @tparam Task_ Integer type for the number of tasks.
 * @tparam Run_ Function that accepts `w`, the index of the task (and thus the worker) as a `Task_`.
 * Any return value is ignored.
 *
 * @param num_tasks Number of tasks.
 * This is also the number of workers as we assume a 1:1 mapping between tasks and workers.
 * @param run_task Function to execute the task for each worker.
 * This will be called exactly once in each worker, where `w` is guaranteed to be in `[0, num_tasks)`.
 * This function may throw an exception if `nothrow_ = false`.
 */
template<bool nothrow_ = false, typename Task_, class Run_>
void parallelize_simple(Task_ num_tasks, Run_ run_task) {
#ifdef SUBPAR_CUSTOM_PARALLELIZE_SIMPLE
    if constexpr(nothrow_) {
#ifdef SUBPAR_CUSTOM_PARALLELIZE_SIMPLE_NOTHROW
        SUBPAR_CUSTOM_PARALLELIZE_SIMPLE_NOTHROW(num_tasks, run_task);
#else
        SUBPAR_CUSTOM_PARALLELIZE_SIMPLE(num_tasks, run_task);
#endif
    } else {
        SUBPAR_CUSTOM_PARALLELIZE_SIMPLE(num_tasks, run_task);
    }

#else
    if (num_tasks == 0) {
        return;
    } else if (num_tasks == 1) {
        run_task(0);
        return;
    }

    // Avoid instantiating a vector if it is known that the function can't throw.
    typename std::conditional<nothrow_, int, std::vector<std::exception_ptr> >::type errors(num_tasks);

#if defined(_OPENMP) && !defined(SUBPAR_NO_OPENMP_SIMPLE)
#define SUBPAR_USES_OPENMP_SIMPLE 1

    // OpenMP doesn't guarantee that we'll actually start 'num_tasks' workers,
    // so we need to do a loop here to ensure that each task simple is executed.
    #pragma omp parallel for num_threads(num_tasks)
    for (Task_ w = 0; w < num_tasks; ++w) {
        if constexpr(nothrow_) {
            run_task(w);
        } else {
            try { 
                run_task(w);
            } catch (...) {
                errors[w] = std::current_exception();
            }
        }
    }

#else
// Wiping it out, just in case.
#undef SUBPAR_USES_OPENMP_SIMPLE

    std::vector<std::thread> workers;
    workers.reserve(num_tasks);

    for (Task_ w = 0; w < num_tasks; ++w) {
        if constexpr(nothrow_) {
            workers.emplace_back(run_task, w);
        } else {
            workers.emplace_back([&run_task,&errors](int w) -> void {
                try {
                    run_task(w);
                } catch (...) {
                    errors[w] = std::current_exception();
                }
            }, w);
        }
    }

    for (auto& wrk : workers) {
        wrk.join();
    }
#endif

    if constexpr(!nothrow_) {
        for (const auto& e : errors) {
            if (e) {
                std::rethrow_exception(e);
            }
        }
    }
#endif
}

}

#endif
