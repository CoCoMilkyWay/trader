#ifndef SUBPAR_RANGE_HPP
#define SUBPAR_RANGE_HPP

#include <limits>

#ifndef SUBPAR_CUSTOM_PARALLELIZE_RANGE
#include <vector>
#include <stdexcept>
#include <thread>
#include <type_traits>
#endif

/**
 * @file range.hpp
 * @brief Parallelize across a range of tasks.
 */

namespace subpar {

/**
 * @cond
 */
namespace internal {

template<typename Task_>
bool ge(int num_workers, Task_ num_tasks) { // We already assume that both of them are non-negative at this point.
    if constexpr(static_cast<size_t>(std::numeric_limits<int>::max()) > static_cast<size_t>(std::numeric_limits<Task_>::max())) {
        return num_workers >= static_cast<int>(num_tasks);
    } else {
        return static_cast<Task_>(num_workers) >= num_tasks;
    }
}

}
/**
 * @endcond
 */

/**
 * @brief Adjust the number of workers to the number of tasks in `parallelize_range()`.
 *
 * It is not strictly necessary to run `sanitize_num_workers()` prior to `parallelize_range()` as the latter will automatically behave correctly with all inputs.
 * However, on occasion, applications need a better upper bound on the number of workers, e.g., to pre-allocate expensive per-worker data structures.
 * In such cases, the return value of `sanitize_num_workers()` can be used by the application before being passed to `parallelize_range()`.
 *
 * @tparam Task_ Integer type for the number of tasks.
 *
 * @param num_workers Number of workers.
 * This may be negative or zero.
 * @param num_tasks Number of tasks.
 * This should be a non-negative integer.
 *
 * @return A more suitable number of workers.
 * Negative or zero `num_workers` are converted to 1 if `num_tasks > 0`, otherwise zero.
 * If `num_workers` is greater than `num_tasks`, the former is set to the latter.
 */
template<typename Task_>
int sanitize_num_workers(int num_workers, Task_ num_tasks) {
    if (num_workers <= 0) {
        return (num_tasks > 0);
    }

    if (internal::ge(num_workers, num_tasks)) {
        return num_tasks;
    }

    return num_workers;
}

/**
 * @brief Parallelize a range of tasks across multiple workers.
 *
 * The aim is to split tasks in `[0, num_tasks)` into non-overlapping contiguous ranges that are executed by different workers.
 * In the default parallelization scheme, we create `num_workers` evenly-sized ranges that are executed via OpenMP (if available) or `<thread>` (otherwise).
 * Not all workers may be used, e.g., if `num_tasks < num_workers`, but each worker will process no more than one range.
 * 
 * The `SUBPAR_USES_OPENMP_RANGE` macro will be defined as 1 if and only if OpenMP was used in the default scheme.
 * Users can define the `SUBPAR_NO_OPENMP_RANGE` macro to force `parallelize_range()` to use `<thread>` even if OpenMP is available.
 * This is occasionally useful when OpenMP cannot be used in some parts of the application, e.g., with POSIX forks.
 *
 * Advanced users can substitute in their own parallelization scheme by defining `SUBPAR_CUSTOM_PARALLELIZE_RANGE` before including the **subpar** header.
 * This should be a function-like macro that accepts the same arguments as `parallelize_range()` or the name of a function that accepts the same arguments as `parallelize_range()`.
 * If defined, the custom scheme will be used instead of the default scheme whenever `parallelize_range()` is called.
 * Macro authors should note the expectations on `run_task_range()`, as well as the one-to-zero-or-one mapping between workers and ranges.
 *
 * If `nothrow_ = true`, exception handling is omitted from the default parallelization scheme.
 * This avoids some unnecessary work when the caller knows that `run_task_range()` will never throw. 
 * For custom schemes, if `SUBPAR_CUSTOM_PARALLELIZE_RANGE_NOTHROW` is defined, it will be used if `nothrow_ = true`;
 * otherwise, `SUBPAR_CUSTOM_PARALLELIZE_RANGE` will continue to be used.
 *
 * @tparam nothrow_ Whether the `Run_` function cannot throw an exception.
 * @tparam Task_ Integer type for the number of tasks.
 * @tparam Run_ Function that accepts three arguments:
 * - `w`, the identity of the worker executing this task range.
 *   This will be passed as an `int`.
 * - `start`, the start index of the task range.
 *   This will be passed as a `Task_`.
 * - `length`, the number of tasks in the task range.
 *   This will be passed as a `Task_`.
 * .
 * Any return value is ignored.
 *
 * @param num_workers Number of workers.
 * This should be a positive integer.
 * Any zero or negative values are treated as 1.
 * (See also `sanitize_num_workers()`.)
 * @param num_tasks Number of tasks.
 * This should be a non-negative integer.
 * @param run_task_range Function to iterate over a range of tasks within a worker.
 * This may be called zero, one or multiple times in any particular worker.
 * In each call:
 * - `w` is guaranteed to be in `[0, num_workers)`.
 * - `[start, start + length)` is guaranteed to be a non-empty range of tasks that lies in `[0, num_tasks)`.
 *   It will not overlap with any other range in any other call to `run_task_range()`.
 * .
 * This function may throw an exception if `nothrow_ = false`.
 */
template<bool nothrow_ = false, typename Task_, class Run_>
void parallelize_range(int num_workers, Task_ num_tasks, Run_ run_task_range) {
#ifdef SUBPAR_CUSTOM_PARALLELIZE_RANGE
    if constexpr(nothrow_) {
#ifdef SUBPAR_CUSTOM_PARALLELIZE_RANGE_NOTHROW
        SUBPAR_CUSTOM_PARALLELIZE_RANGE_NOTHROW(num_workers, num_tasks, run_task_range);
#else
        SUBPAR_CUSTOM_PARALLELIZE_RANGE(num_workers, num_tasks, run_task_range);
#endif
    } else {
        SUBPAR_CUSTOM_PARALLELIZE_RANGE(num_workers, num_tasks, run_task_range);
    }

#else
    if (num_tasks == 0) {
        return;
    }

    if (num_workers <= 1 || num_tasks == 1) {
        run_task_range(0, 0, num_tasks);
        return;
    }

    // All workers with indices below 'remainder' get an extra task to fill up the remainder.
    Task_ tasks_per_worker;
    int remainder;
    if (internal::ge(num_workers, num_tasks)) {
        num_workers = num_tasks;
        tasks_per_worker = 1;
        remainder = 0;
    } else {
        tasks_per_worker = num_tasks / num_workers;
        remainder = num_tasks % num_workers;
    }

    // Avoid instantiating a vector if it is known that the function can't throw.
    typename std::conditional<nothrow_, int, std::vector<std::exception_ptr> >::type errors(num_workers);

#if defined(_OPENMP) && !defined(SUBPAR_NO_OPENMP_RANGE) && !defined(SUBPAR_NO_OPENMP)
#define SUBPAR_USES_OPENMP 1
#define SUBPAR_USES_OPENMP_RANGE 1

    // OpenMP doesn't guarantee that we'll actually start 'num_workers' workers,
    // so we need to do a loop here to ensure that each task range is executed.
    #pragma omp parallel for num_threads(num_workers)
    for (int w = 0; w < num_workers; ++w) {
        Task_ start = w * tasks_per_worker + (w < remainder ? w : remainder); // need to shift the start by the number of previous 't' that added a remainder.
        Task_ length = tasks_per_worker + (w < remainder);

        if constexpr(nothrow_) {
            run_task_range(w, start, length);
        } else {
            try { 
                run_task_range(w, start, length);
            } catch (...) {
                errors[w] = std::current_exception();
            }
        }
    }

#else
// Wiping it out, just in case.
#undef SUBPAR_USES_OPENMP
#undef SUBPAR_USES_OPENMP_RANGE

    Task_ start = 0;
    std::vector<std::thread> workers;
    workers.reserve(num_workers);

    for (int w = 0; w < num_workers; ++w) {
        Task_ length = tasks_per_worker + (w < remainder); 

        if constexpr(nothrow_) {
            workers.emplace_back(run_task_range, w, start, length);
        } else {
            workers.emplace_back([&run_task_range,&errors](int w, Task_ start, Task_ length) -> void {
                try {
                    run_task_range(w, start, length);
                } catch (...) {
                    errors[w] = std::current_exception();
                }
            }, w, start, length);
        }

        start += length;
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

/**
 * @cond
 */
// Back-compatibility only.
template<typename Task_, class Run_>
void parallelize(int num_workers, Task_ num_tasks, Run_ run_task_range) {
    parallelize_range<false, Task_, Run_>(num_workers, num_tasks, std::move(run_task_range));
}
/**
 * @endcond
 */

}

#endif
