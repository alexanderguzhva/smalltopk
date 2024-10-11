#pragma once

#include <omp.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

// stopwatch
struct StopWatch {
    using time_type = std::chrono::time_point<std::chrono::high_resolution_clock>;
    time_type start;

    StopWatch() {
        start = now();
    }

    inline double elapsed() const {
        auto current = now();
        return std::chrono::duration<double>(current - start).count();
    }

    static inline time_type now() {
        return std::chrono::high_resolution_clock::now();
    }
};


template<typename RandomT>
std::vector<float> generate_dataset(const size_t n, const size_t d, RandomT& rng) {

    const int max_nt = omp_get_max_threads();
    
    std::uniform_int_distribution<uint64_t> urng(0, 16777216 * 64);
    std::vector<uint64_t> rngs(max_nt);
    for (int i = 0; i < max_nt; i++) {
        rngs[i] = urng(rng);
    }

    std::vector<float> data(n * d);
#pragma omp parallel
    {
        std::uniform_real_distribution<float> u(-1, 1);

        const int rank = omp_get_thread_num();
        const int nt = omp_get_num_threads();

        RandomT rngx(rngs[rank]);

        const size_t c0 = ((n * d) * rank) / nt;
        const size_t c1 = ((n * d) * (rank + 1)) / nt;
        for (size_t i = c0; i < c1; i++) {
            data[i] = u(rngx);
        }        
    }

    // for (size_t i = 0; i < data.size(); i++) {
    //     data[i] = u(rng);
    // }

    return data;
}

std::vector<float> generate_norms(const size_t n, const size_t d, const std::vector<float>& x) {
    std::vector<float> norms(n);

    for (size_t i = 0; i < n; i++) {
        float norm = 0;
        for (size_t j = 0; j < d; j++) {
            const float v = x[i * d + j];
            norm += v * v;
        }

        norms[i] = norm;
    }

    return norms;
}

//
template<typename IndexT>
double compute_recall_rate(
    const size_t x_size,
    const size_t k,
    const std::vector<IndexT>& ids_ref, 
    const std::vector<IndexT>& ids_new
) {
    if (x_size == 0 || k == 0) {
        return 1;
    }

    // compute the recall rate
    int64_t n_matches = 0;
    int64_t n_total = 0;

    for (size_t ix = 0; ix < x_size; ix++) {
        std::vector<IndexT> ref_v(ids_ref.data() + ix * k, ids_ref.data() + (ix + 1) * k);
        std::sort(ref_v.begin(), ref_v.end());

        std::vector<IndexT> new_v(ids_new.data() + ix * k, ids_new.data() + (ix + 1) * k);
        std::sort(new_v.begin(), new_v.end());

        // intersect
        std::vector<IndexT> similar;
        
        std::set_intersection(
            ref_v.cbegin(),
            ref_v.cend(),
            new_v.cbegin(),
            new_v.cend(),
            std::back_inserter(similar)
        );

        // compute
        n_matches += similar.size();
        n_total += k;
    }

    double recall_rate = double(n_matches) / double(n_total);
    return recall_rate;
}
