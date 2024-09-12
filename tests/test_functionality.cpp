#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

extern "C" {
#include <smalltopk.h>
}

#include "from_faiss/distances.h"
#include "from_faiss/heap.h"
#include "from_faiss/ordered_key_value.h"
#include "from_faiss/platform_macros.h"
#include "from_faiss/result_handlers.h"
#include "from_faiss/search.h"

using namespace smalltopk::from_faiss;

namespace {

template<typename RandomT>
std::vector<float> generate_dataset(const size_t n, const size_t d, RandomT& rng) {
    std::uniform_real_distribution<float> u(-1, 1);

    std::vector<float> data(n * d);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = u(rng);
    }

    return data;
}

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

//
double compute_recall_rate(
    const size_t x_size,
    const size_t k,
    const std::vector<int64_t>& ids_ref, 
    const std::vector<int64_t>& ids_new
) {
    // compute the recall rate
    int64_t n_matches = 0;
    int64_t n_total = 0;

    for (size_t ix = 0; ix < x_size; ix++) {
        std::vector<int64_t> ref_v(ids_ref.data() + ix * k, ids_ref.data() + (ix + 1) * k);
        std::sort(ref_v.begin(), ref_v.end());

        std::vector<int64_t> new_v(ids_new.data() + ix * k, ids_new.data() + (ix + 1) * k);
        std::sort(new_v.begin(), new_v.end());

        // intersect
        std::vector<int64_t> similar;
        
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

}

TEST(foo, shmoo) {
    const size_t x_size = 1000000;

    std::default_random_engine rng(123);

    for (size_t dim : {2, 4, 8, 12, 16, 20, 24, 28, 32}) {
    // for (size_t dim : {28, 32}) {
        std::vector<float> x = generate_dataset(x_size, dim, rng);


        for (size_t y_size : {256}) {
            std::vector<float> y = generate_dataset(y_size, dim, rng);
        
            // // // //
            using C = CMax<float, int64_t>;

            ////////////////////////////////////////
            // perform top1 search
            {
                size_t k = 1;
                using br_type = Top1BlockResultHandler<C>; 

                // reference one
                std::vector<float> dis_ref(x_size * k, std::numeric_limits<float>::max());
                std::vector<int64_t> ids_ref(x_size * k, -1);
                br_type top1_ref(x_size, dis_ref.data(), ids_ref.data());

                // evaluate
                StopWatch sw_ref;

                exhaustive_L2sqr_seq<br_type>(
                    x.data(),
                    y.data(),
                    dim,
                    x_size,
                    y_size,
                    top1_ref
                );

                // exhaustive_L2sqr_blas_default_impl<br_type>(
                //     x.data(),
                //     y.data(),
                //     dim,
                //     x_size,
                //     y_size,
                //     top1_ref,
                //     nullptr
                // );

                const double ref_elapsed = sw_ref.elapsed();

                // candidate one
                {
                    std::vector<float> dis_new(x_size * k, std::numeric_limits<float>::max());
                    std::vector<int64_t> ids_new(x_size * k, -1);
                    // // // br_type top1_new(x_size, dis_new.data(), ids_new.data());

                    // benchmark
                    StopWatch sw_new;

                    // KnnL2sqrParameters params;
                    // params.kernel = 0;
                    // params.n_levels = 8;

                    knn_L2sqr_fp32(
                        x.data(),
                        y.data(),
                        dim,
                        x_size,
                        y_size,
                        1,
                        nullptr,
                        nullptr,
                        dis_new.data(),
                        ids_new.data(),
                        nullptr
                    );

                    const double new_elapsed = sw_new.elapsed();

                    // compute the recall rate
                    double recall_rate = compute_recall_rate(x_size, k, ids_ref, ids_new);

                    // perform heap-based topk search
                    std::cerr << "top1 "
                        << ", x_size = " << x_size
                        << ", y_size = " << y_size
                        << ", k = " << k
                        << ", nlevel = " << 1
                        << ", dim = " << dim 
                        << ", ref = " << ref_elapsed 
                        << ", new = " << new_elapsed 
                        << ", ref/new = " << (ref_elapsed / new_elapsed)
                        << ", recall = " << recall_rate << std::endl;
                }
            }

            ////////////////////////////////////////
            // perform heap search
            for (size_t k = 2; k <= 24; k += 1) {
                using br_type = HeapBlockResultHandler<C>;

                // reference one
                std::vector<float> dis_ref(x_size * k, std::numeric_limits<float>::max());
                std::vector<int64_t> ids_ref(x_size * k, -1);
                br_type heap_ref(x_size, dis_ref.data(), ids_ref.data(), k);

                // evaluate
                StopWatch sw_ref;

                exhaustive_L2sqr_seq<br_type>(
                    x.data(),
                    y.data(),
                    dim,
                    x_size,
                    y_size,
                    heap_ref
                );

                // exhaustive_L2sqr_blas_default_impl<br_type>(
                //     x.data(),
                //     y.data(),
                //     dim,
                //     x_size,
                //     y_size,
                //     heap_ref,
                //     nullptr
                // );

                const double ref_elapsed = sw_ref.elapsed();

                for (size_t nlevels = 1; nlevels <= 1; nlevels++) {
                    // if (nlevels > k) {
                    //     // makes no sense
                    //     continue;
                    // }

                    //for (size_t nxpoint = 1; nxpoint <= 16; nxpoint++) {
                    for (size_t nxpoint = 16; nxpoint <= 16; nxpoint++) {

                        // candidate one
                        std::vector<float> dis_new(x_size * k, std::numeric_limits<float>::max());
                        std::vector<int64_t> ids_new(x_size * k, -1);
                        br_type heap_new(x_size, dis_new.data(), ids_new.data(), k);

                        // benchmark
                        StopWatch sw_new;

                        // KnnL2sqrParameters params;
                        // params.kernel = 0;
                        // params.n_levels = nlevels;

                        knn_L2sqr_fp32(
                            x.data(),
                            y.data(),
                            dim,
                            x_size,
                            y_size,
                            k,
                            nullptr,
                            nullptr,
                            dis_new.data(),
                            ids_new.data(),
                            nullptr
                        );

                        const double new_elapsed = sw_new.elapsed();

                        // compute the recall rate
                        double recall_rate = compute_recall_rate(x_size, k, ids_ref, ids_new);

                        // perform heap-based topk search
                        std::cout << "heap "
                            << ", x_size = " << x_size
                            << ", y_size = " << y_size
                            << ", k = " << k
                            << ", nlevel = " << nlevels
                            << ", nxpoint = " << nxpoint
                            << ", dim = " << dim 
                            << ", ref = " << ref_elapsed 
                            << ", new = " << new_elapsed 
                            << ", ref/new = " << (ref_elapsed / new_elapsed)
                            << ", recall = " << recall_rate << std::endl;
                    }
                }
            }
        }
    }
}
