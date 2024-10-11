#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

extern "C" {
#include <smalltopk/smalltopk.h>
}

#include "from_faiss/distances.h"
#include "from_faiss/heap.h"
#include "from_faiss/ordered_key_value.h"
#include "from_faiss/platform_macros.h"
#include "from_faiss/result_handlers.h"
#include "from_faiss/search.h"

#include "utils.h"

using namespace smalltopk::from_faiss;

// set running mode to 1 to run a subset of tests
// set running mode to 2 to run benchmarks
// otherwise, all tests are run

#define RUNNING_MODE 1

// the default parameters are for RUNNING_MODE 1
struct TestingParameters {
    bool print_log = false;

    std::vector<size_t> typical_x_sizes = 
        { 0, 1, 10, 100, 1000 };
    std::vector<size_t> typical_dims = 
        { 0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 25, 27, 31, 32 };
    std::vector<size_t> typical_y_sizes = 
        { 256 };
    std::vector<size_t> top_k_values = { 
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    };
    std::vector<uint32_t> smalltopk_kernels = { 0, 3 };

    bool compare_baseline_1 = true;
    bool compare_baseline_2 = false;
    bool test_supplied_norms = true;
    bool test_smalltopk_nlevels = false;

    bool validate_recall = true;
};

void perform_test(const TestingParameters& params) {
    for (size_t x_size : params.typical_x_sizes) {
        for (size_t dim : params.typical_dims) {
            const uint64_t rng_seed = 
                std::hash<size_t>()(x_size) ^ std::hash<size_t>()(dim);
            std::default_random_engine rng(rng_seed);

            std::vector<float> x = generate_dataset(x_size, dim, rng);
            std::vector<float> x_norms = generate_norms(x_size, dim, x);

            for (size_t y_size : params.typical_y_sizes) {
                std::vector<float> y = generate_dataset(y_size, dim, rng);
                std::vector<float> y_norms = generate_norms(y_size, dim, y);

                //
                using C = CMax<float, smalltopk_knn_l2sqr_ids_type>;

                ////////////////////////////////////////
                for (size_t k : params.top_k_values) {
                    // reference for baseline 1
                    std::vector<float> dis_ref_1(x_size * k, std::numeric_limits<float>::max());
                    std::vector<smalltopk_knn_l2sqr_ids_type> ids_ref_1(x_size * k, -1);

                    // evaluate baseline 1
                    StopWatch sw_ref_1;

                    if (params.compare_baseline_1) {
                        if (k == 1) {
                            using br_type = Top1BlockResultHandler<C>; 
                            br_type top1_ref(x_size, dis_ref_1.data(), ids_ref_1.data());
                            
                            exhaustive_L2sqr_seq<br_type>(
                                x.data(),
                                y.data(),
                                dim,
                                x_size,
                                y_size,
                                top1_ref
                            );
                        } else {
                            using br_type = HeapBlockResultHandler<C>;
                            br_type heap_ref(x_size, dis_ref_1.data(), ids_ref_1.data(), k);

                            exhaustive_L2sqr_seq<br_type>(
                                x.data(),
                                y.data(),
                                dim,
                                x_size,
                                y_size,
                                heap_ref
                            );
                        }
                    }

                    const double ref_elapsed_1 = sw_ref_1.elapsed();

                    // reference for baseline 2
                    std::vector<float> dis_ref_2(x_size * k, std::numeric_limits<float>::max());
                    std::vector<smalltopk_knn_l2sqr_ids_type> ids_ref_2(x_size * k, -1);

                    // evaluate baseline 2
                    StopWatch sw_ref_2;

                    if (params.compare_baseline_2) {
                        if (k == 1) {
                            using br_type = Top1BlockResultHandler<C>; 
                            br_type top1_ref(x_size, dis_ref_2.data(), ids_ref_2.data());

                            exhaustive_L2sqr_blas_default_impl<br_type>(
                                x.data(),
                                y.data(),
                                dim,
                                x_size,
                                y_size,
                                top1_ref,
                                nullptr
                            );
                        } else {
                            using br_type = HeapBlockResultHandler<C>;
                            br_type heap_ref(x_size, dis_ref_2.data(), ids_ref_2.data(), k);

                            exhaustive_L2sqr_blas_default_impl<br_type>(
                                x.data(),
                                y.data(),
                                dim,
                                x_size,
                                y_size,
                                heap_ref,
                                nullptr
                            );
                        }
                    }

                    const double ref_elapsed_2 = sw_ref_2.elapsed();

                    // decide whether to test multiple params, 
                    //   related to passing norms for x and y
                    std::vector<std::tuple<bool, bool>> norms_params;
                    if (params.test_supplied_norms) {
                        norms_params.push_back(std::make_tuple(false, false));
                        norms_params.push_back(std::make_tuple(true, false));
                        norms_params.push_back(std::make_tuple(false, true));
                        norms_params.push_back(std::make_tuple(true, true));
                    } else {
                        norms_params.push_back(std::make_tuple(false, false));
                    }

                    // candidate
                    for (uint32_t smalltopk_kernel : params.smalltopk_kernels) {
                        // decide whether to test multiple nlevels params,
                        //   which is applicable to approx kernels
                        // as of today, nlevels is fixed to 8. TODO.
                        std::vector<uint32_t> nlevels_params;
                        nlevels_params.push_back(0);

                        // loop
                        for (uint32_t smalltopk_nlevels : nlevels_params) {
                            for (const auto [pass_x_norms, pass_y_norms] : norms_params) {
                                std::vector<float> dis_new(x_size * k, std::numeric_limits<float>::max());
                                std::vector<smalltopk_knn_l2sqr_ids_type> ids_new(x_size * k, -1);

                                // evaluate
                                StopWatch sw_candidate;

                                KnnL2sqrParameters smalltopk_params;
                                smalltopk_params.kernel = smalltopk_kernel;
                                smalltopk_params.n_levels = smalltopk_nlevels;

                                bool success = knn_L2sqr_fp32(
                                    x.data(),
                                    y.data(),
                                    dim,
                                    x_size,
                                    y_size,
                                    k,
                                    pass_x_norms ? x_norms.data() : nullptr,
                                    pass_y_norms ? y_norms.data() : nullptr,
                                    dis_new.data(),
                                    ids_new.data(),
                                    &smalltopk_params
                                );

                                const double candidate_elapsed = sw_candidate.elapsed();

                                // compute the recall rate for ref 1
                                const double recall_rate_1 = 
                                    compute_recall_rate(x_size, k, ids_ref_1, ids_new);
                                const double recall_rate_2 = 
                                    compute_recall_rate(x_size, k, ids_ref_2, ids_new);

                                if (params.print_log) {
                                    std::cout << "test "
                                        << ", x_size = " << x_size
                                        << ", y_size = " << y_size
                                        << ", dim = " << dim 
                                        << ", k = " << k
                                        << ", kernel = " << smalltopk_params.kernel
                                        << ", nlevels = " << smalltopk_params.n_levels
                                        << ", x_norms provided = " << ((pass_x_norms) ? 1 : 0)
                                        << ", y_norms provided = " << ((pass_y_norms) ? 1 : 0)
                                        << ", success = " << ((success) ? 1 : 0)
                                        << ", ref1 time = " << ((params.compare_baseline_1) ? ref_elapsed_1 : -1)
                                        << ", ref2 time = " << ((params.compare_baseline_2) ? ref_elapsed_2 : -1)
                                        << ", cand time = " << candidate_elapsed
                                        << ", ref1/cand = " << ((params.compare_baseline_1 && success) ? (ref_elapsed_1 / candidate_elapsed) : -1)
                                        << ", ref2/cand = " << ((params.compare_baseline_2 && success) ? (ref_elapsed_2 / candidate_elapsed) : -1)
                                        << ", recall1 = " << ((params.compare_baseline_1) ? recall_rate_1 : -1) 
                                        << ", recall2 = " << ((params.compare_baseline_2) ? recall_rate_2 : -1)
                                        << std::endl;
                                }

                                if (params.validate_recall) {
                                    float threshold = 0.99f;
                                    if (smalltopk_params.kernel != 1) {
                                        threshold = 0.98f;
                                    }
                                    if (dim == 1) {
                                        threshold = 0.85f;
                                    }

                                    if (params.compare_baseline_1) {
                                        if (success) {
                                            EXPECT_GT(recall_rate_1, threshold)
                                                << ", x_size = " << x_size
                                                << ", y_size = " << y_size
                                                << ", dim = " << dim 
                                                << ", k = " << k
                                                << ", kernel = " << smalltopk_params.kernel
                                                << ", nlevels = " << smalltopk_params.n_levels;
                                        }
                                    }

                                    if (params.compare_baseline_2) {
                                        if (success) {
                                            EXPECT_GT(recall_rate_2, threshold)
                                                << ", x_size = " << x_size
                                                << ", y_size = " << y_size
                                                << ", dim = " << dim 
                                                << ", k = " << k
                                                << ", kernel = " << smalltopk_params.kernel
                                                << ", nlevels = " << smalltopk_params.n_levels;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#if RUNNING_MODE == 1

TEST(SmallTopKTest, validation_default) {
    TestingParameters params;
    params.print_log = false;
    params.typical_x_sizes = { 0, 1, 10, 100, 1000 };
    params.typical_dims = { 0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 25, 27, 31, 32 };
    params.typical_y_sizes = { 256 };
    params.top_k_values = { 
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    };
    params.smalltopk_kernels = { 3 };

    params.compare_baseline_1 = true;
    params.compare_baseline_2 = false;
    params.test_supplied_norms = true;
    params.test_smalltopk_nlevels = false;

    params.validate_recall = true;

    perform_test(params);
};

#elif RUNNING_MODE == 2

TEST(SmallTopK, validation_benchmark) {
    TestingParameters params;
    params.print_log = true;
    params.typical_x_sizes = { 1000000 };
    params.typical_dims = { 2, 4, 8, 12, 16, 20, 24, 28, 32 };
    params.typical_y_sizes = { 256 };
    params.top_k_values = { 
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    };

    // I prefer to use SMALLTOPK_KERNEL env variable to control this
    params.smalltopk_kernels = { 0 };

    params.compare_baseline_1 = true;
    params.compare_baseline_2 = true;
    params.test_supplied_norms = false;
    params.test_smalltopk_nlevels = true;

    params.validate_recall = false;

    perform_test(params);
}

#else

TEST(SmallTopKTest, validation_full) {
    TestingParameters params;
    params.print_log = false;
    params.typical_x_sizes = {
        0, 1, 2, 3, 10, 16, 17, 39, 47, 100, 1000, 10000, 
        2048, 2056, 2064, 2072, 2080, 2088, 2096, 2104, 2112
    };
    params.typical_dims = { 
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32 
    };
    params.typical_y_sizes = { 256 };
    params.top_k_values = { 
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    };
    params.smalltopk_kernels = { 1, 3, 5 };

    params.compare_baseline_1 = true;
    params.compare_baseline_2 = true;
    params.test_supplied_norms = true;
    params.test_smalltopk_nlevels = true;

    params.validate_recall = true;

    perform_test(params);
};

#endif
