#include <gtest/gtest.h>

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

#include "utils.h"

using namespace smalltopk::from_faiss;


TEST(foo, getmink) {
    const size_t x_size = 256;
    const size_t n_samples = 10000;

    const size_t maxk = 24;

    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> u(0, 1);

    // generate a dataset
    std::vector<float> x(n_samples * x_size * maxk, 0);

    for (size_t i = 0; i < x.size(); i++) {
        x[i] = u(rng); 
    }

    //
    for (size_t k = 1; k <= maxk; k += 1) {

        //
        using C = CMax<float, int>;

        std::vector<float> baseline_dis(k * n_samples, C::neutral());
        std::vector<int> baseline_ids(k * n_samples, -1);

        //
        GetKParameters params;
        params.n_levels = 1 + (k + 1) / 3; //(k <= 8) ? k : 8;
        
        std::vector<float> candidate_dis(k * n_samples, C::neutral());
        std::vector<int32_t> candidate_ids(k * n_samples, -1);

        //
        StopWatch sw_baseline;

        for (size_t i = 0; i < n_samples; i++) {
            heap_addn<C>(
                    k,
                    baseline_dis.data() + i * k,
                    baseline_ids.data() + i * k,
                    x.data() + i * x_size * k,
                    nullptr,
                    x_size * k);
            heap_reorder<C>(k, baseline_dis.data() + i * k, baseline_ids.data() + i * k);
        }

        const double baseline_elapsed = sw_baseline.elapsed();

        //
        StopWatch sw_candidate;

        for (size_t i = 0; i < n_samples; i++) {
            get_min_k_fp32(
                    x.data() + i * x_size * k, 
                    x_size * k, 
                    k, 
                    candidate_dis.data() + i * k, 
                    candidate_ids.data() + i * k, 
                    &params);
        }

        const double candidate_elapsed = sw_candidate.elapsed();

        // calc recall
        const double recall = compute_recall_rate(n_samples, k, baseline_ids, candidate_ids);

        printf("k = %zd, recall = %f, baseline %f ms, candidate %f ms\n", 
            k, recall, baseline_elapsed, candidate_elapsed);

        for (size_t j = 0; j < k; j++) {
            printf("%d ", baseline_ids[j]);
        }
        printf("\n");

        for (size_t j = 0; j < k; j++) {
            printf("%d ", candidate_ids[j]);
        }
        printf("\n");

        printf("\n");

    }
}