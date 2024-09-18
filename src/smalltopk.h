#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "smalltopk_params.h"

#define SMALLTOPK_EXPORT __attribute__((__visibility__("default")))

// performs a knn-search.
// returns whether the operation was performed or not.
// it is expected that:
// * nx is large and ny is relatively small.
// * d is small.
// * k is small.
SMALLTOPK_EXPORT bool knn_L2sqr_fp32(
    const float* const __restrict x,
    const float* const __restrict y,
    const uint8_t d,
    const uint64_t nx,
    const uint64_t ny,
    const uint8_t k,
    const float* const __restrict x_norm_l2sqr,
    const float* const __restrict y_norm_l2sqr,
    float* const __restrict dis,
    int64_t* const __restrict ids,
    const KnnL2sqrParameters* const __restrict params
);

// finds k elements with min distances
SMALLTOPK_EXPORT bool get_min_k_fp32(
    const float* const __restrict src_dis,
    const uint32_t n,
    const uint8_t k,
    float* const __restrict dis,
    int32_t* const __restrict ids,
    const GetKParameters* const __restrict params
);

#undef SMALLTOPK_EXPORT
