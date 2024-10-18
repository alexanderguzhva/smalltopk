#include <smalltopk/dummy.h>

#include <smalltopk/types.h>

namespace smalltopk {

// does nothing
bool knn_L2sqr_fp32_dummy(
    const float* const __restrict,
    const float* const __restrict,
    const uint8_t,
    const uint64_t,
    const uint64_t,
    const uint8_t,
    const float* const __restrict,
    const float* const __restrict,
    float* const __restrict,
    smalltopk_knn_l2sqr_ids_type* const __restrict,
    const KnnL2sqrParameters* const __restrict
) {
    return false;
}

// does nothing
bool get_min_k_fp32_dummy(
    const float* const __restrict,
    const uint32_t,
    const uint8_t,
    float* const __restrict,
    int32_t* const __restrict,
    const GetKParameters* const __restrict
) {
    return false;
}

}  // namespace smalltopk
