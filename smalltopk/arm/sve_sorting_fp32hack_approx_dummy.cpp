#include <smalltopk/arm/sve_sorting_fp32hack_approx.h>

namespace smalltopk {

bool knn_L2sqr_fp32_sve_sorting_fp32hack_approx(
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

}  // namespace smalltopk
