#include <smalltopk/dummy.h>

#include <smalltopk/types.h>

namespace smalltopk {

// does nothing
bool knn_L2sqr_fp32_dummy(
    const float* const __restrict x,
    const float* const __restrict y,
    const uint8_t d,
    const uint64_t nx,
    const uint64_t ny,
    const uint8_t k,
    const float* const __restrict x_norm_l2sqr,
    const float* const __restrict y_norm_l2sqr,
    float* const __restrict dis,
    smalltopk_knn_l2sqr_ids_type* const __restrict ids,
    const KnnL2sqrParameters* const __restrict params
) {
    return false;
}

// does nothing
bool get_min_k_fp32_dummy(
    const float* const __restrict src_dis,
    const uint32_t n,
    const uint8_t k,
    float* const __restrict dis,
    int32_t* const __restrict ids,
    const GetKParameters* const __restrict params
) {
    return false;
}

}  // namespace smalltopk
