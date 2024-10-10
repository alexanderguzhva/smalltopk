#include <smalltopk/x86/avx512_sorting_fp32hack.h>

namespace smalltopk {

bool knn_L2sqr_fp32_avx512_sorting_fp32hack(
    const float* const __restrict x,
    const float* const __restrict y_in,
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

}  // namespace smalltopk
