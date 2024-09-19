#include "sve_getmink_fp32.h"

#include <cstddef>
#include <cstdint>


namespace smalltopk {

// finds k elements with min distances
bool get_min_k_fp32_sve(
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
