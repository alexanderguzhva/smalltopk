#include <smalltopk/x86/avx512_getmink_fp32hack.h>

#include <cstddef>
#include <cstdint>

namespace smalltopk {

// finds k elements with min distances
bool get_min_k_fp32hack_avx512(
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
