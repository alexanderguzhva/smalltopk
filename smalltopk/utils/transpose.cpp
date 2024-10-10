#include <smalltopk/utils/transpose.h>

#include <cstddef>
#include <cstdint>

namespace smalltopk {

void transpose(
    const float* const __restrict src,
    const size_t n_src,
    const size_t d,
    float* const __restrict dst,
    const size_t n_dst
) {
    if (d == 1) {
        return;
    }
    
    for (size_t j = 0; j < d; j++) {
        for (size_t i = 0; i < n_src; i++) {
            dst[j * n_dst + i] = src[j + i * d];
        }
    }
}

}  // namespace smalltopk
