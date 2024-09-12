#pragma once

#include <cstddef>
#include <cstdint>

namespace smalltopk {

//
void transpose(
    const float* const __restrict src,
    const size_t n_src,
    const size_t d,
    float* const __restrict dst,
    const size_t n_dst
);

}  // namespace smalltopk
