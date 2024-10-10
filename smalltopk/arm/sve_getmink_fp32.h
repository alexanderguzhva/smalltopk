#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {
#include <smalltopk/smalltopk_params.h>
}

namespace smalltopk {

// finds k elements with min distances
bool get_min_k_fp32_sve(
    const float* const __restrict src_dis,
    const uint32_t n,
    const uint8_t k,
    float* const __restrict dis,
    int32_t* const __restrict ids,
    const GetKParameters* const __restrict params
);

}  // namespace smalltopk
