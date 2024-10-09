#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace smalltopk {

template<size_t DIM>
__attribute_noinline__ void transpose(
    const float* const __restrict src,
    const size_t n,
    float* const __restrict dst
) {
    for (size_t j = 0; j < DIM; j++) {
        for (size_t i = 0; i < n; i++) {
            dst[j * n + i] = src[j + i * DIM];
        }
    }    
}

// turns (n, d) array into (d, nn) array.
// if (nn > n), then missing parts of the original array
//     will be initialized with a default_value.
template<typename T, typename U = T>
static inline std::unique_ptr<T[]> transpose_and_fill(
    const U* const __restrict src,
    const size_t n,
    const size_t d,
    const size_t nn,
    const T default_value
) {
    std::unique_ptr<T[]> transposed = std::make_unique<T[]>(d * nn);

    // transpose
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            transposed[j * nn + i] = static_cast<T>(src[j + i * d]);
        }
    }

    // leftovers
    for (size_t i = n; i < nn; i++) {
        for (size_t j = 0; j < d; j++) {
            transposed[j * nn + i] = default_value;
        }
    }

    // done
    return transposed;
}

}  // namespace smalltopk
