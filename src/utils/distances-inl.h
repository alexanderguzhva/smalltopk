#pragma once

#include <cstdint>
#include <cstddef>

namespace smalltopk {

template <size_t DIM>
float l2_sqr(const float* const x) {
    float output = 0;

    for (size_t i = 0; i < DIM; i++) {
        output += x[i] * x[i];
    }

    return output;
}

}  // namespace smalltopk
