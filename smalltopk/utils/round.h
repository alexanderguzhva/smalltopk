#pragma once

#include <cstddef>
#include <cstdint>

namespace smalltopk {

static inline uint32_t next_power_of_2(const uint32_t value) {
    // Round up to the next highest power of 2
    uint32_t next_pow2 = value; 
    next_pow2 -= 1;
    next_pow2 |= next_pow2 >> 1;
    next_pow2 |= next_pow2 >> 2;
    next_pow2 |= next_pow2 >> 4;
    next_pow2 |= next_pow2 >> 8;
    next_pow2 |= next_pow2 >> 16;
    next_pow2 += 1;

    return next_pow2;
}

}  // namespace smalltopk
