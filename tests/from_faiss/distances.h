// reworked from faiss

#pragma once

#include <cstddef>
#include <cstdint>

#include "platform_macros.h"

namespace smalltopk {
namespace from_faiss {

//
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
static inline float fvec_norm_L2sqr(const float* x, size_t d) {
    // the double in the _ref is suspected to be a typo. Some of the manual
    // implementations this replaces used float.
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * x[i];
    }

    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

//
static inline void fvec_norms_L2sqr(
        float* __restrict nr,
        const float* __restrict x,
        size_t d,
        size_t nx) {
#pragma omp parallel for if (nx > 10000)
    for (int64_t i = 0; i < nx; i++)
        nr[i] = fvec_norm_L2sqr(x + i * d, d);
}

//
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
static inline float fvec_L2sqr(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

}
}
