#pragma once

#include <cstddef>
#include <cstdint>

#include <smalltopk/x86/avx512_vec_fp16.h>

#include <smalltopk/utils/norms-inl.h>
#include <smalltopk/utils/distances-inl.h>

namespace smalltopk {

template<size_t DIM>
void compute_norms_fp16(
    const float* const __restrict x,
    const size_t nx,
    uint16_t* const __restrict x_norm_i
) {
    for (size_t nx_k = 0; nx_k < nx; nx_k++) {
        // x address
        const float* const x_ptr = x + nx_k * DIM;  

        // norms
        x_norm_i[nx_k] = fp32_to_fp16(l2_sqr<DIM>(x_ptr));
    }
}

static inline void compute_norms_inline_fp16(
    const float* const __restrict x,
    const size_t nx,
    const size_t dim,
    uint16_t* const __restrict x_norm_i
) {
#define DISPATCH(DIM_V) \
    case DIM_V: compute_norms_fp16<DIM_V>(x, nx, x_norm_i); return;

    switch(dim) {
        DISPATCH(1)
        DISPATCH(2)
        DISPATCH(3)
        DISPATCH(4)
        DISPATCH(5)
        DISPATCH(6)
        DISPATCH(7)
        DISPATCH(8)
        DISPATCH(9)
        DISPATCH(10)
        DISPATCH(11)
        DISPATCH(12)
        DISPATCH(13)
        DISPATCH(14)
        DISPATCH(15)
        DISPATCH(16)
        DISPATCH(17)
        DISPATCH(18)
        DISPATCH(19)
        DISPATCH(20)
        DISPATCH(21)
        DISPATCH(22)
        DISPATCH(23)
        DISPATCH(24)
        DISPATCH(25)
        DISPATCH(26)
        DISPATCH(27)
        DISPATCH(28)
        DISPATCH(29)
        DISPATCH(30)
        DISPATCH(31)
        DISPATCH(32)
    }

    for (size_t nx_k = 0; nx_k < nx; nx_k++) {
        // x address
        const float* const x_ptr = x + nx_k * dim;  

        // norms
        float sum = 0;
        for (size_t i = 0; i < dim; i++) {
            sum += x_ptr[i] * x_ptr[i];
        }

        x_norm_i[nx_k] = fp32_to_fp16(sum);
    }

#undef DISPATCH
}

}  // namespace smalltopk
