#include <smalltopk/x86/avx512_getmink_fp32hack.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include <smalltopk/x86/avx512_vec_fp32.h>
#include <smalltopk/x86/kernel_getmink_fp32hack.h>

#include <smalltopk/utils/macro_repeat_define.h>

namespace smalltopk {

// finds k elements with min distances
bool get_min_k_fp32hack_avx512(
    const float* const __restrict src_dis,
    const uint32_t n,
    const uint8_t k,
    float* const __restrict dis,
    int32_t* const __restrict ids,
    const GetKParameters* const __restrict params
) {
    // nothing to do?
    if (n == 0 || k == 0) {
        return true;
    }

    // missing input?
    if (src_dis == nullptr) {
        return false;
    }

    // not supported?
    if (n > 65536) {
        // todo: copy-paste a version of this kernel that has int32_t n counter.
        return false;
    }

    //
    using distances_engine_type = vec_f32x16;
    using indices_engine_type = vec_u32x16;

    // we have sorting networks for 8
    const size_t N_REGISTERS_PER_LOOP = 8;

    //
    size_t n_levels = (params != nullptr) ? params->n_levels : (1 + (k + 1) / 3);
    if (n_levels == 0) {
        return true;
    }
    if (n_levels > k) {
        n_levels = k;
    }

#define DISPATCH_KERNEL(NX) \
        case NX:    \
            return kernel_getmink_fp32hack<distances_engine_type, indices_engine_type, NX, N_REGISTERS_PER_LOOP>(src_dis, n, k, dis, ids); 

    switch(n_levels) {
REPEATR_1D(DISPATCH_KERNEL, 1, 24)

        default:
            return false;
    }

#undef DISPATCH_KERNEL

    // done
    return false;
}

}  // namespace smalltopk

#include <smalltopk/utils/macro_repeat_undefine.h>
