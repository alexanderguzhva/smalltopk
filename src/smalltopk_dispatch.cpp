#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

extern "C" {
#include "smalltopk_params.h"
#include "smalltopk.h"
}

#include "utils/env.h"

#include "dummy.h"

#ifdef __x86_64__
#include "x86/x86_instruction_set.h"
#include "x86/amx_init.h"

#include "x86/avx512_sorting_fp32.h"
#include "x86/avx512_sorting_fp16.h"
#include "x86/avx512_sorting_fp32hack.h"
#include "x86/avx512_sorting_fp32hack_amx.h"
#include "x86/avx512_sorting_fp32hack_approx.h"

#include "x86/avx512_getmink_fp32.h"
#endif

#ifdef __aarch64__
#include "arm/arm_instruction_set.h"
#include "arm/sve_sorting_fp32.h"
#include "arm/sve_sorting_fp16.h"
#include "arm/sve_sorting_fp32hack.h"
#include "arm/sve_sorting_fp32hack_approx.h"
#endif

namespace smalltopk {

//
using knn_l2sqr_fp32_handler_type = bool(*)(
    const float* const __restrict x,
    const float* const __restrict y,
    const uint8_t d,
    const uint64_t nx,
    const uint64_t ny,
    const uint8_t k,
    const float* const __restrict x_norm_l2sqr,
    const float* const __restrict y_norm_l2sqr,
    float* const __restrict dis,
    int64_t* const __restrict ids,
    const KnnL2sqrParameters* const __restrict params
);

knn_l2sqr_fp32_handler_type current_knn_l2sqr_fp32_hook = knn_L2sqr_fp32_dummy;

//
using get_k_fp32_handler_type = bool(*)(
    const float* const __restrict src_dis,
    const uint32_t n,
    const uint8_t k,
    float* const __restrict dis,
    int32_t* const __restrict ids,
    const GetKParameters* const __restrict params
);

get_k_fp32_handler_type current_get_min_k_fp32_hook = get_min_k_fp32_dummy;

//
int32_t verbosity = 0;

//
#ifdef __x86_64__
static void init_hook_x86() {
    std::string env_kernel = get_env("SMALLTOPK_KERNEL").value_or("");
    if (env_kernel == "none" || env_kernel == "disabled" || env_kernel == "off") {
        // disabled
        if (verbosity > 0) {
            printf("smalltopk is disabled\n");
        }

        return;
    }

    init_amx();

    const bool is_avx512_fp32_supported = 
        InstructionSet::get_instance().is_avx512f_supported && 
        InstructionSet::get_instance().is_avx512cd_supported && 
        InstructionSet::get_instance().is_avx512bw_supported && 
        InstructionSet::get_instance().is_avx512dq_supported && 
        InstructionSet::get_instance().is_avx512vl_supported;

    if (env_kernel == "fp16" || env_kernel == "2") {
        current_knn_l2sqr_fp32_hook = knn_L2sqr_fp32_avx512_sorting_fp16;

        if (verbosity > 0) {
            printf("smalltopk uses knn_L2sqr_fp32_avx512_sorting_fp16 kernel as a default one\n");
        }

        return;
    }

    if (env_kernel == "fp32hack" || env_kernel == "hack" || env_kernel == "3") {
        current_knn_l2sqr_fp32_hook = knn_L2sqr_fp32_avx512_sorting_fp32hack;

        if (verbosity > 0) {
            printf("smalltopk uses knn_L2sqr_fp32_avx512_sorting_fp32hack kernel as a default one\n");
        }

        return;
    }

    if (env_kernel == "fp32hackamx" || env_kernel == "hackamx" || 
        env_kernel == "fp32hack_amx" || env_kernel == "hack_amx" || 
        env_kernel == "4") {
        current_knn_l2sqr_fp32_hook = knn_L2sqr_fp32_avx512_sorting_fp32hack_amx;

        if (verbosity > 0) {
            printf("smalltopk uses knn_L2sqr_fp32_avx512_sorting_fp32hack_amx kernel as a default one\n");
        }

        return;
    }

    if (env_kernel == "fp32hack_approx" || env_kernel == "hack_approx" || env_kernel == "5") {
        current_knn_l2sqr_fp32_hook = knn_L2sqr_fp32_avx512_sorting_fp32hack_approx;

        if (verbosity > 0) {
            printf("smalltopk uses knn_L2sqr_fp32_avx512_sorting_fp32hack_approx kernel as a default one\n");
        }

        return;
    }

    if (is_avx512_fp32_supported || (env_kernel == "fp32" || env_kernel == "1")) {
        current_knn_l2sqr_fp32_hook = knn_L2sqr_fp32_avx512_sorting_fp32;
        current_get_min_k_fp32_hook = get_min_k_fp32_avx512;

        if (verbosity > 0) {
            printf("smalltopk uses knn_L2sqr_fp32_avx512_sorting_fp32 kernel as a default one\n");
        }

        return;
    }
}
#endif

#ifdef __aarch64__
static void init_hook_aarch64() {
    std::string env_kernel = get_env("SMALLTOPK_KERNEL").value_or("");
    if (env_kernel == "none" || env_kernel == "disabled" || env_kernel == "off") {
        // disabled
        if (verbosity > 0) {
            printf("smalltopk is disabled\n");
        }

        return;
    }

    if (InstructionSet::get_instance().is_sve_supported) {
        if (env_kernel == "fp16" || env_kernel == "2") {
            if (verbosity > 0) {
                printf("smalltopk uses knn_L2sqr_fp32_sve_sorting_fp16 kernel as a default one\n");
            }

            current_knn_l2sqr_fp32_hook = knn_L2sqr_fp32_sve_sorting_fp16;
        } else if (env_kernel == "fp32hack" || env_kernel == "hack" || env_kernel == "3") {
            if (verbosity > 0) {
                printf("smalltopk uses knn_L2sqr_fp32_sve_sorting_fp32hack kernel as a default one\n");
            }

            current_knn_l2sqr_fp32_hook = knn_L2sqr_fp32_sve_sorting_fp32hack;
        } else if (env_kernel == "fp32hack_approx" || env_kernel == "hack_approx" || env_kernel == "5") {
            if (verbosity > 0) {
                printf("smalltopk uses knn_L2sqr_fp32_sve_sorting_fp32hack_approx kernel as a default one\n");
            }

            current_knn_l2sqr_fp32_hook = knn_L2sqr_fp32_sve_sorting_fp32hack_approx;
        } else if (env_kernel == "fp32" || env_kernel == "1") {
            if (verbosity > 0) {
                printf("smalltopk uses knn_L2sqr_fp32_sve_sorting_fp32 kernel as a default one\n");
            }

            current_knn_l2sqr_fp32_hook = knn_L2sqr_fp32_sve_sorting_fp32;
        } else {
            if (verbosity > 0) {
                printf("smalltopk uses knn_L2sqr_fp32_sve_sorting_fp32 kernel as a default one\n");
            }

            current_knn_l2sqr_fp32_hook = knn_L2sqr_fp32_sve_sorting_fp32;
        }
    } else {
        if (verbosity > 0) {
            printf("smalltopk is disabled, because ARM SVE seems to be disabled\n");
        }
    }
}
#endif

//
static void init_hook() {
    current_knn_l2sqr_fp32_hook = knn_L2sqr_fp32_dummy;
    current_get_min_k_fp32_hook = get_min_k_fp32_dummy;

    //
    std::string env_verbose = get_env("SMALLTOPK_VERBOSE").value_or("");
    if (env_verbose == "1" || env_verbose == "yes" || env_verbose == "true" || env_verbose == "info") {
        verbosity = 1;
    }

    if (env_verbose == "2" || env_verbose == "debug") {
        verbosity = 2;
    }

    if (verbosity > 0) {
        printf("smalltopk verbosity level is %d\n", verbosity);
    }

#ifdef __x86_64__
    init_hook_x86();
#endif

#ifdef __aarch64__
    init_hook_aarch64();
#endif
}

}  // namespace smalltopk

//
bool knn_L2sqr_fp32(
    const float* const __restrict x,
    const float* const __restrict y,
    const uint8_t d,
    const uint64_t nx,
    const uint64_t ny,
    const uint8_t k,
    const float* const __restrict x_norm_l2sqr,
    const float* const __restrict y_norm_l2sqr,
    float* const __restrict dis,
    int64_t* const __restrict ids,
    const KnnL2sqrParameters* const __restrict params
) {
    if (smalltopk::verbosity == 2) {
        printf("smalltopk running knn_L2sqr_fp32, d=%" PRIu64 
            ", nx=%" PRIu64 ", ny=%" PRIu64 ", k=%" PRIu64
            "\n",
            uint64_t(d),
            uint64_t(nx),
            uint64_t(ny),
            uint64_t(k));
    }

#ifdef __aarch64__
    // a default kernel
    if (params == nullptr) {
        return smalltopk::current_knn_l2sqr_fp32_hook(x, y, d, nx, ny, k, x_norm_l2sqr, y_norm_l2sqr, dis, ids, params);
    }

    switch (params->kernel) {
        case 1:
            return smalltopk::knn_L2sqr_fp32_sve_sorting_fp32(x, y, d, nx, ny, k, x_norm_l2sqr, y_norm_l2sqr, dis, ids, params);
        case 2:
            return smalltopk::knn_L2sqr_fp32_sve_sorting_fp16(x, y, d, nx, ny, k, x_norm_l2sqr, y_norm_l2sqr, dis, ids, params);
        case 3:
            return smalltopk::knn_L2sqr_fp32_sve_sorting_fp32hack(x, y, d, nx, ny, k, x_norm_l2sqr, y_norm_l2sqr, dis, ids, params);
        case 4:
            // no AMX on SVE
            return false;
        case 5:
            return smalltopk::knn_L2sqr_fp32_sve_sorting_fp32hack_approx(x, y, d, nx, ny, k, x_norm_l2sqr, y_norm_l2sqr, dis, ids, params);
        case 0:
        default:
            return smalltopk::current_knn_l2sqr_fp32_hook(x, y, d, nx, ny, k, x_norm_l2sqr, y_norm_l2sqr, dis, ids, params);
    }

    return false;
#endif

#ifdef __x86_64__
    // a default kernel
    if (params == nullptr) {
        return smalltopk::current_knn_l2sqr_fp32_hook(x, y, d, nx, ny, k, x_norm_l2sqr, y_norm_l2sqr, dis, ids, params);
    }

    switch (params->kernel) {
        case 1:
            return smalltopk::knn_L2sqr_fp32_avx512_sorting_fp32(x, y, d, nx, ny, k, x_norm_l2sqr, y_norm_l2sqr, dis, ids, params);
        case 2:
            return smalltopk::knn_L2sqr_fp32_avx512_sorting_fp16(x, y, d, nx, ny, k, x_norm_l2sqr, y_norm_l2sqr, dis, ids, params);
        case 3:
            return smalltopk::knn_L2sqr_fp32_avx512_sorting_fp32hack(x, y, d, nx, ny, k, x_norm_l2sqr, y_norm_l2sqr, dis, ids, params);
        case 4:
            return smalltopk::knn_L2sqr_fp32_avx512_sorting_fp32hack_amx(x, y, d, nx, ny, k, x_norm_l2sqr, y_norm_l2sqr, dis, ids, params);
        case 5:
            return smalltopk::knn_L2sqr_fp32_avx512_sorting_fp32hack_approx(x, y, d, nx, ny, k, x_norm_l2sqr, y_norm_l2sqr, dis, ids, params);
        case 0:
        default:
            return smalltopk::current_knn_l2sqr_fp32_hook(x, y, d, nx, ny, k, x_norm_l2sqr, y_norm_l2sqr, dis, ids, params);
    }

    return false;
#endif
}

// finds k elements with min distances
bool get_min_k_fp32(
    const float* const __restrict src_dis,
    const uint32_t n,
    const uint8_t k,
    float* const __restrict dis,
    int32_t* const __restrict ids,
    const GetKParameters* const __restrict params
) {
    return smalltopk::current_get_min_k_fp32_hook(src_dis, n, k, dis, ids, params);
}

// init hook
struct HookInit {
    HookInit() { 
        smalltopk::init_hook(); 
    }
};

HookInit hook_init;
