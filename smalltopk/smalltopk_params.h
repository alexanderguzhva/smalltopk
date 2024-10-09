#pragma once

#include <stdint.h>

typedef struct {
    // Which kernel to use:
    // 0 - default. Can be overriden via 'SMALLTOPK_KERNEL' env variable.
    // 1 - fp32
    // 2 - fp16
    // 3 - fp32 hack
    // 4 - fp32 hack + Intel AMX
    // 5 - fp32 hack + 'fixed number of worthy candidates' approach
    uint32_t kernel;
    // Number of levels for tracing topk for approx kernels (such as kernel 5).
    //   Higher value, higher precision, less performance.
    // 0 for same as k
    // I'd use something like (1 + k / 2) or (1 + (k + 1) / 3) for AVX512 
    uint32_t n_levels;
} KnnL2sqrParameters;

typedef struct {
    // 0 - default. Can be overriden via 'SMALLTOPK_KERNEL' env variable.
    // 1 - fp32
    // 3 - fp32 hack
    uint32_t kernel;
    // Number of levels for tracing topk.
    //   Higher value, higher precision, less performance.
    // 0 for same as k
    // I'd use something like (1 + k / 2) or (1 + (k + 1) / 3) for AVX512 
    uint32_t n_levels;
} GetKParameters;
