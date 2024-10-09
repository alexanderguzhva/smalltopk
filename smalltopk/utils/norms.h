#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>

namespace smalltopk {

void compute_norms(
    const float* const __restrict x,
    const size_t nx,
    const size_t dim,
    float* const __restrict x_norm_i
);

// we need (nnx) norms, either computed over x_in (nx, dim),
//   or copied from externally provided x_norms (nx).
// if (nnx > nx), then missing parts will be initialized with default_value.
std::unique_ptr<float[]> copy_or_compute_norms(
    const float* const __restrict x_in,
    const float* const __restrict x_norms,
    const size_t nx,
    const size_t dim,
    const size_t nnx,
    const float default_value
);

}  // namespace smalltopk
