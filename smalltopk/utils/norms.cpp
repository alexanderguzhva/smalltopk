#include <smalltopk/utils/norms.h>

#include <cstdint>
#include <cstddef>

#include <smalltopk/utils/distances.h>

namespace smalltopk {

void compute_norms(
    const float* const __restrict x,
    const size_t nx,
    const size_t dim,
    float* const __restrict x_norm_i
) {
    for (size_t nx_k = 0; nx_k < nx; nx_k++) {
        // x address
        const float* const x_ptr = x + nx_k * dim;  

        // norms
        x_norm_i[nx_k] = l2_sqr(x_ptr, dim);
    }
}


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
) {
    std::unique_ptr<float[]> result = std::make_unique<float[]>(nnx);

    if (x_norms == nullptr) {
        // manually compute norms
        compute_norms(x_in, nx, dim, result.get());
    } else {
        // copy norms 
        for (size_t i = 0; i < nx; i++) {
            result[i] = x_norms[i];
        }
    }

    // fill leftovers with infinity
    for (size_t i = nx; i < nnx; i++) {
        result[i] = default_value;
    }

    // done
    return result;
}

}  // namespace smalltopk
