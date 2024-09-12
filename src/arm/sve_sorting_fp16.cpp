#include "sve_sorting_fp16.h"

#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include "../utils/distances.h"
#include "../utils/norms.h"
#include "../utils/norms-inl.h"
#include "../utils/transpose.h"
#include "../utils/transpose-inl.h"

#include "kernel_sorting.h"
#include "sve_vec.h"

namespace smalltopk {

namespace {

template<size_t DIM>
void compute_norms_fp16(
    const float* const __restrict x,
    const size_t nx,
    float16_t* const __restrict x_norm_i
) {
    for (size_t nx_k = 0; nx_k < nx; nx_k++) {
        // x address
        const float* const x_ptr = x + nx_k * DIM;  

        // norms
        x_norm_i[nx_k] = float16_t(l2_sqr<DIM>(x_ptr));
    }
}

static inline void compute_norms_inline_fp16(
    const float* const __restrict x,
    const size_t nx,
    const size_t dim,
    float16_t* const __restrict x_norm_i
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

        x_norm_i[nx_k] = float16_t(sum);
    }

#undef DISPATCH
}

}

//
bool knn_L2sqr_fp32_sve_sorting_fp16(
    const float* const __restrict x,
    const float* const __restrict y_in,
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
    // nothing to do?
    if (nx == 0 || ny == 0) {
        return true;
    }

    // not supported?
    if (ny > 65536) {
        // todo: copy-paste a version of this kernel that has int32_t ny counter.
        return false;
    }

    //
    using distances_engine_type = vec_f16;
    using indices_engine_type = vec_u16;

    // This is just the number of reserve buffer. This is needed. 
    constexpr size_t NY_POINTS_PER_TILE = 8;
    // number of x points that we're processing per kernel
    const auto nx_points_per_tile = distances_engine_type::width();


    // compute norms for y.
    const size_t ny_with_buffer = ((ny + NY_POINTS_PER_TILE - 1) / NY_POINTS_PER_TILE) * NY_POINTS_PER_TILE;
 
    // always create norms
    std::unique_ptr<float16_t[]> y_norms = std::make_unique<float16_t[]>(ny_with_buffer);

    if (y_norm_l2sqr == nullptr) {
        // manually compute norms
        for (size_t i = 0; i < ny; i++) {
            y_norms[i] = float16_t(l2_sqr(y_in + i * d, d));
        }
    } else {
        // copy norms 
        for (size_t i = 0; i < ny; i++) {
            y_norms[i] = float16_t(y_norm_l2sqr[i]);
        }
    }

    // fill leftovers with infinity
    for (size_t i = ny; i < ny_with_buffer; i++) {
        y_norms[i] = float16_t(std::numeric_limits<float>::max());
    } 

    //
    // transpose y into (d, ny)
    std::unique_ptr<float16_t[]> tmp_y_transposed = 
        transpose_and_fill<float16_t, float>(y_in, ny, d, ny_with_buffer, 0.0f);
    
    const float16_t* __restrict y = tmp_y_transposed.get();


    // the main loop.
    //
    // most likely, this function will be called multiple times.
    // so, we'd like to make sure that the same input data hits
    //   the same kernels in order to help CPU caches.

    // number of tiles of nx_points_per_tile size that fits into nx
    const size_t nx_tiles = nx / nx_points_per_tile;
    // number of points to be processed in parallel
    const size_t nx_with_points = nx_tiles * nx_points_per_tile; 

#pragma omp parallel
    {
        const int rank = omp_get_thread_num();
        const int nt = omp_get_num_threads();

        const size_t c0 = (nx_tiles * rank) / nt;
        const size_t c1 = (nx_tiles * (rank + 1)) / nt;

        // allocate a temporary buffer for x_norms
        std::unique_ptr<float16_t[]> tmp_x = std::make_unique<float16_t[]>(nx_points_per_tile * d);
        std::unique_ptr<float16_t[]> tmp_x_norms = std::make_unique<float16_t[]>(nx_points_per_tile);

        for (size_t i = c0; i < c1; i++) {
            const size_t idx_x_start = i * nx_points_per_tile;
            const size_t idx_x_end = (i + 1) * nx_points_per_tile;


            for (size_t j = idx_x_start; j < idx_x_end; j++) {
                for (size_t dd = 0; dd < d; dd++) {
                    tmp_x[(j - idx_x_start) * d + dd] = float16_t(x[j * d + dd]);
                }
            }

            // set up norms
            if (x_norm_l2sqr != nullptr) {
                // copy
                for (size_t j = idx_x_start; j < idx_x_end; j++) {
                    tmp_x_norms[j - idx_x_start] = float16_t(x_norm_l2sqr[j]);
                }
            } else {
                // compute
                compute_norms_inline_fp16(x + idx_x_start * d, nx_points_per_tile, d, tmp_x_norms.get());
            }

            const bool success = kernel_sorting_pre_k<distances_engine_type, indices_engine_type, NY_POINTS_PER_TILE>(
                tmp_x.get(),
                y,
                d,
                ny_with_buffer,
                k,
                tmp_x_norms.get(),
                y_norms.get(),
                (dis == nullptr) ? nullptr : (dis + idx_x_start * k),
                (ids == nullptr) ? nullptr : (ids + idx_x_start * k)
            );
        }
    }

    // process leftovers
    if (nx_with_points != nx) {
        // we don't want to instantiate a separate kernel for a different nx_points_per_tile value.
        // sure, it might require a biiiiiiiit more time, but it will Significantly
        //   decrease the compilation time and the binary size.

        // let's create a temporary buffer and process
        std::unique_ptr<float16_t[]> tmp_x = std::make_unique<float16_t[]>(nx_points_per_tile * d);
        std::unique_ptr<float[]> tmp_dis = std::make_unique<float[]>(nx_points_per_tile * k);
        std::unique_ptr<int64_t[]> tmp_ids = std::make_unique<int64_t[]>(nx_points_per_tile * k);
        std::unique_ptr<float16_t[]> tmp_x_norms = std::make_unique<float16_t[]>(nx_points_per_tile);

        // populate tmp_x
        for (size_t i = nx_with_points; i < nx; i++) {
            for (size_t dd = 0; dd < d; dd++) {
                tmp_x[(i - nx_with_points) * d + dd] = float16_t(x[i * d + dd]);
            }
        }

        if (x_norm_l2sqr != nullptr) {
            for (size_t i = nx_with_points; i < nx; i++) {
                tmp_x_norms[i - nx_with_points] = float16_t(x_norm_l2sqr[i]);
            }
        } else {
            for (size_t i = nx_with_points; i < nx; i++) {
                tmp_x_norms[i - nx_with_points] = float16_t(l2_sqr(x + i * d, d));
            }
        }

        const bool success = kernel_sorting_pre_k<distances_engine_type, indices_engine_type, NY_POINTS_PER_TILE>(
            tmp_x.get(),
            y,
            d,
            ny_with_buffer,
            k,
            tmp_x_norms.get(),
            y_norms.get(),
            tmp_dis.get(),
            tmp_ids.get()
        );

        // copy back dis and ids
        for (size_t i = nx_with_points; i < nx; i++) {
            for (size_t j = 0; j < k; j++) {
                if (ids != nullptr) {
                    ids[i * k + j] = tmp_ids[(i - nx_with_points) * k + j];
                }
                
                if (dis != nullptr) {
                    dis[i * k + j] = tmp_dis[(i - nx_with_points) * k + j];
                }
            }
        }        
    }

    return true;
};

}  // namespace smalltopk



