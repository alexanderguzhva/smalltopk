#include <smalltopk/x86/avx512_sorting_fp32hack.h>

#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include <smalltopk/utils/norms.h>
#include <smalltopk/utils/norms-inl.h>
#include <smalltopk/utils/transpose.h>
#include <smalltopk/utils/transpose-inl.h>

#include <smalltopk/x86/kernel_sorting_fp32hack.h>
#include <smalltopk/x86/avx512_vec_fp32.h>

namespace smalltopk {

//
bool knn_L2sqr_fp32_avx512_sorting_fp32hack(
    const float* const __restrict x,
    const float* const __restrict y_in,
    const uint8_t d,
    const uint64_t nx,
    const uint64_t ny,
    const uint8_t k,
    const float* const __restrict x_norm_l2sqr,
    const float* const __restrict y_norm_l2sqr,
    float* const __restrict dis,
    smalltopk_knn_l2sqr_ids_type* const __restrict ids,
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
    using distances_engine_type = vec_f32x16;
    using indices_engine_type = vec_u32x16;

    // This is just the number of reserve buffer. This is needed, because we'll process
    //   NY_POINTS_PER_TILE of y values per tile. 
    // If this values is changed, then it is needed to add more sorting network kernels.
    constexpr size_t NY_POINTS_PER_TILE = 16;
    // number of x points that we're processing per kernel
    constexpr auto NX_POINTS_PER_TILE = distances_engine_type::SIMD_WIDTH;

    static_assert(distances_engine_type::SIMD_WIDTH == indices_engine_type::SIMD_WIDTH);


    // compute norms for y.
    const size_t ny_with_buffer = ((ny + NY_POINTS_PER_TILE - 1) / NY_POINTS_PER_TILE) * NY_POINTS_PER_TILE;
 
    std::unique_ptr<float[]> tmp_y_norms;
    const float* __restrict y_norms = y_norm_l2sqr;

    if (y_norms == nullptr || ny != ny_with_buffer) {
        tmp_y_norms = 
            copy_or_compute_norms(y_in, y_norms, ny, d, ny_with_buffer, std::numeric_limits<float>::max()); 

        y_norms = tmp_y_norms.get();
    }


    // // normal y, which is (ny, d)
    // std::unique_ptr<float[]> tmp_y;
    // const float* __restrict y = y_in;
    //
    // if (ny != ny_with_buffer) {
    //     tmp_y = std::make_unique<float[]>(ny_with_buffer * d);
    //
    //     for (size_t i = 0; i < ny * d; i++) {
    //         tmp_y[i] = y_in[i];
    //     }
    //
    //     // 0 is by design; an overflow comes from y norms
    //     for (size_t i = ny * d; i < ny_with_buffer * d; i++) {
    //         tmp_y[i] = 0;
    //     }
    //
    //     y = tmp_y.get();
    // }


    // transpose y into (d, ny)
    std::unique_ptr<float[]> tmp_y_transposed = 
        transpose_and_fill<float>(y_in, ny, d, ny_with_buffer, 0.0f);
    const float* __restrict y = tmp_y_transposed.get();


    // the main loop.
    //
    // most likely, this function will be called multiple times.
    // so, we'd like to make sure that the same input data hits
    //   the same kernels in order to help CPU caches.

    // number of tiles of NX_POINTS_PER_TILE size that fits into nx
    const size_t nx_tiles = nx / NX_POINTS_PER_TILE;
    // number of points to be processed in parallel
    const size_t nx_with_points = nx_tiles * NX_POINTS_PER_TILE; 

#pragma omp parallel
    {
        const int rank = omp_get_thread_num();
        const int nt = omp_get_num_threads();

        const size_t c0 = (nx_tiles * rank) / nt;
        const size_t c1 = (nx_tiles * (rank + 1)) / nt;

        // allocate a temporary buffer for x_norms
        std::unique_ptr<float[]> tmp_x_norms= std::make_unique<float[]>(NX_POINTS_PER_TILE);

        for (size_t i = c0; i < c1; i++) {
            const size_t idx_x_start = i * NX_POINTS_PER_TILE;
            const size_t idx_x_end = (i + 1) * NX_POINTS_PER_TILE;

            // set up norms
            if (x_norm_l2sqr != nullptr) {
                // copy
                for (size_t j = idx_x_start; j < idx_x_end; j++) {
                    tmp_x_norms[j - idx_x_start] = x_norm_l2sqr[j];
                }
            } else {
                // compute
                compute_norms_inline(x + idx_x_start * d, NX_POINTS_PER_TILE, d, tmp_x_norms.get());
            }

            const bool success = kernel_sorting_fp32hack_pre_k<distances_engine_type, indices_engine_type, NY_POINTS_PER_TILE, smalltopk_knn_l2sqr_ids_type>(
                x + idx_x_start * d,
                y,
                d,
                ny_with_buffer,
                k,
                tmp_x_norms.get(),
                y_norms,
                (dis == nullptr) ? nullptr : (dis + idx_x_start * k),
                (ids == nullptr) ? nullptr : (ids + idx_x_start * k)
            );

            if (!success) {
                break;
            }
        }
    }

    // process leftovers
    if (nx_with_points != nx) {
        // we don't want to instantiate a separate kernel for a different NX_POINTS_PER_TILE value.
        // sure, it might require a biiiiiiiit more time, but it will Significantly
        //   decrease the compilation time and the binary size.

        // let's create a temporary buffer and process
        std::unique_ptr<float[]> tmp_x = std::make_unique<float[]>(NX_POINTS_PER_TILE * d);
        std::unique_ptr<float[]> tmp_dis = std::make_unique<float[]>(NX_POINTS_PER_TILE * k);
        std::unique_ptr<smalltopk_knn_l2sqr_ids_type[]> tmp_ids = 
            std::make_unique<smalltopk_knn_l2sqr_ids_type[]>(NX_POINTS_PER_TILE * k);
        std::unique_ptr<float[]> tmp_x_norms = std::make_unique<float[]>(NX_POINTS_PER_TILE);

        // populate tmp_x
        for (size_t i = nx_with_points; i < nx; i++) {
            for (size_t dd = 0; dd < d; dd++) {
                tmp_x[(i - nx_with_points) * d + dd] = x[i * d + dd];
            }
        }

        if (x_norm_l2sqr != nullptr) {
            for (size_t i = nx_with_points; i < nx; i++) {
                tmp_x_norms[i - nx_with_points] = x_norm_l2sqr[i];
            }
        } else {
            compute_norms_inline(tmp_x.get(), NX_POINTS_PER_TILE, d, tmp_x_norms.get());
        }

        const bool success = kernel_sorting_fp32hack_pre_k<distances_engine_type, indices_engine_type, NY_POINTS_PER_TILE, smalltopk_knn_l2sqr_ids_type>(
            tmp_x.get(),
            y,
            d,
            ny_with_buffer,
            k,
            tmp_x_norms.get(),
            y_norms,
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
