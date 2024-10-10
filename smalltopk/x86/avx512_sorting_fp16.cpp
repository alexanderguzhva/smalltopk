#include <smalltopk/x86/avx512_sorting_fp16.h>

#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include <smalltopk/utils/distances.h>
#include <smalltopk/utils/norms.h>
#include <smalltopk/utils/norms-inl.h>
#include <smalltopk/utils/transpose.h>
#include <smalltopk/utils/transpose-inl.h>

#include <smalltopk/x86/avx512_vec_fp16.h>
#include <smalltopk/x86/kernel_sorting.h>

#include <smalltopk/x86/fp16_norms.h>

namespace smalltopk {

//
bool knn_L2sqr_fp32_avx512_sorting_fp16(
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
    using distances_engine_type = vec_f16x32;
    using indices_engine_type = vec_u16x32;

    // This is just the number of reserve buffer. This is needed, because we'll process
    //   NY_POINTS_PER_TILE of y values per tile. 
    constexpr size_t NY_POINTS_PER_TILE = 16;
    // number of x points that we're processing per kernel
    const auto NX_POINTS_PER_TILE = distances_engine_type::SIMD_WIDTH;


    // compute norms for y.
    const size_t ny_with_buffer = ((ny + NY_POINTS_PER_TILE - 1) / NY_POINTS_PER_TILE) * NY_POINTS_PER_TILE;
 
    // always create norms
    std::unique_ptr<uint16_t[]> y_norms = std::make_unique<uint16_t[]>(ny_with_buffer);

    if (y_norm_l2sqr == nullptr) {
        // manually compute norms
        for (size_t i = 0; i < ny; i++) {
            y_norms[i] = fp32_to_fp16(l2_sqr(y_in + i * d, d));
        }
    } else {
        // copy norms 
        fp32_to_fp16(y_norm_l2sqr, y_norms.get(), ny);
    }

    // fill leftovers with infinity
    for (size_t i = ny; i < ny_with_buffer; i++) {
        y_norms[i] = fp32_to_fp16(std::numeric_limits<float>::max());
    } 


    // // normal y, which is (ny, d). Just convert it into fp16
    // std::unique_ptr<uint16_t[]> tmp_y;
    //
    // {
    //     tmp_y = std::make_unique<uint16_t[]>(ny_with_buffer * d);
    //
    //     fp32_to_fp16(y_in, tmp_y.get(), ny * d);
    //
    //     // 0 is by design; an overflow comes from y norms
    //     for (size_t i = ny * d; i < ny_with_buffer * d; i++) {
    //         tmp_y[i] = fp32_to_fp16(0);
    //     }
    // }
    //
    // const uint16_t* __restrict y_fp16 = tmp_y.get();


    // transpose y into (d, ny)
    std::unique_ptr<uint16_t[]> tmp_y_transposed;

    {
        std::unique_ptr<uint16_t[]> tmp_y = std::make_unique<uint16_t[]>(ny_with_buffer * d);
        fp32_to_fp16(y_in, tmp_y.get(), ny * d);

        tmp_y_transposed = transpose_and_fill<uint16_t>(tmp_y.get(), ny, d, ny_with_buffer, 0);
    }

    const uint16_t* __restrict y_fp16 = tmp_y_transposed.get();


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
        std::unique_ptr<uint16_t[]> tmp_x = std::make_unique<uint16_t[]>(NX_POINTS_PER_TILE * d);
        std::unique_ptr<uint16_t[]> tmp_x_norms = std::make_unique<uint16_t[]>(NX_POINTS_PER_TILE);

        for (size_t i = c0; i < c1; i++) {
            const size_t idx_x_start = i * NX_POINTS_PER_TILE;
            const size_t idx_x_end = (i + 1) * NX_POINTS_PER_TILE;

            // populate tmp_x
            fp32_to_fp16(x + idx_x_start * d, tmp_x.get(), (idx_x_end - idx_x_start) * d);

            // set up norms
            if (x_norm_l2sqr != nullptr) {
                // copy
                fp32_to_fp16(x_norm_l2sqr + idx_x_start, tmp_x_norms.get(), (idx_x_end - idx_x_start));
            } else {
                // compute
                compute_norms_inline_fp16(x + idx_x_start * d, NX_POINTS_PER_TILE, d, tmp_x_norms.get());
            }

            const bool success = kernel_sorting_pre_k<distances_engine_type, indices_engine_type, NY_POINTS_PER_TILE, smalltopk_knn_l2sqr_ids_type>(
                tmp_x.get(),
                y_fp16,
                d,
                ny_with_buffer,
                k,
                tmp_x_norms.get(),
                y_norms.get(),
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
        std::unique_ptr<uint16_t[]> tmp_x = std::make_unique<uint16_t[]>(NX_POINTS_PER_TILE * d);
        std::unique_ptr<float[]> tmp_dis = std::make_unique<float[]>(NX_POINTS_PER_TILE * k);
        std::unique_ptr<smalltopk_knn_l2sqr_ids_type[]> tmp_ids = 
            std::make_unique<smalltopk_knn_l2sqr_ids_type[]>(NX_POINTS_PER_TILE * k);
        std::unique_ptr<uint16_t[]> tmp_x_norms = std::make_unique<uint16_t[]>(NX_POINTS_PER_TILE);

        // populate tmp_x
        fp32_to_fp16(x + nx_with_points * d, tmp_x.get(), (nx - nx_with_points) * d);

        if (x_norm_l2sqr != nullptr) {
            fp32_to_fp16(x_norm_l2sqr + nx_with_points, tmp_x_norms.get(), (nx - nx_with_points));
        } else {
            compute_norms_inline_fp16(x + nx_with_points * d, NX_POINTS_PER_TILE, d, tmp_x_norms.get());
        }

        const bool success = kernel_sorting_pre_k<distances_engine_type, indices_engine_type, NY_POINTS_PER_TILE, smalltopk_knn_l2sqr_ids_type>(
            tmp_x.get(),
            y_fp16,
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
