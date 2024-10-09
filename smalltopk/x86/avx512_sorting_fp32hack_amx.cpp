#include "avx512_sorting_fp32hack_amx.h"

#include <immintrin.h>
#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include "../utils/distances.h"
#include "../utils/norms.h"
#include "../utils/norms-inl.h"
#include "../utils/transpose.h"

#include "kernel_sorting_fp32hack_amx.h"

#include "avx512_vec_fp32.h"
#include "avx512_vec_fp16.h"

#include "fp16_norms.h"


namespace smalltopk {

//
bool knn_L2sqr_fp32_avx512_sorting_fp32hack_amx(
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

    // the current code works with only 1 AMX tile.
    if (d > 32) {
        return false;
    }

    //
    using distances_engine_type = vec_f32x16;
    using indices_engine_type = vec_u32x16;

    // This is just the number of reserve buffer. This is needed, because we'll process
    //   NY_POINTS_PER_TILE of y values per tile. 
    // If this values is changed, then it is needed to add more sorting network kernels.
    // DO NOT CHANGE THIS FOR NOW, bcz AMX TILE code depends on this value being 16.
    constexpr size_t NY_POINTS_PER_TILE = 16;
    // number of x points that we're processing per kernel
    constexpr auto NX_POINTS_PER_TILE = distances_engine_type::SIMD_WIDTH;

    static_assert(distances_engine_type::SIMD_WIDTH == indices_engine_type::SIMD_WIDTH);


    // compute norms for y.
    const size_t ny_16p = ((ny + NY_POINTS_PER_TILE - 1) / NY_POINTS_PER_TILE) * NY_POINTS_PER_TILE;
 
    std::unique_ptr<float[]> tmp_y_norms;
    const float* __restrict y_norms = y_norm_l2sqr;

    if (y_norms == nullptr || ny != ny_16p) {
        tmp_y_norms = 
            copy_or_compute_norms(y_in, y_norms, ny, d, ny_16p, std::numeric_limits<float>::max()); 

        y_norms = tmp_y_norms.get();
    }


    // //
    // auto y_transposed = std::make_unique<uint16_t[]>(32 * ny_16p);
    // for (size_t i = 0; i < ny; i += 16) {
    //     float buf[32][16];
    //     for (size_t j = 0; j < 32; j++) {
    //         for (size_t ii = 0; ii < 16; ii++) {
    //             buf[j][ii] = 0;
    //         }
    //     }
    //
    //     for (size_t ii = 0; ii < 16; ii++) {
    //         for (size_t j = 0; j < d; j++) {
    //             if (i + ii < ny) {
    //                 buf[j][ii] = y_in[j + (i + ii) * d];
    //             }
    //         }
    //     }
    //
    //     for (size_t ii = 0; ii < 16; ii++) {
    //         ConvertB(&(buf[0][0]) + ii * 32, 16, y_transposed.get() + (i + ii) * 32);
    //     }
    // }
    //
    // const uint16_t* __restrict y = y_transposed.get();


    // regular y. Prepare tiles.
    auto y_bf16 = std::make_unique<uint16_t[]>(32 * ny_16p);
    for (size_t i = 0; i < ny; i += 16) {
        float buf[16][32];
        for (size_t j = 0; j < 16; j++) {
            for (size_t ii = 0; ii < 32; ii++) {
                buf[j][ii] = 0;
            }
        }

        for (size_t ii = 0; ii < 16; ii++) {
            for (size_t j = 0; j < d; j++) {
                if (i + ii < ny) {
                    buf[ii][j] = y_in[j + (i + ii) * d];
                }
            }
        }

        for (size_t ii = 0; ii < 16; ii++) {
            convert_for_matrix_A(buf[ii], y_bf16.get() + (i + ii) * 32);
        }
    }

    const uint16_t* __restrict y = y_bf16.get();


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

        // set up AMX
        TileConfig conf = {};
        conf.paletteId = 1; 
        conf.rows[0] = 16; 
        conf.colsb[0] = 16 * 4; 
        // x
        conf.rows[1] = 16; 
        conf.colsb[1] = 16 * 4;
        // y
        conf.rows[2] = 16;
        conf.colsb[2] = 16 * 4;
        _tile_loadconfig(&conf);


        // // I'm leaving the following code just in case
        // //
        // // set up AMX for tile0 += tile2 * tile1 operation.
        // // d is [1, 32]
        // // colsb should be divisible by 4, this is why I'm using (d + 1) / 2
        // TileConfig conf = {};
        //
        // conf.paletteId = 1; 
        //
        // conf.rows[0] = 16; 
        // conf.colsb[0] = 16 * 4; 
        // // x
        // conf.rows[1] = (d + 1) / 2; 
        // conf.colsb[1] = 16 * 4;
        // // y
        // conf.rows[2] = 16;
        // conf.colsb[2] = (d + 1) / 2 * 4;
        // _tile_loadconfig(&conf);


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

            const bool success = kernel_sorting_fp32hack_amx_pre_k<distances_engine_type, indices_engine_type, NY_POINTS_PER_TILE>(
                x + idx_x_start * d,
                y,
                d,
                ny_16p,
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

        _tile_release();
    }

    // process leftovers
    if (nx_with_points != nx) {
        // set up AMX
        TileConfig conf = {};
        conf.paletteId = 1; 
        conf.rows[0] = 16; 
        conf.colsb[0] = 16 * 4; 
        conf.rows[1] = 16; 
        conf.colsb[1] = 16 * 4;
        conf.rows[2] = 16;
        conf.colsb[2] = 16 * 4;
        _tile_loadconfig(&conf);

        // we don't want to instantiate a separate kernel for a different NX_POINTS_PER_TILE value.
        // sure, it might require a biiiiiiiit more time, but it will Significantly
        //   decrease the compilation time and the binary size.

        // let's create a temporary buffer and process
        std::unique_ptr<float[]> tmp_x = std::make_unique<float[]>(NX_POINTS_PER_TILE * d);
        std::unique_ptr<float[]> tmp_dis = std::make_unique<float[]>(NX_POINTS_PER_TILE * k);
        std::unique_ptr<int64_t[]> tmp_ids = std::make_unique<int64_t[]>(NX_POINTS_PER_TILE * k);
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

        const bool success = kernel_sorting_fp32hack_amx_pre_k<distances_engine_type, indices_engine_type, NY_POINTS_PER_TILE>(
            tmp_x.get(),
            y,
            d,
            ny_16p,
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

        _tile_release();  
    }

    return true;
};

}  // namespace smalltopk
