#pragma once

#include <cstddef>
#include <cstdint>

#include "../utils/macro_repeat_define.h"


namespace smalltopk {


// transpose (NX_POINTS, DIM) into (DIM, NX_POINTS)
template<typename DistancesEngineT, size_t DIM>
//__attribute_noinline__
__attribute__((always_inline))
void transpose(
    const typename DistancesEngineT::scalar_type* const __restrict x,
    typename DistancesEngineT::scalar_type* const __restrict output
) {
    const auto dis_simd_width = DistancesEngineT::width();

    // cover the most typical cases
    if (dis_simd_width == 4) {
        constexpr size_t WIDTH = 4;

        for (size_t nx_k = 0; nx_k < WIDTH; nx_k++) {
            for (size_t dd = 0; dd < DIM; dd++) {
                output[dd * WIDTH + nx_k] = x[nx_k * DIM + dd];
            }
        }
    } else if (dis_simd_width == 8) {
        constexpr size_t WIDTH = 8;

        for (size_t nx_k = 0; nx_k < WIDTH; nx_k++) {
            for (size_t dd = 0; dd < DIM; dd++) {
                output[dd * WIDTH + nx_k] = x[nx_k * DIM + dd];
            }
        }
    } else if (dis_simd_width == 16) {
        constexpr size_t WIDTH = 16;

        for (size_t nx_k = 0; nx_k < WIDTH; nx_k++) {
            for (size_t dd = 0; dd < DIM; dd++) {
                output[dd * WIDTH + nx_k] = x[nx_k * DIM + dd];
            }
        }
    } else if (dis_simd_width == 32) {
        constexpr size_t WIDTH = 32;

        for (size_t nx_k = 0; nx_k < WIDTH; nx_k++) {
            for (size_t dd = 0; dd < DIM; dd++) {
                output[dd * WIDTH + nx_k] = x[nx_k * DIM + dd];
            }
        }
    } else {
        // a general-purpose case
        for (size_t nx_k = 0; nx_k < dis_simd_width; nx_k++) {
            for (size_t dd = 0; dd < DIM; dd++) {
                output[dd * dis_simd_width + nx_k] = x[nx_k * DIM + dd];
            }
        }
    }
}



// MAX NY_POINTS_PER_LOOP is 16

#define DECLARE_DP_PARAM(NX) typename DistancesEngineT::simd_type& __restrict dp_i_##NX,

template <
    typename DistancesEngineT,
    typename IndicesEngineT,
    size_t DIM,
    size_t NY_POINTS_PER_LOOP>
//__attribute_noinline__
__attribute__((always_inline))
void distances(
    const typename DistancesEngineT::scalar_type* const __restrict y,
    const size_t ny,
    const typename DistancesEngineT::scalar_type* const __restrict y_norms,
    const typename DistancesEngineT::scalar_type* __restrict x_transposed_values,
    const size_t j,
    // MAX_NY_POINTS_PER_LOOP
    REPEAT_1D(DECLARE_DP_PARAM, 16)
    const svbool_t dis_mask
) {
    const auto dis_simd_width = DistancesEngineT::width();

    using distances_type = typename DistancesEngineT::simd_type;
    using indices_type = typename IndicesEngineT::simd_type;

    using distance_type = typename DistancesEngineT::scalar_type;
    using index_type = typename IndicesEngineT::scalar_type;


    // perform dp = x[0] * y[0]
    // DIM 0 that uses MUL
    {
        auto mul = [&dis_mask, y, j, ny](
                const distances_type& x_transposed, 
                const size_t dd,
                const size_t ny_k
            ){
                // const auto* const y_ptr = y + (j + ny_k) * DIM;  
                // const distances_type yp = DistancesEngineT::set1(y_ptr[dd]);
                const auto* const y_ptr = y + ny * dd;  
                const distances_type yp = DistancesEngineT::set1(y_ptr[j + ny_k]);

                return DistancesEngineT::mul(dis_mask, x_transposed, yp);
            };


        // if constexpr (NY_POINTS_PER_LOOP >= 1 + 7) {
        //     distances_type x_transposed = DistancesEngineT::load(dis_mask, tmp[0]);
        //     dp_i_7 = mul(x_transposed, 0, 7);
        // }

#define PERFORM_MUL(NX) \
        if constexpr (NY_POINTS_PER_LOOP >= NX + 1) { \
            distances_type x_transposed = DistancesEngineT::load(dis_mask, x_transposed_values + 0); \
            dp_i_##NX = mul(x_transposed, 0, NX); \
        }

        // MAX_NY_POINTS_PER_LOOP
        REPEAT_1D(PERFORM_MUL, 16)

#undef PERFORM_MUL
    }


    // perform dp += x[1..] * y[1..]
    // other DIMs that use FMA
    {
        auto fmadd = [&dis_mask, y, j, ny](
                const distances_type& x_transposed, 
                const size_t dd,
                const size_t ny_k,
                const distances_type& accum
            ){
                // const auto* const y_ptr = y + (j + ny_k) * DIM;  
                // const distances_type yp = DistancesEngineT::set1(y_ptr[dd]);
                const auto* const y_ptr = y + ny * dd;  
                const distances_type yp = DistancesEngineT::set1(y_ptr[j + ny_k]);

                return DistancesEngineT::fmadd(dis_mask, x_transposed, yp, accum);
            };

        auto macro_fmadd = [&](
            const distances_type& x_transposed,
            const size_t dd
        ){
#define PERFORM_FMADD(NZ) \
            if constexpr (NY_POINTS_PER_LOOP >= NZ + 1) { \
                dp_i_##NZ = fmadd(x_transposed, dd, NZ, dp_i_##NZ); \
            }

            // MAX_NY_POINTS_PER_LOOP
            REPEAT_1D(PERFORM_FMADD, 16)
#undef PERFORM_FMADD
        };


        // the following loop has been unrolled in a regular,
        //   because macro unrolling Screws the performance of a generated code,
        //   because a C++ compiler introduces numerous register spills
        for (size_t p = 1; p < DIM; p++) {
            distances_type x_transposed = DistancesEngineT::load(dis_mask, x_transposed_values + p * dis_simd_width);
            macro_fmadd(x_transposed, p);
        }

// // #define PERFORM_MACRO_FMADD(NX) \
// //             if constexpr (DIM >= 1 + NX) { \
// //                 distances_type x_transposed = DistancesEngineT::load(dis_mask, x_transposed_values[NX]);    \
// //                 macro_fmadd(x_transposed, NX);  \
// //             }
// //
// //             REPEAT_P1_1D(PERFORM_MACRO_FMADD, 31)
// //
// // #undef PERFORM_MACRO_FMADD
    }


    // xy -> y^2 - 2xy
    {
        auto fnmadd = [&dis_mask, y, j, y_norms](
                const size_t ny_k,
                const distances_type& dp
            ) {
                const distances_type y_l2_sqr = DistancesEngineT::set1(y_norms[j + ny_k]);
                return DistancesEngineT::fnmadd(dis_mask, dp, DistancesEngineT::from_i32(2), y_l2_sqr);
            };

        //
#define PERFORM_FNMADD(NX)  \
        if constexpr (NY_POINTS_PER_LOOP >= NX + 1) { dp_i_##NX = fnmadd(NX, dp_i_##NX); }

        // MAX_NY_POINTS_PER_LOOP
        REPEAT_1D(PERFORM_FNMADD, 16)

#undef PERFORM_FNMADD
    }    
}

#undef DECLARE_DP_PARAM


template<size_t SORTING_K>
//__attribute_noinline__
__attribute__((always_inline))
void offload(
    const float* const __restrict output_d,
    const uint32_t* const __restrict output_i,
    float* const __restrict dis,
    int64_t* const __restrict ids,
    const uint64_t dis_simd_width
) {
    // transpose output results
    // cover the most typical cases
    if (dis_simd_width == 4) {
        constexpr size_t WIDTH = 4;

        for (size_t nx_k = 0; nx_k < WIDTH; nx_k++) {
            for (size_t i_k = 0; i_k < SORTING_K; i_k++) {
                dis[nx_k * SORTING_K + i_k] = output_d[nx_k + i_k * WIDTH];
                ids[nx_k * SORTING_K + i_k] = output_i[nx_k + i_k * WIDTH];
            }
        }
    } else if (dis_simd_width == 4) {
        constexpr size_t WIDTH = 4;

        for (size_t nx_k = 0; nx_k < WIDTH; nx_k++) {
            for (size_t i_k = 0; i_k < SORTING_K; i_k++) {
                dis[nx_k * SORTING_K + i_k] = output_d[nx_k + i_k * WIDTH];
                ids[nx_k * SORTING_K + i_k] = output_i[nx_k + i_k * WIDTH];
            }
        }
    } else if (dis_simd_width == 4) {
        constexpr size_t WIDTH = 4;

        for (size_t nx_k = 0; nx_k < WIDTH; nx_k++) {
            for (size_t i_k = 0; i_k < SORTING_K; i_k++) {
                dis[nx_k * SORTING_K + i_k] = output_d[nx_k + i_k * WIDTH];
                ids[nx_k * SORTING_K + i_k] = output_i[nx_k + i_k * WIDTH];
            }
        }
    } else if (dis_simd_width == 4) {
        constexpr size_t WIDTH = 4;

        for (size_t nx_k = 0; nx_k < WIDTH; nx_k++) {
            for (size_t i_k = 0; i_k < SORTING_K; i_k++) {
                dis[nx_k * SORTING_K + i_k] = output_d[nx_k + i_k * WIDTH];
                ids[nx_k * SORTING_K + i_k] = output_i[nx_k + i_k * WIDTH];
            }
        }
    } else {
        // a general-purpose case
        for (size_t nx_k = 0; nx_k < dis_simd_width; nx_k++) {
            for (size_t i_k = 0; i_k < SORTING_K; i_k++) {
                dis[nx_k * SORTING_K + i_k] = output_d[nx_k + i_k * dis_simd_width];
                ids[nx_k * SORTING_K + i_k] = output_i[nx_k + i_k * dis_simd_width];
            }
        }
    }
}

}  // namespace smalltopk

#include "../utils/macro_repeat_undefine.h"

