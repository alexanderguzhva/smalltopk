#pragma once

#include <cstddef>
#include <cstdint>

namespace smalltopk {

// transpose (NX_POINTS, DIM) into (DIM, NX_POINTS)
template<typename DistancesEngineT, size_t NX_POINTS, size_t DIM>
//__attribute_noinline__
__attribute__((always_inline))
void transpose(
    const typename DistancesEngineT::scalar_type* const __restrict x,
    typename DistancesEngineT::scalar_type* const __restrict output
) {
    for (size_t nx_k = 0; nx_k < NX_POINTS; nx_k++) {
        for (size_t dd = 0; dd < DIM; dd++) {
            output[dd * NX_POINTS + nx_k] = x[nx_k * DIM + dd];
        }
    }
}

// compute a set of y^2 - 2xy values
template <
    typename DistancesEngineT,
    typename IndicesEngineT,
    size_t DIM,
    size_t NX_POINTS,
    size_t NY_POINTS_PER_LOOP>
//__attribute_noinline__
__attribute__((always_inline))
void distances(
    const typename DistancesEngineT::scalar_type* const __restrict y_transposed,
    const size_t ny,
    const typename DistancesEngineT::scalar_type* const __restrict y_norms,
    const typename DistancesEngineT::scalar_type* __restrict x_transposed,
    const size_t j,
    typename DistancesEngineT::simd_type* __restrict dp_i
) {
    using distances_type = typename DistancesEngineT::simd_type;

    // perform dp = x[0] * y[0]
    // DIM 0 that uses MUL
    {
        const distances_type x_i = DistancesEngineT::load(x_transposed + 0 * NX_POINTS);

        for (size_t ny_k = 0; ny_k < NY_POINTS_PER_LOOP; ny_k++) {
            // // regular y
            // const auto* const y_ptr = y + (j + ny_k) * DIM;  
            // const distances_type yp = DistancesEngineT::set1(y_ptr[0]);
            
            // transposed y
            const auto* const y_ptr = y_transposed + 0 * ny;
            const distances_type yp = DistancesEngineT::set1(y_ptr[j + ny_k]);

            dp_i[ny_k] = DistancesEngineT::mul(x_i, yp);
        }
    }

    // perform dp += x[1..] * y[1..]
    // other DIMs that use FMA
    for (size_t dd = 1; dd < DIM; dd++) {
        const distances_type x_i = DistancesEngineT::load(x_transposed + dd * NX_POINTS);

        for (size_t ny_k = 0; ny_k < NY_POINTS_PER_LOOP; ny_k++) {
            // // regular y
            // const auto* const y_ptr = y + (j + ny_k) * DIM;  
            // const distances_type yp = DistancesEngineT::set1(y_ptr[dd]);

            // transposed y
            const auto* const y_ptr = y_transposed + ny * dd;
            const distances_type yp = DistancesEngineT::set1(y_ptr[j + ny_k]);

            dp_i[ny_k] = DistancesEngineT::fmadd(x_i, yp, dp_i[ny_k]);
        }
    }

    // xy -> y^2 - 2xy
    for (size_t ny_k = 0; ny_k < NY_POINTS_PER_LOOP; ny_k++) {
        const distances_type y_l2_sqr = DistancesEngineT::set1(*(y_norms + j + ny_k));

        dp_i[ny_k] = DistancesEngineT::fnmadd(dp_i[ny_k], DistancesEngineT::from_i32(2), y_l2_sqr);
    }
}


// transpose (SORTING_K, NX_POINTS) from final-s and 
//   write (NX_POINTS, SORTING_K) into (dis, ids) 
template<size_t NX_POINTS, size_t SORTING_K, typename output_ids_type>
//__attribute_noinline__
__attribute__((always_inline))
void offload(
    const float* const __restrict final_d,
    const uint32_t* const __restrict final_i,
    float* const __restrict dis,
    output_ids_type* const __restrict ids
) {
    if (dis != nullptr) {
        for (size_t nx_k = 0; nx_k < NX_POINTS; nx_k++) {
            for (size_t i_k = 0; i_k < SORTING_K; i_k++) {
                dis[nx_k * SORTING_K + i_k] = final_d[nx_k + i_k * NX_POINTS];
            }
        }
    }

    if (ids != nullptr) {
        for (size_t nx_k = 0; nx_k < NX_POINTS; nx_k++) {
            for (size_t i_k = 0; i_k < SORTING_K; i_k++) {
                ids[nx_k * SORTING_K + i_k] = 
                    static_cast<output_ids_type>(final_i[nx_k + i_k * NX_POINTS]);
            }
        }
    }
}


}  // namespace smalltopk
