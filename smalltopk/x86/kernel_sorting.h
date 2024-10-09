#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "../utils/macro_repeat_define.h"

#include "kernel_components.h"
#include "sorting_networks.h"

namespace smalltopk {


namespace {

template<typename DistancesEngineT, typename IndicesEngineT>
void cmpxchg(
    typename DistancesEngineT::simd_type& __restrict a_d, 
    typename IndicesEngineT::simd_type& __restrict a_i, 
    typename DistancesEngineT::simd_type& __restrict b_d, 
    typename IndicesEngineT::simd_type& __restrict b_i
) {
    using distances_type = typename DistancesEngineT::simd_type;
    using indices_type = typename IndicesEngineT::simd_type;

    //
    const auto cmp_d = DistancesEngineT::compare_le(a_d, b_d);

    const distances_type min_d_new = DistancesEngineT::select(cmp_d, b_d, a_d);
    const indices_type min_i_new = IndicesEngineT::select(cmp_d, b_i, a_i);

    const distances_type max_d_new = DistancesEngineT::select(cmp_d, a_d, b_d);
    const indices_type max_i_new = IndicesEngineT::select(cmp_d, a_i, b_i);

    a_d = min_d_new;
    a_i = min_i_new;
    b_d = max_d_new;
    b_i = max_i_new;                    
};


template <
    typename DistancesEngineT,
    typename IndicesEngineT,
    size_t NX_POINTS,
    size_t NY_POINTS_PER_LOOP,
    size_t SORTING_K>
//__attribute_noinline__
__attribute__((always_inline))
void offload1(
        const typename DistancesEngineT::scalar_type* const __restrict x_norms,
        float* const __restrict dis,
        int64_t* const __restrict ids,
        const typename DistancesEngineT::simd_type* const __restrict sorting_d,
        const typename IndicesEngineT::simd_type* const __restrict sorting_i
) {
    using distances_type = typename DistancesEngineT::simd_type;
    using indices_type = typename IndicesEngineT::simd_type;

    using distance_type = typename DistancesEngineT::scalar_type;
    using index_type = typename IndicesEngineT::scalar_type;

    // turn y^2 - 2xy -> x^2 + y^2 - 2xy
    const distances_type additional_norm = DistancesEngineT::load(x_norms);

    // temporary buffers
    float output_d[NX_POINTS * SORTING_K];
    uint32_t output_i[NX_POINTS * SORTING_K];

    for (size_t i_k = 0; i_k < SORTING_K; i_k++) {
        // y^2 - 2xy -> x^2 + y^2 - 2xy
        distances_type final_distance = DistancesEngineT::add(
            additional_norm,
            sorting_d[i_k]
        );

        // dist -> max(0, dist)
        final_distance = DistancesEngineT::max(
            DistancesEngineT::zero(),
            final_distance
        );

        // save to a temporary buffer
        DistancesEngineT::store_as_f32(output_d + NX_POINTS * i_k, final_distance);
        IndicesEngineT::store_as_u32(output_i + NX_POINTS * i_k, sorting_i[i_k]);
    }

    // offload
    offload<NX_POINTS, SORTING_K>(output_d, output_i, dis, ids);
}

}



template<
    typename DistancesEngineT,
    typename IndicesEngineT,
    size_t NY_POINTS_PER_LOOP>
bool kernel_sorting_pre_k(
        const typename DistancesEngineT::scalar_type* const __restrict x,
        const typename DistancesEngineT::scalar_type* const __restrict y_transposed,
        const size_t d,
        const size_t ny,
        const size_t k,
        const typename DistancesEngineT::scalar_type* const __restrict x_norms,
        const typename DistancesEngineT::scalar_type* const __restrict y_norms,
        float* const __restrict dis,
        int64_t* const __restrict ids
) {
    //
    using distances_type = typename DistancesEngineT::simd_type;
    using indices_type = typename IndicesEngineT::simd_type;

    using distance_type = typename DistancesEngineT::scalar_type;
    using index_type = typename IndicesEngineT::scalar_type;

    // 
    static_assert(DistancesEngineT::SIMD_WIDTH == IndicesEngineT::SIMD_WIDTH);
    static constexpr auto NX_POINTS = DistancesEngineT::SIMD_WIDTH;

    // MAX DIM is 32
    // MAX SORTING_K is 24

    // transpose x values: (NX_POINTS, DIM) into (DIM, NX_POINTS)
    // MAX_DIM
    distance_type transposed_x_values[32 * NX_POINTS];

#define DISPATCH_TRANSPOSE(DIM) \
    case DIM: transpose<DistancesEngineT, NX_POINTS, DIM>(x, transposed_x_values); break;

    switch(d) {
        // MAX_DIM
        REPEATR_1D(DISPATCH_TRANSPOSE, 1, 32);
        default:
            // not supported
            return false;
    }

#undef DISPATCH_TRANSPOSE


    ////////////////////////////////////////////////////////////////////////
    // introduce sorted indices and distances

    // MAX_SORTING_K
    if (k > 24) {
        // not supported
        return false;
    }

    // MAX_SORTING_K
    distances_type sorting_d[24];
    indices_type sorting_i[24];

    for (size_t i_k = 0; i_k < k; i_k++) {
        sorting_d[i_k] = DistancesEngineT::max_value();
        sorting_i[i_k] = IndicesEngineT::zero();
    }


    ////////////////////////////////////////////////////////////////////////
    // main loop
    const size_t ny_16 = (ny / NY_POINTS_PER_LOOP) * NY_POINTS_PER_LOOP;

    for (size_t j = 0; j < ny_16; j += NY_POINTS_PER_LOOP) {
        // introduce dot products
        distances_type dp_i[NY_POINTS_PER_LOOP];

        // compute y^2 - 2xy values
#define DISPATCH_DISTANCES(DIM) \
        case DIM: {                                                                             \
            distances<DistancesEngineT, IndicesEngineT, DIM, NX_POINTS, NY_POINTS_PER_LOOP>(    \
                y_transposed, ny, y_norms, transposed_x_values, j, dp_i                         \
            );                                                                                  \
            break;                                                                              \
        }

        // MAX_DIM
        switch(d) {
            REPEATR_1D(DISPATCH_DISTANCES, 1, 32)
            default:
                // not supported
                return false;
        }

#undef DISPATCH_DISTANCES

        // apply sorting networks
        {
            // introduce index candidates
            indices_type ids_candidate[NY_POINTS_PER_LOOP];
            for (size_t ny_k = 0; ny_k < NY_POINTS_PER_LOOP; ny_k++) {
                ids_candidate[ny_k] = IndicesEngineT::set1(j + ny_k);
            }

            // sorting network

            static constexpr auto comparer = cmpxchg<DistancesEngineT, IndicesEngineT>;

#define DISPATCH_PARTIAL_SN(SRT_K, SRT_N, OFFSET_N)                                                              \
        {                                                                                                        \
            PartialSortingNetwork<SRT_K, SRT_N>::template sort<DistancesEngineT, IndicesEngineT, decltype(comparer)>(    \
                sorting_d,                                                                                       \
                sorting_i,                                                                                       \
                dp_i + OFFSET_N,                                                                                 \
                ids_candidate + OFFSET_N,                                                                        \
                comparer                                                                                         \
            );                                                                                                   \
        }

        // dispatch for NY_POINTS_PER_LOOP = 16, else fail
#define DISPATCH_SN(SORTING_K)                          \
        case SORTING_K:                                 \
            if constexpr(NY_POINTS_PER_LOOP == 16) {    \
                DISPATCH_PARTIAL_SN(SORTING_K, 8, 0);   \
                DISPATCH_PARTIAL_SN(SORTING_K, 8, 8);   \
            } else {                                    \
                return false;                           \
            }                                           \
            break;

            switch(k) {
                // MAX_SORTING_K
                REPEATR_1D(DISPATCH_SN, 1, 24)
                default:
                    // not supported
                    return false;
            }
        }

#undef DISPATCH_PARTIAL_SN
#undef DISPATCH_SN

    }


    // offload the results
#define DISPATCH_OFFLOAD(SORTING_K)                                                                 \
        case SORTING_K:                                                                             \
            offload1<DistancesEngineT, IndicesEngineT, NX_POINTS, NY_POINTS_PER_LOOP, SORTING_K>(   \
                x_norms, dis, ids, sorting_d, sorting_i                                             \
            );                                                                                      \
            break; 

    switch(k) {
        // MAX_SORTING_K
        REPEATR_1D(DISPATCH_OFFLOAD, 1, 24)
        default:
            // not supported
            return false;
    }

#undef DISPATCH_OFFLOAD

    return true;
}

}  // namespace smalltopk

#include "../utils/macro_repeat_undefine.h"
