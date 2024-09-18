#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "../utils/macro_repeat_define.h"

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

    using distance_type = typename DistancesEngineT::scalar_type;
    using index_type = typename IndicesEngineT::scalar_type;

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

}


template<
    typename DistancesEngineT,
    typename IndicesEngineT,
    size_t N_MAX_LEVELS,
    size_t N_REGISTERS_PER_LOOP>
bool kernel_getmink(
    const typename DistancesEngineT::scalar_type* const __restrict src_dis,
    const size_t ny,
    const size_t k,
    float* const __restrict out_dis,
    int32_t* const __restrict out_ids
) {
    //
    using distances_type = typename DistancesEngineT::simd_type;
    using indices_type = typename IndicesEngineT::simd_type;

    using distance_type = typename DistancesEngineT::scalar_type;
    using index_type = typename IndicesEngineT::scalar_type;

    // 
    static_assert(DistancesEngineT::SIMD_WIDTH == IndicesEngineT::SIMD_WIDTH);

    // check whether the task can be accomplished
    if (k > DistancesEngineT::SIMD_WIDTH * N_MAX_LEVELS) {
        // say, we have two 16 lane registers and k is 48
        return false;
    }

    // 
    distances_type sorting_d[N_MAX_LEVELS];
    indices_type sorting_i[N_MAX_LEVELS];

    for (size_t i_k = 0; i_k < N_MAX_LEVELS; i_k++) {
        sorting_d[i_k] = DistancesEngineT::max_value();
        sorting_i[i_k] = IndicesEngineT::zero();
    }

    ////////////////////////////////////////////////////////////////////////
    // main loop
    const size_t ny_16 = (ny / (N_REGISTERS_PER_LOOP * DistancesEngineT::SIMD_WIDTH)) * (N_REGISTERS_PER_LOOP * DistancesEngineT::SIMD_WIDTH);

    indices_type offset_base = IndicesEngineT::staircase();

    for (size_t j = 0; j < ny_16; j += DistancesEngineT::SIMD_WIDTH * N_REGISTERS_PER_LOOP) {
        // introduce index candidates
        indices_type ids_candidate[N_REGISTERS_PER_LOOP];
        for (size_t ny_k = 0; ny_k < N_REGISTERS_PER_LOOP; ny_k++) {
            ids_candidate[ny_k] = offset_base;
            offset_base = IndicesEngineT::add(offset_base, IndicesEngineT::set1(DistancesEngineT::SIMD_WIDTH));
        }

        // introduce values
        distances_type dis_candidate[N_REGISTERS_PER_LOOP];
        
        // if () [[likely]] {
            for (size_t ny_k = 0; ny_k < N_REGISTERS_PER_LOOP; ny_k++) {
                dis_candidate[ny_k] = DistancesEngineT::load(src_dis + j + ny_k * DistancesEngineT::SIMD_WIDTH);
            }
        // } else {

        // }


        // sorting network

        static constexpr auto comparer = cmpxchg<DistancesEngineT, IndicesEngineT>;

#define DISPATCH_PARTIAL_SN(SRT_K, SRT_N, OFFSET_N)                                                              \
        {                                                                                                        \
            PartialSortingNetwork<SRT_K, SRT_N>::template sort<DistancesEngineT, IndicesEngineT, decltype(comparer)>(     \
                sorting_d,                                                                                       \
                sorting_i,                                                                                       \
                dis_candidate + OFFSET_N,                                                                        \
                ids_candidate + OFFSET_N,                                                                        \
                comparer                                                                                         \
            );                                                                                                   \
        }

        DISPATCH_PARTIAL_SN(N_MAX_LEVELS, N_REGISTERS_PER_LOOP, 0)

#undef DISPATCH_PARTIAL_SN

    }

    // todo: k=1 case?

    // extract k min values from a stack of lane-sorted SIMD registers.
    // note that sorting[0] contains the smallest values for every lane.
    size_t n_extracted = 0;
    while (n_extracted < k) {
        // horizontal min reduce into a scalar value
        const auto min_distance_v = DistancesEngineT::reduce_min(sorting_d[0]);

        // find lanes with corresponding min_distance_v
        const auto mindmask = DistancesEngineT::compare_eq(
            sorting_d[0],
            DistancesEngineT::set1(min_distance_v));

        // save indices
        const indices_type saved_indices = sorting_i[0];
        int n_new = __builtin_popcount(mindmask);
        
        // do a shift in corresponding lanes one level down
        for (size_t p = 0; p < N_MAX_LEVELS - 1; p++) {
            sorting_d[p] = DistancesEngineT::select(
                mindmask,
                sorting_d[p],
                sorting_d[p + 1]
            );

            sorting_i[p] = IndicesEngineT::select(
                mindmask,
                sorting_i[p],
                sorting_i[p + 1]
            );
        }

        // kill item on last level by setting it to an infinity()
        sorting_d[N_MAX_LEVELS - 1] = DistancesEngineT::select(
            mindmask,
            sorting_d[N_MAX_LEVELS - 1],
            DistancesEngineT::max_value()
        );

        // store
        if (n_extracted + n_new > k) {
            n_new = k - n_extracted;
        }

        if (n_new == 1) [[likely]] {
            out_dis[n_extracted] = static_cast<float>(min_distance_v);

            IndicesEngineT::compress_store_1_as_i32(
                out_ids + n_extracted, mindmask, saved_indices);
        } else {
            for (size_t q = 0; q < n_new; q++) {
                out_dis[n_extracted + q] = static_cast<float>(min_distance_v);
            }

            IndicesEngineT::compress_store_n_as_i32(
                out_ids + n_extracted, n_new, mindmask, saved_indices
            );
        }

        n_extracted += n_new;
    }

    return true;
}

}

#include "../utils/macro_repeat_undefine.h"

