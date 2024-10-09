#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "../utils/macro_repeat_define.h"
#include "../utils/round.h"

#include "sorting_networks.h"


namespace smalltopk {

namespace {

template<typename DistancesEngineT, typename IndicesEngineT>
void cmpxchg(
    typename DistancesEngineT::simd_type& __restrict a_d, 
    typename IndicesEngineT::simd_type& __restrict, 
    typename DistancesEngineT::simd_type& __restrict b_d, 
    typename IndicesEngineT::simd_type& __restrict
) {
    using distances_type = typename DistancesEngineT::simd_type;

    using distance_type = typename DistancesEngineT::scalar_type;

    const distances_type min_d_new = DistancesEngineT::min(a_d, b_d);
    const distances_type max_d_new = DistancesEngineT::max(a_d, b_d);

    a_d = min_d_new;
    b_d = max_d_new;
};

}


template<
    typename DistancesEngineT,
    typename IndicesEngineT,
    size_t N_MAX_LEVELS,
    size_t N_REGISTERS_PER_LOOP>
bool kernel_getmink_fp32hack(
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

    // Round up to the next highest power of 2
    uint32_t ny_power = next_power_of_2(ny);

    // should be 0xFF for ny=256 (2^8) or 0x1FF for ny=512 (2^9)
    // should be 0x1FF for ny=257 (because 2^9 bits are needed)
    const uint32_t hacky_blender = ny_power - 1;

    // 
    distances_type sorting_d[N_MAX_LEVELS];
    indices_type sorting_i[N_MAX_LEVELS];   // unused

    for (size_t i_k = 0; i_k < N_MAX_LEVELS; i_k++) {
        sorting_d[i_k] = DistancesEngineT::max_value();
    }

    ////////////////////////////////////////////////////////////////////////
    // main loop
    const size_t ny_16 = 
        ((ny + (N_REGISTERS_PER_LOOP * DistancesEngineT::SIMD_WIDTH) - 1) 
            / (N_REGISTERS_PER_LOOP * DistancesEngineT::SIMD_WIDTH)) 
            * (N_REGISTERS_PER_LOOP * DistancesEngineT::SIMD_WIDTH);

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

        // load
        if (j + DistancesEngineT::SIMD_WIDTH * N_REGISTERS_PER_LOOP <= ny) [[likely]] {
            // regular load: all distances are fully loaded
            for (size_t ny_k = 0; ny_k < N_REGISTERS_PER_LOOP; ny_k++) {
                dis_candidate[ny_k] = DistancesEngineT::load(src_dis + j + ny_k * DistancesEngineT::SIMD_WIDTH);
            }
        } else {
            // partial load: only some of distances are available
            const distances_type maxv = DistancesEngineT::max_value();

            for (size_t ny_k = 0; ny_k < N_REGISTERS_PER_LOOP; ny_k++) {
                const auto mask = DistancesEngineT::whilelt(j + ny_k * DistancesEngineT::SIMD_WIDTH, ny);
                dis_candidate[ny_k] = DistancesEngineT::mask_load(mask, maxv, src_dis + j + ny_k * DistancesEngineT::SIMD_WIDTH);
            }
        }

        // apply fp32hack
        for (size_t ny_k = 0; ny_k < N_REGISTERS_PER_LOOP; ny_k++) {
            const __m512 dp = dis_candidate[ny_k];

            const __m512i reduced_dis = _mm512_and_si512((__m512i)dp, _mm512_set1_epi32(~hacky_blender));
            const __m512i blended_dis_u32 = _mm512_or_si512(reduced_dis, ids_candidate[ny_k]);

            dis_candidate[ny_k] = (__m512)blended_dis_u32;
        }

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

        // do a shift in corresponding lanes one level down
        for (size_t p = 0; p < N_MAX_LEVELS - 1; p++) {
            sorting_d[p] = DistancesEngineT::select(
                mindmask,
                sorting_d[p],
                sorting_d[p + 1]
            );
        }

        // kill item on last level by setting it to an infinity()
        sorting_d[N_MAX_LEVELS - 1] = DistancesEngineT::select(
            mindmask,
            sorting_d[N_MAX_LEVELS - 1],
            DistancesEngineT::max_value()
        );

        // store
        //
        // fp32hack makes sure that all values are unique,
        //   so n_new = 1
        uint32_t min_distance_u = *reinterpret_cast<const uint32_t*>(&min_distance_v);
        uint32_t min_distance_dis = min_distance_u & (~hacky_blender);
        uint32_t min_distance_ids = min_distance_u & (hacky_blender);  
        out_dis[n_extracted] = *(reinterpret_cast<const float*>(&min_distance_dis));
        out_ids[n_extracted] = static_cast<int32_t>(min_distance_ids);

        // done
        n_extracted += 1;
    }

    return true;
}

}

#include "../utils/macro_repeat_undefine.h"
