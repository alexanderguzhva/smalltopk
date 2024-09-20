#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "kernel_components.h"
#include "sorting_networks.h"

#include "../utils/macro_repeat_define.h"


namespace smalltopk {

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
    const auto dis_mask = DistancesEngineT::pred_all();

    const auto cmp_d = DistancesEngineT::compare_le(dis_mask, a_d, b_d);

    const distances_type min_d_new = DistancesEngineT::select(cmp_d, b_d, a_d);
    const indices_type min_i_new = IndicesEngineT::select(cmp_d, b_i, a_i);

    const distances_type max_d_new = DistancesEngineT::select(cmp_d, a_d, b_d);
    const indices_type max_i_new = IndicesEngineT::select(cmp_d, a_i, b_i);

    a_d = min_d_new;
    a_i = min_i_new;
    b_d = max_d_new;
    b_i = max_i_new;                    
};

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


    ////////////////////////////////////////////////////////////////////////
    constexpr size_t SVE_MAX_WIDTH = 2048;
    const auto dis_simd_width = DistancesEngineT::width();

    // check whether the task can be accomplished
    if (k > dis_simd_width * N_MAX_LEVELS) {
        // say, we have two 16 lane registers and k is 48
        return false;
    }


    ////////////////////////////////////////////////////////////////////////
    // We're assuming that bit widths are the same.
    // 1. It is not worth considering a different scenario (impractical).
    // 2. Unlike x86, there's no performance gain for having f32+u16 vs f32+u32.
    static_assert(sizeof(distance_type) == sizeof(index_type));

    const auto dis_mask = DistancesEngineT::pred_all();


    ////////////////////////////////////////////////////////////////////////
    // introduce sorted indices and distances

    // N_MAX_LEVELS = 24
    // MAX_N_REGISTERS_PER_LOOP = 8

    // distances_type sorting_d_0 = DistancesEngineT::max_value();
    // indices_type sorting_i_0 = IndicesEngineT::zero();
#define INTRO_SORTING(NX) \
    distances_type sorting_d_##NX = DistancesEngineT::max_value();  \
    indices_type sorting_i_##NX = IndicesEngineT::zero();

    // N_MAX_LEVELS
    if (k > 24) {
        return false;
    }

    // N_MAX_LEVELS
    REPEAT_1D(INTRO_SORTING, 24)

#undef INTRO_SORTING


    ////////////////////////////////////////////////////////////////////////
    // main loop
    const size_t ny_16 = 
        ((ny + (N_REGISTERS_PER_LOOP * dis_simd_width) - 1) 
            / (N_REGISTERS_PER_LOOP * dis_simd_width)) 
            * (N_REGISTERS_PER_LOOP * dis_simd_width);

    indices_type offset_base = IndicesEngineT::staircase();

    for (size_t j = 0; j < ny_16; j += dis_simd_width * N_REGISTERS_PER_LOOP) {

        // apply sorting networks
        {
            // introduce indices for candidates.

#define INTRO_IDS_CANDIDATE(NX) \
            indices_type ids_candidate_##NX = offset_base; \
            offset_base = IndicesEngineT::add(dis_mask, offset_base, IndicesEngineT::set1(dis_simd_width));

            // MAX_N_REGISTERS_PER_LOOP
            REPEAT_1D(INTRO_IDS_CANDIDATE, 8)

#undef INTRO_IDS_CANDIDATE


            // introduce distances for candidates.
    
#define INTRO_DIS_CANDIDATE(NX) \
            distances_type dis_candidate_##NX;

            // MAX_N_REGISTERS_PER_LOOP
            REPEAT_1D(INTRO_DIS_CANDIDATE, 8)

#undef INTRO_DIS_CANDIDATE

            // load
            if (j + dis_simd_width * N_REGISTERS_PER_LOOP <= ny) {
                // regular load: all distances are fully loaded

#define LOAD_DIS_CANDIDATE(NX) \
                dis_candidate_##NX = DistancesEngineT::load(dis_mask, src_dis + j + NX * dis_simd_width);

                // MAX_N_REGISTERS_PER_LOOP
                REPEAT_1D(LOAD_DIS_CANDIDATE, 8)

#undef LOAD_DIS_CANDIDATE

            } else {
                // partial load: only some of distances are available
                const distances_type maxv = DistancesEngineT::max_value();

#define PARTIAL_LOAD_DIS_CANDIDATE(NX) \
                { \
                    const auto tmp_mask = IndicesEngineT::whilelt(j + dis_simd_width * NX, ny); \
                    const distances_type tmp_dis = DistancesEngineT::load(tmp_mask, src_dis + j + NX * dis_simd_width); \
                    dis_candidate_##NX = DistancesEngineT::select(tmp_mask, maxv, tmp_dis); \
                }

                // MAX_N_REGISTERS_PER_LOOP
                REPEAT_1D(PARTIAL_LOAD_DIS_CANDIDATE, 8)

#undef PARTIAL_LOAD_DIS_CANDIDATE
            }


            // pick and apply an appropriate sorting network

            static constexpr auto comparer = cmpxchg<DistancesEngineT, IndicesEngineT>;

#define ADD_SORTING_PAIR(NX) sorting_d_##NX, sorting_i_##NX,
#define ADD_CANDIDATE_PAIR(NX) dis_candidate_##NX, ids_candidate_##NX,

                // PartialSortingNetwork<3, 8>::sort<DistancesEngineT, IndicesEngineT, decltype(cmpxchg)>(
                //     sorting_d_0, sorting_i_0,
                //     sorting_d_1, sorting_i_1,
                //     sorting_d_2, sorting_i_2,
                //     dp_i_0, ids_candidate_0,
                //     dp_i_1, ids_candidate_1,
                //     dp_i_2, ids_candidate_2,
                //     dp_i_3, ids_candidate_3,
                //     dp_i_4, ids_candidate_4,
                //     dp_i_5, ids_candidate_5,
                //     dp_i_6, ids_candidate_6,
                //     dp_i_7, ids_candidate_7,
                //     cmpxchg
                // );
#define DISPATCH_PARTIAL_SN(SRT_K, SRT_N) \
                PartialSortingNetwork<SRT_K, SRT_N>::template sort<DistancesEngineT, IndicesEngineT, decltype(comparer)>( \
                    REPEAT_1D(ADD_SORTING_PAIR, SRT_K)  \
                    REPEAT_1D(ADD_CANDIDATE_PAIR, SRT_N)    \
                    comparer \
                );

#define DISPATCH_SORTING(N_LEVELS)                  \
    case N_LEVELS:                                  \
        if constexpr (N_REGISTERS_PER_LOOP == 8) {  \
            DISPATCH_PARTIAL_SN(N_LEVELS, 8);       \
        } else {                                    \
            return false;                           \
        };                                          \
        break;

            // warning: NY_POINTS_PER_LOOP != SRT_N
            // For example, for NY_POINTS_PER_LOOP==16 it is better to perform 
            //   process two dispatch (1,8) rather than a single dispatch(1,16)

            switch(N_MAX_LEVELS) {
                REPEAT_P1_1D(DISPATCH_SORTING, 24)
                default:
                    // not supported
                    return false;
            }

#undef DISPATCH_SORTING
#undef DISPATCH_PARTIAL_SN
#undef ADD_CANDIDATE_PAIR
#undef ADD_SORTING_PAIR
        }
    }

    // todo: k=1 case?

    // extract k min values from a stack of lane-sorted SIMD registers.
    // note that sorting[0] contains the smallest values for every lane.
    size_t n_extracted = 0;
    while (n_extracted < k) {
        // horizontal min reduce into a scalar value
        const auto min_distance_v = DistancesEngineT::reduce_min(dis_mask, sorting_d_0);

        // find lanes with corresponding min_distance_v
        const auto mindmask = DistancesEngineT::compare_eq(
            dis_mask, 
            sorting_d_0,
            DistancesEngineT::set1(min_distance_v));

        // save indices
        const indices_type saved_indices = sorting_i_0;
        int n_new = IndicesEngineT::mask_popcount(mindmask);
        
        // do a shift in corresponding lanes one level down
#define MOVE_LANES(NX, NP1)  \
        if (NX + 1 < N_MAX_LEVELS) {    \
            sorting_d_##NX = DistancesEngineT::select(  \
                mindmask,   \
                sorting_d_##NX, \
                sorting_d_##NP1 \
            );  \
            sorting_i_##NX = IndicesEngineT::select(    \
                mindmask,   \
                sorting_i_##NX, \
                sorting_i_##NP1 \
            );  \
        }

        // N_MAX_LEVELS
        REPEAT_NP1_1D(MOVE_LANES, 23)

#undef MOVE_LANES

        // kill item on last level by setting it to an infinity()

#define SCRAP_LAST(NX)  \
        if (N_MAX_LEVELS == NX + 1) {   \
            sorting_d_##NX = DistancesEngineT::select(  \
                mindmask,   \
                sorting_d_##NX, \
                DistancesEngineT::max_value()   \
            );  \
        }

        // N_MAX_LEVELS
        REPEAT_1D(SCRAP_LAST, 24)

#undef SCRAP_LAST

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


    // done
    return true;
}

}  // namespace smalltopk

#include "../utils/macro_repeat_undefine.h"
