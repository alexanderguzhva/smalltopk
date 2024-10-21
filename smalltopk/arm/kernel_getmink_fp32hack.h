#pragma once

#include <arm_sve.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <smalltopk/utils/round.h>

#include <smalltopk/arm/sorting_networks.h>

#include <smalltopk/utils/macro_repeat_define.h>

namespace smalltopk {

template<typename DistancesEngineT, typename IndicesEngineT>
void cmpxchg(
    typename DistancesEngineT::simd_type& __restrict a_d, 
    typename IndicesEngineT::simd_type& __restrict, 
    typename DistancesEngineT::simd_type& __restrict b_d, 
    typename IndicesEngineT::simd_type& __restrict
) {
    using distances_type = typename DistancesEngineT::simd_type;

    //
    const auto dis_mask = DistancesEngineT::pred_all();

    const distances_type min_d_new = DistancesEngineT::min(dis_mask, a_d, b_d);
    const distances_type max_d_new = DistancesEngineT::max(dis_mask, a_d, b_d);

    a_d = min_d_new;
    b_d = max_d_new;
};

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

    // hacky pack requirements
    static_assert(std::is_same_v<distance_type, float>);
    static_assert(std::is_same_v<index_type, uint32_t>);

    // Round up to the next highest power of 2
    uint32_t ny_power = next_power_of_2(ny);

    // should be 0xFF for ny=256 (2^8) or 0x1FF for ny=512 (2^9)
    // should be 0x1FF for ny=257 (because 2^9 bits are needed)
    const uint32_t hacky_blender = ny_power - 1;


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
    indices_type sorting_i_##NX = IndicesEngineT::zero();   // indices are ignored

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


            // hacky pack indices with distances

            auto hacky_pack = [&dis_mask, hacky_blender](const distances_type& dis, const indices_type& ids) {
                // basically, we chop off lowest bits from dis 
                //   and replace ones with from ids
                const svuint32_t reduced_dis = svand_n_u32_x(dis_mask, svreinterpret_u32_f32(dis), ~hacky_blender);
                // const svuint32_t reduced_ids = svand_n_u32_x(dis_mask, ids, hacky_blender);

                const svuint32_t blended_dis_u32 = svorr_u32_x(dis_mask, reduced_dis, ids);
                const svfloat32_t blended_dis = svreinterpret_f32_u32(blended_dis_u32);

                return blended_dis;
            };

#define BLEND_WITH_DIS(NX) \
            if constexpr (N_REGISTERS_PER_LOOP >= NX + 1) { \
                dis_candidate_##NX = hacky_pack(dis_candidate_##NX, ids_candidate_##NX);  \
            }

            // MAX_NY_POINTS_PER_LOOP
            REPEAT_1D(BLEND_WITH_DIS, 8)

#undef BLEND_WITH_DIS


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
        
        // do a shift in corresponding lanes one level down
#define MOVE_LANES(NX, NP1)  \
        if (NX + 1 < N_MAX_LEVELS) {    \
            sorting_d_##NX = DistancesEngineT::select(  \
                mindmask,   \
                sorting_d_##NX, \
                sorting_d_##NP1 \
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

        uint32_t min_distance_u = *reinterpret_cast<const uint32_t*>(&min_distance_v);
        uint32_t min_distance_dis = min_distance_u & (~hacky_blender);
        uint32_t min_distance_ids = min_distance_u & (hacky_blender);  
        out_dis[n_extracted] = *(reinterpret_cast<const float*>(&min_distance_dis));
        out_ids[n_extracted] = static_cast<int32_t>(min_distance_ids);

        // done
        n_extracted += 1;
    }


    // done
    return true;
}

}  // namespace smalltopk

#include <smalltopk/utils/macro_repeat_undefine.h>
