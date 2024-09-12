#pragma once

#include <arm_sve.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "../utils/round.h"

#include "sve_vec.h"

#include "kernel_components.h"
#include "sorting_networks.h"

#include "../utils/macro_repeat_define.h"


namespace smalltopk {

namespace {


#define DECLARE_SORTING_PARAM(NX)   \
    const typename DistancesEngineT::simd_type sorting_d_##NX,

template <
    typename DistancesEngineT,
    typename IndicesEngineT,
    size_t NY_POINTS_PER_LOOP,
    size_t SORTING_K>
//__attribute_noinline__
__attribute__((always_inline))
void offload1(
        const typename DistancesEngineT::scalar_type* const __restrict x_norms,
        float* const __restrict dis,
        int64_t* const __restrict ids,
        // MAX_SORTING_K
        REPEAT_1D(DECLARE_SORTING_PARAM, 24)
        const uint32_t hacky_blender,
        const svbool_t dis_mask
) {
    using distances_type = typename DistancesEngineT::simd_type;
    using indices_type = typename IndicesEngineT::simd_type;

    using distance_type = typename DistancesEngineT::scalar_type;
    using index_type = typename IndicesEngineT::scalar_type;


    const auto dis_simd_width = DistancesEngineT::width();

    constexpr size_t SVE_MAX_WIDTH = 2048;

    // extract results

        const distances_type additional_norm = DistancesEngineT::load(dis_mask, x_norms);

        //
        float output_d[SVE_MAX_WIDTH * SORTING_K];
        uint32_t output_i[SVE_MAX_WIDTH * SORTING_K];

        // y^2 - 2xy -> max(0, y^2 - 2xy + x^2)
        auto finalize = [&dis_mask, &additional_norm, &output_d, &output_i, dis_simd_width, hacky_blender](
            const size_t i_k, const distances_type y2m2xy
        ){
            // hacky unpack 
            const svfloat32_t dis_v = svreinterpret_f32_u32(
                svand_n_u32_x(dis_mask, svreinterpret_u32_f32(y2m2xy), ~hacky_blender)
            );
            const svuint32_t ids_v = 
                svand_n_u32_x(dis_mask, svreinterpret_u32_f32(y2m2xy), hacky_blender);

            //
            distances_type final_distance = DistancesEngineT::add(
                dis_mask,
                additional_norm,
                dis_v
            );
            final_distance = DistancesEngineT::max(
                dis_mask,
                DistancesEngineT::zero(),
                final_distance
            );

            DistancesEngineT::store_as_f32(dis_mask, output_d + dis_simd_width * i_k, final_distance);
            IndicesEngineT::store_as_u32(dis_mask, output_i + dis_simd_width * i_k, ids_v);
        };

#define FINALIZE(NX) \
        if constexpr (SORTING_K >= NX + 1) { finalize(NX, sorting_d_##NX); }

        // MAX_SORTING_K
        REPEAT_1D(FINALIZE, 24)

#undef FINALIZE

    offload<SORTING_K>(output_d, output_i, dis, ids, dis_simd_width);
}

#undef DECLARE_SORTING_PARAM

}


template<
    typename DistancesEngineT,
    typename IndicesEngineT,
    size_t NY_POINTS_PER_LOOP>
bool kernel_sorting_fp32hack_pre_k(
        const typename DistancesEngineT::scalar_type* const __restrict x,
        const typename DistancesEngineT::scalar_type* const __restrict y,
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


    ////////////////////////////////////////////////////////////////////////
    // We're assuming that bit widths are the same.
    // 1. It is not worth considering a different scenario (impractical).
    // 2. Unlike x86, there's no performance gain for having f32+u16 vs f32+u32.
    static_assert(sizeof(distance_type) == sizeof(index_type));

    const auto dis_mask = DistancesEngineT::pred_all();


    // MAX DIM is 32
    // MAX SORTING_K is 24
    // MAX NY_POINTS_PER_LOOP 16

    ////////////////////////////////////////////////////////////////////////
    // transpose x
    // MAX_DIM
    distance_type transposed_x_values[32 * SVE_MAX_WIDTH];
    
#define DISPATCH_TRANSPOSED(DIM) \
    case DIM: transpose<DistancesEngineT, DIM>(x, transposed_x_values); break;

    switch(d) {
        // MAX_DIM
        REPEAT_P1_1D(DISPATCH_TRANSPOSED, 32);
        default:
            // not supported
            return false;        
    }


    ////////////////////////////////////////////////////////////////////////
    // introduce sorted indices and distances

    // distances_type sorting_d_0 = DistancesEngineT::max_value();
    // indices_type sorting_i_0 = IndicesEngineT::zero();
#define INTRO_SORTING(NX) \
    distances_type sorting_d_##NX = DistancesEngineT::max_value();  \
    indices_type sorting_i_##NX = IndicesEngineT::zero();   // indices are unused

    // MAX_SORTING_K
    if (k > 24) {
        return false;
    }

    // MAX_SORTING_K
    REPEAT_1D(INTRO_SORTING, 24)

#undef INTRO_SORTING


    ////////////////////////////////////////////////////////////////////////
    // main loop
    const size_t ny_16 = (ny / NY_POINTS_PER_LOOP) * NY_POINTS_PER_LOOP;

    for (size_t j = 0; j < ny_16; j += NY_POINTS_PER_LOOP) {
        // introduce dot products

#define INTRO_DP(NX) distances_type dp_i_##NX;

        // MAX NY_POINTS_PER_LOOP
        REPEAT_1D(INTRO_DP, 16)

#undef INTRO_DP


#define USE_DP_PARAM(NX) dp_i_##NX,

        // MAX_POINTS_PER_LOOP
#define DISPATCH_DISTANCES_X(DIM)                                                   \
        case DIM:                                                                   \
            distances<DistancesEngineT, IndicesEngineT, DIM, NY_POINTS_PER_LOOP>(   \
                y, ny, y_norms, transposed_x_values, j,                             \
                REPEAT_1D(USE_DP_PARAM, 16)                                         \
                dis_mask                                                            \
            );                                                                      \
            break;

        switch(d) {
            REPEAT_P1_1D(DISPATCH_DISTANCES_X, 32)
            default:
                // not supported
                return false;
        }


        // apply sorting networks
        {
            // Compare-and-exchange for a_d and b_d,
            //   based on a_d <=> b_d comparison.
            auto cmpxchg = [&dis_mask](
                distances_type& a_d, indices_type&,
                distances_type& b_d, indices_type&
            ) {
                // const auto cmp_d = DistancesEngineT::compare_le(dis_mask, a_d, b_d);

                // const distances_type min_d_new = DistancesEngineT::select(cmp_d, b_d, a_d);
                // const distances_type max_d_new = DistancesEngineT::select(cmp_d, a_d, b_d);

                const distances_type min_d_new = DistancesEngineT::min(dis_mask, a_d, b_d);
                const distances_type max_d_new = DistancesEngineT::max(dis_mask, a_d, b_d);

                a_d = min_d_new;
                b_d = max_d_new;
            };

            // introduce indices for candidates.
            // candidate distances are dp_i_NX

            // auto ids_candidate_0 = IndicesEngineT::set1(j + 0);
#define INTRO_IDS_CANDIDATE(NX) \
            indices_type ids_candidate_##NX = IndicesEngineT::set1(j + NX);

            // MAX_NY_POINTS_PER_LOOP
            REPEAT_1D(INTRO_IDS_CANDIDATE, 16)

#undef INTRO_IDS_CANDIDATE


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
            if constexpr (NY_POINTS_PER_LOOP >= NX + 1) { \
                dp_i_##NX = hacky_pack(dp_i_##NX, ids_candidate_##NX);  \
            }

            // MAX_NY_POINTS_PER_LOOP
            REPEAT_1D(BLEND_WITH_DIS, 16)

#undef BLEND_WITH_DIS


            // pick and apply an appropriate sorting network

#define ADD_SORTING_PAIR(NX) sorting_d_##NX, sorting_i_##NX,
#define ADD_CANDIDATE_PAIR(NX) dp_i_##NX, ids_candidate_##NX,

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
                PartialSortingNetwork<SRT_K, SRT_N>::sort<DistancesEngineT, IndicesEngineT, decltype(cmpxchg)>( \
                    REPEAT_1D(ADD_SORTING_PAIR, SRT_K)  \
                    REPEAT_1D(ADD_CANDIDATE_PAIR, SRT_N)    \
                    cmpxchg \
                );

#define DISPATCH_SORTING(SORTING_K)                 \
    case SORTING_K:                                 \
        if constexpr (NY_POINTS_PER_LOOP == 8) {    \
            DISPATCH_PARTIAL_SN(SORTING_K, 8);      \
        } else {                                    \
            return false;                           \
        };                                          \
        break;

            // warning: NY_POINTS_PER_LOOP != SRT_N
            // For example, for NY_POINTS_PER_LOOP==16 it is better to perform 
            //   process two dispatch (1,8) rather than a single dispatch(1,16)

            switch(k) {
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

    
#define USE_SORTING_PARAM(NX) sorting_d_##NX,

    // MAX_SORTING_K
#define DISPATCH_OFFLOAD(SORTING_K)                                                                 \
        case SORTING_K: offload1<DistancesEngineT, IndicesEngineT, NY_POINTS_PER_LOOP, SORTING_K>(  \
            x_norms, dis, ids,                                                                      \
            REPEAT_1D(USE_SORTING_PARAM, 24)                                                        \
            hacky_blender, dis_mask                                                                 \
        );                                                                                          \
        break;

    switch(k) {
        // MAX_SORTING_K
        REPEAT_P1_1D(DISPATCH_OFFLOAD, 24)
        default:
            // not supported
            return false;
    }

#undef USE_SORTING_PARAM
#undef DISPATCH_OFFLOAD

    // done
    return true;
}

}  // namespace smalltopk

#include "../utils/macro_repeat_undefine.h"
