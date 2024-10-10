#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <smalltopk/utils/round.h>

#include <smalltopk/x86/avx512_vec_fp32.h>

#include <smalltopk/x86/kernel_components.h>
#include <smalltopk/x86/sorting_networks.h>

#include <smalltopk/utils/macro_repeat_define.h>

namespace smalltopk {

namespace {

static inline void cmpxchg(
    vec_f32x16::simd_type& a_d, vec_u32x16::simd_type&,
    vec_f32x16::simd_type& b_d, vec_u32x16::simd_type&
) {
    const vec_f32x16::simd_type min_d_new = vec_f32x16::min(a_d, b_d);
    const vec_f32x16::simd_type max_d_new = vec_f32x16::max(a_d, b_d);

    a_d = min_d_new;
    b_d = max_d_new;
};


template <
    typename DistancesEngineT,
    typename IndicesEngineT,
    size_t NX_POINTS,
    size_t NY_POINTS_PER_LOOP,
    size_t SORTING_K,
    typename output_ids_type>
//__attribute_noinline__
__attribute__((always_inline))
void offload1(
        const typename DistancesEngineT::scalar_type* const __restrict x_norms,
        float* const __restrict dis,
        output_ids_type* const __restrict ids,
        const typename DistancesEngineT::simd_type* const __restrict sorting_d,
        const uint32_t hacky_blender
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
        // hacky unpack
        const __m512 dis_v = (__m512)_mm512_and_si512((__m512i)sorting_d[i_k], _mm512_set1_epi32(~hacky_blender));
        const __m512i ids_v = _mm512_and_si512((__m512i)sorting_d[i_k], _mm512_set1_epi32(hacky_blender));

        // y^2 - 2xy -> x^2 + y^2 - 2xy
        distances_type final_distance = DistancesEngineT::add(
            additional_norm,
            dis_v
        );

        // dist -> max(0, dist)
        final_distance = DistancesEngineT::max(
            DistancesEngineT::zero(),
            final_distance
        );

        // save to a temporary buffer
        DistancesEngineT::store_as_f32(output_d + NX_POINTS * i_k, final_distance);
        IndicesEngineT::store_as_u32(output_i + NX_POINTS * i_k, ids_v);
    }

    // offload
    offload<NX_POINTS, SORTING_K, output_ids_type>(
        output_d, output_i, dis, ids);
}

}


//
struct TileConfig
{
    // must be 1
    uint8_t paletteId;
    // must be 0
    uint8_t startRow;
    uint8_t reserved[14];
    // measured in bytes
    uint16_t colsb[16];
    // measured in rows
    uint8_t rows[16];
};

static inline void convert_for_matrix_A(
    const float* const __restrict src, 
    uint16_t* const __restrict dst
) {
    const __m512 s0 = _mm512_loadu_ps(src + 0 * 16);
    const __m512 s1 = _mm512_loadu_ps(src + 1 * 16);
    _mm512_storeu_si512(dst, (__m512i)_mm512_cvtne2ps_pbh(s1, s0));
}

static inline void convert_for_matrix_B(
    const float* const __restrict src, 
    const size_t stride, 
    uint16_t* const __restrict dst
) {
    const __m512i PERM_IDX = _mm512_set_epi16(
        0x1f, 0x0f, 0x1e, 0x0e, 0x1d, 0x0d, 0x1c, 0x0c, 
        0x1b, 0x0b, 0x1a, 0x0a, 0x19, 0x09, 0x18, 0x08,
        0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14, 0x04, 
        0x13, 0x03, 0x12, 0x02, 0x11, 0x01, 0x10, 0x00);

    const __m512 s0 = _mm512_loadu_ps(src + 0 * stride);
    const __m512 s1 = _mm512_loadu_ps(src + 1 * stride);
    const __m512i d = (__m512i)_mm512_cvtne2ps_pbh(s1, s0);
    _mm512_storeu_si512(dst, _mm512_permutexvar_epi16(PERM_IDX, d));
}

template<
    typename DistancesEngineT,
    typename IndicesEngineT,
    size_t NY_POINTS_PER_LOOP,
    typename output_ids_type>
bool kernel_sorting_fp32hack_amx_pre_k(
        const float* const __restrict x,
        const uint16_t* const __restrict y,
        const size_t d,
        const size_t ny,
        const size_t k,
        const float* const __restrict x_norms,
        const float* const __restrict y_norms,
        float* const __restrict dis,
        output_ids_type* const __restrict ids
) {
    //
    using distances_type = typename DistancesEngineT::simd_type;
    using indices_type = typename IndicesEngineT::simd_type;

    using distance_type = typename DistancesEngineT::scalar_type;
    using index_type = typename IndicesEngineT::scalar_type;

    // this is for f32 only
    static_assert(std::is_same_v<distances_type, __m512>);
    static_assert(std::is_same_v<indices_type, __m512i>);

    // 
    static constexpr auto NX_POINTS = DistancesEngineT::SIMD_WIDTH;

    // Round up to the next highest power of 2
    uint32_t ny_power = next_power_of_2(ny);

    // should be 0xFF for ny=256 (2^8) or 0x1FF for ny=512 (2^9)
    // should be 0x1FF for ny=257 (because 2^9 bits are needed)
    const uint32_t hacky_blender = ny_power - 1;

    // MAX DIM is 32
    // MAX SORTING_K is 24

    float xu[32][16] = {};

#define DISPATCH_XU(DIM) \
    case DIM:   \
        for (size_t nx_k = 0; nx_k < 16; nx_k++) {  \
            for (size_t dd32 = 0; dd32 < DIM; dd32++) {   \
                xu[dd32][nx_k] = x[nx_k * DIM + dd32];    \
            }   \
        }   \
        break;

    switch(d) {
        // MAX_DIM
        REPEATR_1D(DISPATCH_XU, 1, 32);
        default:
            // not supported
            return false;
    }

#undef DISPATCH_XU

    // convert to bf16
    uint16_t x_i_bf16[16][32];

    for (size_t nx_k = 0; nx_k < 16; nx_k++) {
        convert_for_matrix_B(&(xu[0][0]) + nx_k * 32, 16, x_i_bf16[nx_k]);
    }

    // load Xt into tile 1
    _tile_loadd(1, x_i_bf16, 64);


    ////////////////////////////////////////////////////////////////////////
    // introduce sorted indices and distances

    // MAX_SORTING_K
    if (k > 24) {
        // not supported
        return false;
    }

    // MAX_SORTING_K
    distances_type sorting_d[24];
    indices_type sorting_i[24];      // indices are unused

    for (size_t i_k = 0; i_k < k; i_k++) {
        sorting_d[i_k] = DistancesEngineT::max_value();
    }


    ////////////////////////////////////////////////////////////////////////
    // main loop
    const size_t ny_16 = (ny / NY_POINTS_PER_LOOP) * NY_POINTS_PER_LOOP;

    for (size_t j = 0; j < ny_16; j += NY_POINTS_PER_LOOP) {
        // AMX dot products
        float dot_products[16][16];

        // we perform dot_products += Y*Xt;
        // clear tile 0
        _tile_zero(0);
        // load y into tile 2
        _tile_loadd(2, y + (j + 0 * 16) * 32, 64);
        // tile 0 += tile 2 * tile 1t
        _tile_dpbf16ps(0, 2, 1);
        // done, save tile 0
        _tile_stored(0, dot_products, 64);

        // introduce AVX dot products
        distances_type dp_i[NY_POINTS_PER_LOOP];

        for (size_t nx_k = 0; nx_k < 16; nx_k++) {
            dp_i[nx_k] = _mm512_loadu_ps(dot_products[nx_k]);
            dp_i[nx_k] = _mm512_fnmadd_ps(dp_i[nx_k], _mm512_set1_ps(2), _mm512_set1_ps(y_norms[j + nx_k]));
        }

        // apply sorting networks
        {
            // introduce index candidates
            indices_type ids_candidate[NY_POINTS_PER_LOOP];
            for (size_t ny_k = 0; ny_k < NY_POINTS_PER_LOOP; ny_k++) {
                ids_candidate[ny_k] = IndicesEngineT::set1(j + ny_k);
            }

            // hacky pack index candidates with distance candidates
            for (size_t ny_k = 0; ny_k < NY_POINTS_PER_LOOP; ny_k++) {
                const __m512i reduced_dis = _mm512_and_si512((__m512i)dp_i[ny_k], _mm512_set1_epi32(~hacky_blender));
                const __m512i blended_dis_u32 = _mm512_or_si512(reduced_dis, ids_candidate[ny_k]);
                
                dp_i[ny_k] = (__m512)blended_dis_u32;
            }


            // sorting network

#define DISPATCH_PARTIAL_SN(SRT_K, SRT_N, OFFSET_N)                                                             \
        {                                                                                                       \
            PartialSortingNetwork<SRT_K, SRT_N>::template sort<DistancesEngineT, IndicesEngineT, decltype(&cmpxchg)>(    \
                sorting_d,                                                                                      \
                sorting_i,                                                                                      \
                dp_i + OFFSET_N,                                                                                \
                ids_candidate + OFFSET_N,                                                                       \
                cmpxchg                                                                                         \
            );                                                                                                  \
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
#define DISPATCH_OFFLOAD(SORTING_K)                                                                                  \
        case SORTING_K:                                                                                              \
            offload1<DistancesEngineT, IndicesEngineT, NX_POINTS, NY_POINTS_PER_LOOP, SORTING_K, output_ids_type>(   \
                x_norms, dis, ids, sorting_d, hacky_blender                                                          \
            );                                                                                                       \
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

#include <smalltopk/utils/macro_repeat_undefine.h>
