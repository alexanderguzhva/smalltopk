#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace smalltopk {


static inline uint16_t fp32_to_fp16(const float v) {
    const __m128 xf = _mm_set1_ps(v);
    const __m128i xi = _mm_cvtps_ph(xf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return static_cast<uint16_t>(_mm_cvtsi128_si32(xi) & 0xffff);
}

static inline void fp32_to_fp16(
    const float* const __restrict src, 
    uint16_t* const __restrict dst,
    const size_t d
) {
    const size_t d_16 = (d / 16) * 16;
    for (size_t i = 0; i < d_16; i += 16) {
        const __m512 src_v = _mm512_loadu_ps(src + i);
        const __m256i dst_v = _mm512_cvtps_ph(src_v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_storeu_si256((__m256i*)(dst + i), dst_v);
    }

    if (d_16 != d) [[unlikely]] {
        const size_t leftovers = d - d_16;
        const __mmask16 mask = (1U << leftovers) - 1U;

        const __m512 src_v = _mm512_maskz_loadu_ps(mask, src + d_16);
        const __m256i dst_v = _mm512_cvtps_ph(src_v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_mask_storeu_epi16(dst + d_16, mask, dst_v);
    }
}

static inline float fp16_to_fp32(const uint16_t v) {
    const __m128i xi = _mm_set1_epi16(v);
    const __m128 xf = _mm_cvtph_ps(xi);
    return _mm_cvtss_f32(xf);
}


struct vec_f16x32 {
    static constexpr size_t SIMD_WIDTH = 32;

    using scalar_type = uint16_t;
    using simd_type = __m512i;

    static simd_type zero() {
        return _mm512_set1_ph(0);
    }

    static simd_type load(const scalar_type* const __restrict src) {
        return _mm512_loadu_ph(src);
    }

    static void store(scalar_type* const __restrict dst, const simd_type a) {
        _mm512_storeu_ph(dst, a);
    }

    static void store_as_f32(float* const __restrict dst, const simd_type a) {
        _mm512_storeu_ps(dst, _mm512_cvtph_ps(_mm512_extracti32x8_epi32(a, 0)));
        _mm512_storeu_ps(dst + 16, _mm512_cvtph_ps(_mm512_extracti32x8_epi32(a, 1)));
    }

    static simd_type max_value() {
        return _mm512_set1_ph(std::numeric_limits<float>::max());
    }

    static simd_type set1(const scalar_type v) {
        return _mm512_set1_epi16(v);
    }

    static simd_type from_i32(const int32_t v) {
        return _mm512_set1_ph(v);
    }

    static simd_type add(const simd_type a, const simd_type b) {
        return _mm512_add_ph(a, b);
    }

    static simd_type mul(const simd_type a, const simd_type b) {
        return _mm512_mul_ph(a, b);
    }

    static simd_type fmadd(const simd_type a, const simd_type b, const simd_type accum) {
        return _mm512_fmadd_ph(a, b, accum);
    }

    static simd_type fnmadd(const simd_type a, const simd_type b, const simd_type accum) {
        return _mm512_fnmadd_ph(a, b, accum);
    }

    static simd_type select(const __mmask32 comparison, const simd_type if_reset, const simd_type if_set) {
        return _mm512_mask_blend_ph(comparison, if_reset, if_set);
    }

    static __mmask32 compare_eq(const simd_type a, const simd_type b) {
        return _mm512_cmp_ph_mask(a, b, _CMP_EQ_OQ);
    }

    static __mmask32 compare_le(const simd_type a, const simd_type b) {
        return _mm512_cmp_ph_mask(a, b, _CMP_LE_OQ);
    }

    static simd_type max(const simd_type a, const simd_type b) {
        return _mm512_max_ph(a, b);
    }
};

struct vec_u16x32 {
    static constexpr size_t SIMD_WIDTH = 32;

    using scalar_type = uint16_t;
    using simd_type = __m512i;

    static simd_type zero() {
        return _mm512_set1_epi16(0);
    }

    static simd_type select(const __mmask32 comparison, const simd_type if_reset, const simd_type if_set) {
        return _mm512_mask_blend_epi16(comparison, if_reset, if_set);
    }

    static simd_type set1(const scalar_type v) {
        return _mm512_set1_epi16(v);
    }

    static void store(scalar_type* const __restrict dst, const simd_type a) {
        _mm512_storeu_si512(dst, a);
    }

    static void store_as_u32(uint32_t* const __restrict dst, const simd_type a) {
        _mm512_storeu_si512(dst, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(a, 0)));
        _mm512_storeu_si512(dst + 16, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(a, 1)));
    }
};

}  // namespace smalltopk
