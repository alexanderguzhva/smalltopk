#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace smalltopk {

struct vec_f32x16 {
    static constexpr size_t SIMD_WIDTH = 16;

    using scalar_type = float;
    using simd_type = __m512;

    static simd_type zero() {
        return _mm512_set1_ps(0);
    }

    static simd_type load(const scalar_type* const __restrict src) {
        return _mm512_loadu_ps(src);
    }

    static void store(scalar_type* const __restrict dst, const simd_type a) {
        _mm512_storeu_ps(dst, a);
    }

    static void store_as_f32(float* const __restrict dst, const simd_type a) {
        store(dst, a);
    }

    static simd_type max_value() {
        return _mm512_set1_ps(std::numeric_limits<scalar_type>::max());
    }

    static simd_type set1(const scalar_type v) {
        return _mm512_set1_ps(v);
    }

    static simd_type from_i32(const int32_t v) {
        return set1(static_cast<scalar_type>(v));
    }

    static simd_type add(const simd_type a, const simd_type b) {
        return _mm512_add_ps(a, b);
    }

    static simd_type mul(const simd_type a, const simd_type b) {
        return _mm512_mul_ps(a, b);
    }

    static simd_type fmadd(const simd_type a, const simd_type b, const simd_type accum) {
        return _mm512_fmadd_ps(a, b, accum);
    }

    static simd_type fnmadd(const simd_type a, const simd_type b, const simd_type accum) {
        return _mm512_fnmadd_ps(a, b, accum);
    }

    static simd_type select(const __mmask16 comparison, const simd_type if_reset, const simd_type if_set) {
        return _mm512_mask_blend_ps(comparison, if_reset, if_set);
    }

    static __mmask16 compare_eq(const simd_type a, const simd_type b) {
        return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
    }

    static __mmask16 compare_le(const simd_type a, const simd_type b) {
        return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ);
    }

    static simd_type min(const simd_type a, const simd_type b) {
        return _mm512_min_ps(a, b);
    }

    static simd_type max(const simd_type a, const simd_type b) {
        return _mm512_max_ps(a, b);
    }
};

struct vec_u16x16 {
    static constexpr size_t SIMD_WIDTH = 16;

    using scalar_type = uint16_t;
    using simd_type = __m256i;

    static simd_type zero() {
        return _mm256_set1_epi16(0);
    }

    static simd_type select(const __mmask16 comparison, const simd_type if_reset, const simd_type if_set) {
        return _mm256_mask_blend_epi16(comparison, if_reset, if_set);
    }

    static simd_type set1(const scalar_type v) {
        return _mm256_set1_epi16(v);
    }

    static void store(scalar_type* const __restrict dst, const simd_type a) {
        _mm256_storeu_si256((__m256i*)dst, a);
    }

    static void store_as_u32(uint32_t* const __restrict dst, const simd_type a) {
        _mm512_storeu_si512(dst, _mm512_cvtepu16_epi32(a));
    }
};


struct vec_u32x16 {
    static constexpr size_t SIMD_WIDTH = 16;

    using scalar_type = uint32_t;
    using simd_type = __m512i;

    static simd_type zero() {
        return _mm512_set1_epi16(0);
    }

    static simd_type select(const __mmask16 comparison, const simd_type if_reset, const simd_type if_set) {
        return _mm512_mask_blend_epi32(comparison, if_reset, if_set);
    }

    static simd_type set1(const scalar_type v) {
        return _mm512_set1_epi32(v);
    }

    static void store(scalar_type* const __restrict dst, const simd_type a) {
        _mm512_storeu_si512((__m256i*)dst, a);
    }

    static void store_as_u32(uint32_t* const __restrict dst, const simd_type a) {
        store(dst, a);
    }
};

}  // namespace smalltopk