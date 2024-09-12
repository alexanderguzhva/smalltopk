#pragma once

#include <arm_sve.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace smalltopk {

struct vec_f32 {
    using scalar_type = float;
    using simd_type = svfloat32_t;

    static simd_type add(const svbool_t mask, const simd_type a, const simd_type b) {
        return svadd_f32_x(mask, a, b);
    }

    static simd_type mul(const svbool_t mask, const simd_type a, const simd_type b) {
        return svmul_f32_x(mask, a, b);
    }

    static simd_type min(const svbool_t mask, const simd_type a, const simd_type b) {
        return svmin_f32_x(mask, a, b);
    }

    static simd_type max(const svbool_t mask, const simd_type a, const simd_type b) {
        return svmax_f32_x(mask, a, b);
    }

    static simd_type load(const svbool_t mask, const scalar_type* const __restrict src) {
        return svld1_f32(mask, src);
    }

    static void store(const svbool_t mask, scalar_type* const __restrict dst, const simd_type a) {
        svst1_f32(mask, dst, a);
    }

    static uint64_t width() {
        return svcntw();
    }

    static simd_type set1(const scalar_type v) {
        return svdup_n_f32(v);
    }

    static simd_type fmadd(const svbool_t mask, const simd_type a, const simd_type b, const simd_type accum) {
        return svmad_f32_x(mask, a, b, accum);
    }

    static simd_type fnmadd(const svbool_t mask, const simd_type a, const simd_type b, const simd_type accum) {
        return svmsb_f32_x(mask, a, b, accum);
    }

    static simd_type from_i32(const int32_t v) {
        return svdup_n_f32(v);
    }

    static simd_type max_value() {
        return svdup_n_f32(std::numeric_limits<scalar_type>::max());
    }

    static simd_type zero() {
        return svdup_n_f32(0);
    }

    static svbool_t compare_eq(const svbool_t mask, const simd_type a, const simd_type b) {
        return svcmpeq_f32(mask, a, b);
    }

    static svbool_t compare_le(const svbool_t mask, const simd_type a, const simd_type b) {
        return svcmple_f32(mask, a, b);
    }

    static simd_type select(const svbool_t mask, const simd_type if_reset, const simd_type if_set) {
        return svsel_f32(mask, if_set, if_reset);
    }

    static void store_as_f32(const svbool_t mask, float* const __restrict dst, const simd_type a) {
        store(mask, dst, a);
    }

    static simd_type dup_lane(const simd_type a, const uint16_t lane) {
        return svdup_lane_f32(a, lane);
    }

    static svbool_t pred_all() {
        return svptrue_b32();
    }
};


struct vec_f16 {
    using scalar_type = float16_t;
    using simd_type = svfloat16_t;

    static simd_type add(const svbool_t mask, const simd_type a, const simd_type b) {
        return svadd_f16_x(mask, a, b);
    }

    static simd_type mul(const svbool_t mask, const simd_type a, const simd_type b) {
        return svmul_f16_x(mask, a, b);
    }

    static simd_type max(const svbool_t mask, const simd_type a, const simd_type b) {
        return svmax_f16_x(mask, a, b);
    }

    static simd_type load(const svbool_t mask, const scalar_type* const __restrict src) {
        return svld1_f16(mask, src);
    }

    static void store(const svbool_t mask, scalar_type* const __restrict dst, const simd_type a) {
        svst1_f16(mask, dst, a);
    }

    static uint64_t width() {
        return svcnth();
    }

    static simd_type set1(const scalar_type v) {
        return svdup_n_f16(v);
    }

    static simd_type fmadd(const svbool_t mask, const simd_type a, const simd_type b, const simd_type accum) {
        return svmad_f16_x(mask, a, b, accum);
    }

    static simd_type fnmadd(const svbool_t mask, const simd_type a, const simd_type b, const simd_type accum) {
        return svmsb_f16_x(mask, a, b, accum);
    }

    static simd_type from_i32(const int32_t v) {
        return svdup_n_f16(v);
    }

    static simd_type max_value() {
        auto maxv = std::numeric_limits<float>::max();
        auto u = svdup_n_f16(maxv);
        return u;
//        return svdup_n_f16(std::numeric_limits<scalar_type>::max());
    }

    static simd_type zero() {
        return svdup_n_f16(0);
    }

    static svbool_t compare_eq(const svbool_t mask, const simd_type a, const simd_type b) {
        return svcmpeq_f16(mask, a, b);
    }

    static svbool_t compare_le(const svbool_t mask, const simd_type a, const simd_type b) {
        return svcmple_f16(mask, a, b);
    }

    static simd_type select(const svbool_t mask, const simd_type if_reset, const simd_type if_set) {
        return svsel_f16(mask, if_set, if_reset);
    }

    static void store_as_f32(const svbool_t mask, float* const __restrict dst, const simd_type a) {
        const svbool_t mask_lo = svunpklo_b(mask);
        const svbool_t mask_hi = svunpkhi_b(mask);

        const svfloat16_t cvt_fp16_lo = svzip1_f16(a, svdup_n_f16(0));
        const svfloat32_t cvt_lo = svcvt_f32_f16_x(mask_lo, cvt_fp16_lo);
        const svfloat16_t cvt_fp16_hi = svzip2_f16(a, svdup_n_f16(0));
        const svfloat32_t cvt_hi = svcvt_f32_f16_x(mask_hi, cvt_fp16_hi); 

        svst1_f32(mask_lo, dst, cvt_lo);
        svst1_f32(mask_hi, dst + svcntw(), cvt_hi);
    }

    static simd_type dup_lane(const simd_type a, const uint16_t lane) {
        return svdup_lane_f16(a, lane);
    }

    static svbool_t pred_all() {
        return svptrue_b16();
    }
};


struct vec_u32 {
    using scalar_type = uint32_t;
    using simd_type = svuint32_t;

    static simd_type zero() {
        return svdup_n_u32(0);
    }

    static simd_type select(const svbool_t mask, const simd_type if_reset, const simd_type if_set) {
        return svsel_u32(mask, if_set, if_reset);
    }

    static simd_type set1(const scalar_type v) {
        return svdup_n_u32(v);
    }

    static void store_as_u32(const svbool_t mask, uint32_t* const __restrict dst, const simd_type a) {
        svst1_u32(mask, dst, a);
    }

    static simd_type load(const svbool_t mask, const scalar_type* const __restrict src) {
        return svld1_u32(mask, src);
    }

    static void store(const svbool_t mask, scalar_type* const __restrict dst, const simd_type a) {
        svst1_u32(mask, dst, a);
    }

    static svbool_t pred_all() {
        return svptrue_b32();
    }
};


struct vec_u16 {
    using scalar_type = uint16_t;
    using simd_type = svuint16_t;

    static simd_type zero() {
        return svdup_n_u16(0);
    }

    static simd_type select(const svbool_t mask, const simd_type if_reset, const simd_type if_set) {
        return svsel_u16(mask, if_set, if_reset);
    }

    static simd_type set1(const scalar_type v) {
        return svdup_n_u16(v);
    }

    static void store_as_u32(const svbool_t mask, uint32_t* const __restrict dst, const simd_type a) {
        const svuint32_t a_lo = svunpklo_u32(a);
        const svuint32_t a_hi = svunpkhi_u32(a);
        const svbool_t mask_lo = svunpklo_b(mask);
        const svbool_t mask_hi = svunpkhi_b(mask);

        svst1_u32(mask_lo, dst, a_lo);
        svst1_u32(mask_hi, dst + svcntw(), a_hi);
    }

    static simd_type load(const svbool_t mask, const scalar_type* const __restrict src) {
        return svld1_u16(mask, src);
    }

    static void store(const svbool_t mask, scalar_type* const __restrict dst, const simd_type a) {
        svst1_u16(mask, dst, a);
    }

    static svbool_t pred_all() {
        return svptrue_b16();
    }
};

}  // namespace smalltopk
