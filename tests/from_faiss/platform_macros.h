// reworked from faiss

#pragma once


// Localized enablement of imprecise floating point operations
// You need to use all 3 macros to cover all compilers.
#if defined(_MSC_VER)
#define FAISS_PRAGMA_IMPRECISE_LOOP
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    __pragma(float_control(precise, off, push))
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_END __pragma(float_control(pop))
#elif defined(__clang__)
#if defined(__PPC__)
#define FAISS_PRAGMA_IMPRECISE_LOOP \
    _Pragma("clang loop vectorize_width(4) interleave_count(8)")
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("float_control(precise, off, push)")
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_END _Pragma("float_control(pop)")
#else
#define FAISS_PRAGMA_IMPRECISE_LOOP \
    _Pragma("clang loop vectorize(enable) interleave(enable)")

// clang-format off

// the following ifdef is needed, because old versions of clang (prior to 14)
// do not generate FMAs on x86 unless this pragma is used. On the other hand,
// ARM does not support the following pragma flag.
// TODO: find out how to enable FMAs on clang 10 and earlier.
#if defined(__x86_64__) && (defined(__clang_major__) && (__clang_major__ > 10))
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("float_control(precise, off, push)")
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_END _Pragma("float_control(pop)")
#else
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_END
#endif
#endif
#elif defined(__GNUC__)
// Unfortunately, GCC does not provide a pragma for detecting it.
// So, we have to stick to GNUC, which is defined by MANY compilers.
// This is why clang/icc needs to be checked first.

// todo: add __INTEL_COMPILER check for the classic ICC
// todo: add __INTEL_LLVM_COMPILER for ICX

#define FAISS_PRAGMA_IMPRECISE_LOOP
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("GCC push_options") \
    _Pragma("GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_END \
    _Pragma("GCC pop_options")
#else
#define FAISS_PRAGMA_IMPRECISE_LOOP
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_END
#endif
