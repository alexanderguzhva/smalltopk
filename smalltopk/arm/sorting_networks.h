#pragma once

#include <cstddef>
#include <cstdint>

#include <smalltopk/utils/macro_repeat_define.h>

namespace smalltopk {

// constructed using
// * https://bertdobbelaere.github.io/sorting_networks.html
// * https://github.com/bertdobbelaere/SorterHunter

// SIMD Sorting Network that does performs a partial sort.
//
// The purpose: merge N unsorted candidates into topk K sorted elements,
//   get updated top K sorted elements and discard leftover N candidates.
//
// * K + N elements are passed as an input and provided as an output.
// * First K input elements are assumed to be already sorted.
// * Next N input elements are not implied to be sorted.
// * First K output elements are sorted.
// * Next N output elements are not implied to be sorted.
//
// todo: these ones maybe unoptimal, reeval later.

#define SN_STEP(IDXA, IDXB) { func(distances_##IDXA, indices_##IDXA, distances_##IDXB, indices_##IDXB); }

#define GENERATE_INPUTS(NX) \
    typename DistancesEngineT::simd_type& __restrict distances_##NX, \
    typename IndicesEngineT::simd_type& __restrict indices_##NX, 


template<size_t K, size_t N>
struct PartialSortingNetwork {};

template<>
struct PartialSortingNetwork<1, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 9)
        Func func
    ) {
        // 8
        SN_STEP(2,3); SN_STEP(1,6); SN_STEP(1,8); SN_STEP(4,5); SN_STEP(1,7); 
        SN_STEP(0,4); SN_STEP(0,2); SN_STEP(0,1);
    }
};

template<>
struct PartialSortingNetwork<2, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 10)
        Func func
    ) {
        // 16
        SN_STEP(2,3); SN_STEP(8,9); SN_STEP(4,6); SN_STEP(7,8); SN_STEP(2,6); 
        SN_STEP(4,5); SN_STEP(2,7); SN_STEP(0,3); SN_STEP(7,8); SN_STEP(1,5); 
        SN_STEP(0,4); SN_STEP(0,9); SN_STEP(0,7); SN_STEP(1,2); SN_STEP(1,4); 
        SN_STEP(0,1);
    }
};

template<>
struct PartialSortingNetwork<3, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 11)
        Func func
    ) {
        // 19
        SN_STEP(3,4); SN_STEP(6,9); SN_STEP(5,10); SN_STEP(6,8); SN_STEP(3,7); 
        SN_STEP(0,6); SN_STEP(4,9); SN_STEP(3,5); SN_STEP(0,3); SN_STEP(4,10); 
        SN_STEP(7,8); SN_STEP(1,4); SN_STEP(5,6); SN_STEP(2,7); SN_STEP(1,2); 
        SN_STEP(3,5); SN_STEP(2,3); SN_STEP(1,5); SN_STEP(1,2);
    }
};

template<>
struct PartialSortingNetwork<4, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 12)
        Func func
    ) {
        // 25
        SN_STEP(8,10); SN_STEP(4,9); SN_STEP(4,7); SN_STEP(3,11); SN_STEP(3,4); 
        SN_STEP(5,6); SN_STEP(5,8); SN_STEP(5,9); SN_STEP(6,10); SN_STEP(0,10); 
        SN_STEP(2,4); SN_STEP(1,9); SN_STEP(6,8); SN_STEP(0,5); SN_STEP(7,8); 
        SN_STEP(1,7); SN_STEP(1,6); SN_STEP(1,2); SN_STEP(3,5); SN_STEP(5,7); 

        SN_STEP(5,6); SN_STEP(0,3); SN_STEP(2,5); SN_STEP(1,3); SN_STEP(2,3);
    }
};

template<>
struct PartialSortingNetwork<5, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 13)
        Func func
    ) {
        // 27
        SN_STEP(7,8); SN_STEP(10,12); SN_STEP(6,11); SN_STEP(11,12); SN_STEP(5,9); 
        SN_STEP(0,10); SN_STEP(8,9); SN_STEP(6,7); SN_STEP(5,6); SN_STEP(8,11); 
        SN_STEP(9,11); SN_STEP(7,10); SN_STEP(0,5); SN_STEP(9,12); SN_STEP(5,7); 
        SN_STEP(6,8); SN_STEP(1,6); SN_STEP(2,8); SN_STEP(6,9); SN_STEP(4,5); 

        SN_STEP(2,7); SN_STEP(3,10); SN_STEP(3,6); SN_STEP(1,4); SN_STEP(2,4);
        SN_STEP(3,7); SN_STEP(3,4);
    }
};

template<>
struct PartialSortingNetwork<6, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 14)
        Func func
    ) {
        // 31
        SN_STEP(12,13); SN_STEP(8,12); SN_STEP(6,11); SN_STEP(7,10); SN_STEP(7,9); 
        SN_STEP(10,12); SN_STEP(7,8); SN_STEP(3,12); SN_STEP(9,13); SN_STEP(4,13); 
        SN_STEP(10,11); SN_STEP(1,10); SN_STEP(0,6); SN_STEP(5,7); SN_STEP(0,5); 
        SN_STEP(2,8); SN_STEP(6,9); SN_STEP(3,9); SN_STEP(1,2); SN_STEP(4,11); 

        SN_STEP(4,8); SN_STEP(3,10); SN_STEP(3,4); SN_STEP(5,6); SN_STEP(1,5); 
        SN_STEP(2,6); SN_STEP(6,10); SN_STEP(4,6); SN_STEP(2,5); SN_STEP(3,5); 
        SN_STEP(4,5);
    }
};

template<>
struct PartialSortingNetwork<7, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 15)
        Func func
    ) {
        // 34
        SN_STEP(9,11); SN_STEP(7,10); SN_STEP(8,13); SN_STEP(12,14); SN_STEP(13,14); 
        SN_STEP(10,11); SN_STEP(9,12); SN_STEP(7,8); SN_STEP(8,12); SN_STEP(7,9); 
        SN_STEP(11,14); SN_STEP(8,9); SN_STEP(1,8); SN_STEP(3,13); SN_STEP(2,9); 
        SN_STEP(6,11); SN_STEP(5,10); SN_STEP(0,7); SN_STEP(6,8); SN_STEP(3,5); 

        SN_STEP(9,13); SN_STEP(5,9); SN_STEP(3,7); SN_STEP(4,12); SN_STEP(1,3); 
        SN_STEP(5,6); SN_STEP(4,8); SN_STEP(4,7); SN_STEP(2,4); SN_STEP(2,3); 
        SN_STEP(6,7); SN_STEP(4,5); SN_STEP(3,4); SN_STEP(5,6);
    }
};

template<>
struct PartialSortingNetwork<8, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 16)
        Func func
    ) {
        // 38
        SN_STEP(13,14); SN_STEP(10,12); SN_STEP(8,15); SN_STEP(10,13); SN_STEP(9,11); 
        SN_STEP(8,9); SN_STEP(11,15); SN_STEP(8,10); SN_STEP(9,13); SN_STEP(0,8); 
        SN_STEP(5,13); SN_STEP(12,14); SN_STEP(1,10); SN_STEP(6,9); SN_STEP(14,15); 
        SN_STEP(11,12); SN_STEP(4,14); SN_STEP(6,11); SN_STEP(2,12); SN_STEP(7,8); 

        SN_STEP(3,11); SN_STEP(4,10); SN_STEP(7,15); SN_STEP(1,7); SN_STEP(5,12); 
        SN_STEP(2,6); SN_STEP(3,7); SN_STEP(4,7); SN_STEP(5,10); SN_STEP(1,2); 
        SN_STEP(6,10); SN_STEP(5,11); SN_STEP(5,6); SN_STEP(4,5); SN_STEP(2,3); 
        SN_STEP(6,7); SN_STEP(5,6); SN_STEP(3,4);
    }
};

template<>
struct PartialSortingNetwork<9, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 17)
        Func func
    ) {
        // 40
        SN_STEP(10,11); SN_STEP(9,13); SN_STEP(11,13); SN_STEP(12,15); SN_STEP(14,16); 
        SN_STEP(15,16); SN_STEP(11,15); SN_STEP(9,10); SN_STEP(13,16); SN_STEP(12,14); 
        SN_STEP(13,15); SN_STEP(9,12); SN_STEP(6,13); SN_STEP(10,14); SN_STEP(0,9); 
        SN_STEP(7,16); SN_STEP(2,11); SN_STEP(5,14); SN_STEP(3,15); SN_STEP(5,11); 

        SN_STEP(2,9); SN_STEP(8,9); SN_STEP(10,12); SN_STEP(1,10); SN_STEP(4,12); 
        SN_STEP(6,10); SN_STEP(3,10); SN_STEP(4,8); SN_STEP(6,8); SN_STEP(3,5); 
        SN_STEP(5,8); SN_STEP(1,2); SN_STEP(7,12); SN_STEP(7,11); SN_STEP(3,6); 
        SN_STEP(5,6); SN_STEP(2,4); SN_STEP(3,4); SN_STEP(7,10); SN_STEP(7,8);
    }
};

template<>
struct PartialSortingNetwork<10, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 18)
        Func func
    ) {
        // 42
        SN_STEP(10,11); SN_STEP(13,14); SN_STEP(16,17); SN_STEP(12,15); SN_STEP(11,17); 
        SN_STEP(14,15); SN_STEP(12,16); SN_STEP(8,11); SN_STEP(8,14); SN_STEP(10,13); 
        SN_STEP(13,16); SN_STEP(15,17); SN_STEP(14,16); SN_STEP(10,12); SN_STEP(8,12); 
        SN_STEP(0,10); SN_STEP(12,13); SN_STEP(14,15); SN_STEP(9,10); SN_STEP(2,17); 
        
        SN_STEP(8,12); SN_STEP(8,9); SN_STEP(15,16); SN_STEP(4,15); SN_STEP(7,12); 
        SN_STEP(13,14); SN_STEP(2,13); SN_STEP(5,14); SN_STEP(2,9); SN_STEP(3,16); 
        SN_STEP(4,9); SN_STEP(3,7); SN_STEP(6,13); SN_STEP(1,8); SN_STEP(3,8); 
        SN_STEP(5,8); SN_STEP(7,8); SN_STEP(2,3); SN_STEP(4,5); SN_STEP(6,9); 
        
        SN_STEP(8,9); SN_STEP(6,7);
    }
};

template<>
struct PartialSortingNetwork<11, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 19)
        Func func
    ) {
        // 44
        SN_STEP(12,16); SN_STEP(14,17); SN_STEP(11,18); SN_STEP(12,14); SN_STEP(13,15); 
        SN_STEP(11,13); SN_STEP(17,18); SN_STEP(15,16); SN_STEP(15,17); SN_STEP(13,14); 
        SN_STEP(13,15); SN_STEP(16,18); SN_STEP(3,18); SN_STEP(9,13); SN_STEP(16,17); 
        SN_STEP(14,16); SN_STEP(11,12); SN_STEP(12,15); SN_STEP(14,15); SN_STEP(0,12); 
        
        SN_STEP(10,11); SN_STEP(7,14); SN_STEP(6,15); SN_STEP(3,7); SN_STEP(9,12);
        SN_STEP(8,12); SN_STEP(1,9); SN_STEP(16,17); SN_STEP(2,10); SN_STEP(4,17); 
        SN_STEP(4,8); SN_STEP(3,9); SN_STEP(5,16); SN_STEP(6,10); SN_STEP(5,9); 
        SN_STEP(0,2); SN_STEP(4,6); SN_STEP(3,4); SN_STEP(8,10); SN_STEP(5,6); 
        
        SN_STEP(7,9); SN_STEP(7,8); SN_STEP(9,10); SN_STEP(1,2);
    }
};

template<>
struct PartialSortingNetwork<12, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 20)
        Func func
    ) {
        // 47
        SN_STEP(13,19); SN_STEP(14,15); SN_STEP(12,18); SN_STEP(11,14); SN_STEP(10,18); 
        SN_STEP(11,12); SN_STEP(10,19); SN_STEP(16,17); SN_STEP(13,16); SN_STEP(11,13); 
        SN_STEP(15,17); SN_STEP(10,15); SN_STEP(17,19); SN_STEP(12,16); SN_STEP(12,13); 
        SN_STEP(15,17); SN_STEP(10,13); SN_STEP(10,12); SN_STEP(9,12); SN_STEP(8,13); 
        
        SN_STEP(15,16); SN_STEP(8,15); SN_STEP(0,11); SN_STEP(4,19); SN_STEP(10,11); 
        SN_STEP(1,10); SN_STEP(16,17); SN_STEP(7,15); SN_STEP(2,11); SN_STEP(6,16); 
        SN_STEP(9,10); SN_STEP(3,9); SN_STEP(2,8); SN_STEP(4,8); SN_STEP(5,17); 
        SN_STEP(6,11); SN_STEP(7,9); SN_STEP(5,10); SN_STEP(4,6); SN_STEP(9,10); 
        
        SN_STEP(5,7); SN_STEP(8,11); SN_STEP(8,9); SN_STEP(6,7); SN_STEP(4,5); 
        SN_STEP(10,11); SN_STEP(2,3)
    }
};

template<>
struct PartialSortingNetwork<13, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 21)
        Func func
    ) {
        // 50
        SN_STEP(13,18); SN_STEP(14,16); SN_STEP(17,19); SN_STEP(16,18); SN_STEP(15,20); 
        SN_STEP(14,17); SN_STEP(13,15); SN_STEP(15,17); SN_STEP(19,20); SN_STEP(18,20); 
        SN_STEP(17,18); SN_STEP(16,19); SN_STEP(18,19); SN_STEP(13,14); SN_STEP(17,18); 
        SN_STEP(11,16); SN_STEP(7,13); SN_STEP(13,18); SN_STEP(5,20); SN_STEP(11,15); 
        
        SN_STEP(14,15); SN_STEP(15,17); SN_STEP(0,7); SN_STEP(11,14); SN_STEP(2,14); 
        SN_STEP(3,11); SN_STEP(6,19); SN_STEP(8,17); SN_STEP(4,8); SN_STEP(1,15); 
        SN_STEP(12,13); SN_STEP(9,15); SN_STEP(5,9); SN_STEP(1,3); SN_STEP(10,14); 
        SN_STEP(6,10); SN_STEP(11,12); SN_STEP(8,11); SN_STEP(4,7); SN_STEP(10,11); 
        
        SN_STEP(5,8); SN_STEP(9,12); SN_STEP(6,7); SN_STEP(11,12); SN_STEP(5,6); 
        SN_STEP(9,10); SN_STEP(7,8); SN_STEP(2,4); SN_STEP(3,4); SN_STEP(1,2);
    }
};

template<>
struct PartialSortingNetwork<14, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 22)
        Func func
    ) {
        // 52
        SN_STEP(14,16); SN_STEP(17,18); SN_STEP(19,21); SN_STEP(14,17); SN_STEP(15,20); 
        SN_STEP(20,21); SN_STEP(10,21); SN_STEP(16,18); SN_STEP(12,16); SN_STEP(10,18); 
        SN_STEP(12,20); SN_STEP(15,19); SN_STEP(17,19); SN_STEP(12,17); SN_STEP(10,19); 
        SN_STEP(14,15); SN_STEP(11,17); SN_STEP(10,20); SN_STEP(19,20); SN_STEP(4,19); 
        
        SN_STEP(13,14); SN_STEP(12,15); SN_STEP(11,15); SN_STEP(10,15); SN_STEP(6,18); 
        SN_STEP(0,13); SN_STEP(12,13); SN_STEP(2,10); SN_STEP(1,12); SN_STEP(9,15); 
        SN_STEP(5,12); SN_STEP(9,12); SN_STEP(7,20); SN_STEP(5,9); SN_STEP(3,11); 
        SN_STEP(7,11); SN_STEP(8,19); SN_STEP(2,13); SN_STEP(3,5); SN_STEP(2,3); 
        
        SN_STEP(11,12); SN_STEP(6,10); SN_STEP(8,13); SN_STEP(4,8); SN_STEP(7,9); 
        SN_STEP(6,8); SN_STEP(10,13); SN_STEP(8,9); SN_STEP(12,13); SN_STEP(6,7);
        SN_STEP(4,5); SN_STEP(10,11)
    }
};

template<>
struct PartialSortingNetwork<15, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 23)
        Func func
    ) {
        // 55
        SN_STEP(18,19); SN_STEP(21,22); SN_STEP(16,17); SN_STEP(18,21); SN_STEP(16,21); 
        SN_STEP(14,21); SN_STEP(19,22); SN_STEP(15,20); SN_STEP(17,20); SN_STEP(16,18); 
        SN_STEP(17,19); SN_STEP(0,15); SN_STEP(20,22); SN_STEP(17,18); SN_STEP(14,15); 
        SN_STEP(19,20); SN_STEP(15,20); SN_STEP(15,19); SN_STEP(14,18); SN_STEP(15,18); 
        
        SN_STEP(0,16); SN_STEP(8,20); SN_STEP(14,16); SN_STEP(16,17); SN_STEP(13,16); 
        SN_STEP(10,18); SN_STEP(2,10); SN_STEP(4,17); SN_STEP(1,13); SN_STEP(9,19); 
        SN_STEP(11,15); SN_STEP(6,14); SN_STEP(3,11); SN_STEP(9,13); SN_STEP(12,17); 
        SN_STEP(7,22); SN_STEP(7,11); SN_STEP(5,9); SN_STEP(10,14); SN_STEP(9,13); 
        
        SN_STEP(2,6); SN_STEP(8,12); SN_STEP(3,5); SN_STEP(8,10); SN_STEP(11,13); 
        SN_STEP(12,14); SN_STEP(7,9); SN_STEP(7,8); SN_STEP(4,6); SN_STEP(11,12); 
        SN_STEP(5,6); SN_STEP(9,10); SN_STEP(1,2); SN_STEP(3,4); SN_STEP(13,14);
    }
};

template<>
struct PartialSortingNetwork<16, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 24)
        Func func
    ) {
        // 59
        SN_STEP(16,23); SN_STEP(18,22); SN_STEP(17,20); SN_STEP(22,23); SN_STEP(19,21); 
        SN_STEP(16,18); SN_STEP(14,22); SN_STEP(17,19); SN_STEP(20,21); SN_STEP(21,23); 
        SN_STEP(18,20); SN_STEP(14,19); SN_STEP(16,17); SN_STEP(15,20); SN_STEP(15,21); 
        SN_STEP(15,19); SN_STEP(19,21); SN_STEP(4,23); SN_STEP(17,18); SN_STEP(14,18); 

        SN_STEP(5,21); SN_STEP(14,17); SN_STEP(8,23); SN_STEP(13,17); SN_STEP(0,16); 
        SN_STEP(15,18); SN_STEP(15,16); SN_STEP(14,15); SN_STEP(2,15); SN_STEP(6,15); 
        SN_STEP(11,18); SN_STEP(8,16); SN_STEP(1,14); SN_STEP(4,8); SN_STEP(9,21); 
        SN_STEP(12,16); SN_STEP(10,19); SN_STEP(6,10); SN_STEP(7,11); SN_STEP(10,15); 
        
        SN_STEP(13,14); SN_STEP(9,14); SN_STEP(11,14); SN_STEP(3,13); SN_STEP(12,15); 
        SN_STEP(4,6); SN_STEP(8,10); SN_STEP(7,13); SN_STEP(5,9); SN_STEP(9,13); 
        SN_STEP(5,7); SN_STEP(11,13); SN_STEP(10,11); SN_STEP(2,3); SN_STEP(4,5); 
        SN_STEP(12,13); SN_STEP(6,7); SN_STEP(14,15); SN_STEP(8,9);
    }
};

template<>
struct PartialSortingNetwork<17, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 25)
        Func func
    ) {
        // 60
        SN_STEP(16,22); SN_STEP(19,21); SN_STEP(16,17); SN_STEP(18,24); SN_STEP(20,23); 
        SN_STEP(21,23); SN_STEP(16,18); SN_STEP(17,24); SN_STEP(18,21); SN_STEP(19,20); 
        SN_STEP(23,24); SN_STEP(17,20); SN_STEP(13,23); SN_STEP(16,19); SN_STEP(17,18); 
        SN_STEP(20,21); SN_STEP(13,20); SN_STEP(20,21); SN_STEP(7,20); SN_STEP(18,19); 

        SN_STEP(1,24); SN_STEP(13,19); SN_STEP(17,18); SN_STEP(0,16); SN_STEP(12,19); 
        SN_STEP(9,24); SN_STEP(5,13); SN_STEP(9,13); SN_STEP(2,18); SN_STEP(3,17); 
        SN_STEP(4,12); SN_STEP(4,16); SN_STEP(8,16); SN_STEP(14,18); SN_STEP(6,21); 
        SN_STEP(10,14); SN_STEP(2,4); SN_STEP(1,3); SN_STEP(7,17); SN_STEP(11,17); 

        SN_STEP(12,16); SN_STEP(15,20); SN_STEP(6,10); SN_STEP(3,5); SN_STEP(9,11); 
        SN_STEP(5,7); SN_STEP(6,8); SN_STEP(10,12); SN_STEP(11,12); SN_STEP(15,17); 
        SN_STEP(14,21); SN_STEP(14,16); SN_STEP(13,15); SN_STEP(9,10); SN_STEP(5,6); 
        SN_STEP(3,4); SN_STEP(7,8); SN_STEP(1,2); SN_STEP(13,14); SN_STEP(15,16);
    }
};

template<>
struct PartialSortingNetwork<18, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 26)
        Func func
    ) {
        // 63
        SN_STEP(17,23); SN_STEP(21,22); SN_STEP(17,19); SN_STEP(24,25); SN_STEP(18,20); 
        SN_STEP(18,24); SN_STEP(17,21); SN_STEP(19,22); SN_STEP(19,21); SN_STEP(16,25); 
        SN_STEP(16,20); SN_STEP(16,24); SN_STEP(15,21); SN_STEP(15,24); SN_STEP(17,18); 
        SN_STEP(20,22); SN_STEP(1,17); SN_STEP(9,17); SN_STEP(16,19); SN_STEP(14,20); 
        
        SN_STEP(14,19); SN_STEP(16,18); SN_STEP(15,18); SN_STEP(14,16); SN_STEP(19,24); 
        SN_STEP(16,18); SN_STEP(2,16); SN_STEP(3,15); SN_STEP(0,14); SN_STEP(13,18); 
        SN_STEP(15,24); SN_STEP(8,16); SN_STEP(3,9); SN_STEP(13,17); SN_STEP(11,24); 
        SN_STEP(10,22); SN_STEP(5,13); SN_STEP(4,19); SN_STEP(6,14); SN_STEP(4,6);
        
        SN_STEP(2,4); SN_STEP(10,19); SN_STEP(12,19); SN_STEP(0,1); SN_STEP(10,14); 
        SN_STEP(12,16); SN_STEP(12,14); SN_STEP(5,9); SN_STEP(7,15); SN_STEP(4,5); 
        SN_STEP(7,9); SN_STEP(11,15); SN_STEP(8,10); SN_STEP(6,8); SN_STEP(8,9); 
        SN_STEP(15,17); SN_STEP(11,13); SN_STEP(16,17); SN_STEP(14,15); SN_STEP(10,11); 

        SN_STEP(6,7); SN_STEP(2,3); SN_STEP(12,13);

    }
};

template<>
struct PartialSortingNetwork<19, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 27)
        Func func
    ) {
        // 65
        SN_STEP(20,23); SN_STEP(21,25); SN_STEP(17,25); SN_STEP(17,23); SN_STEP(22,24); 
        SN_STEP(19,26); SN_STEP(24,26); SN_STEP(19,20); SN_STEP(15,23); SN_STEP(17,24); 
        SN_STEP(21,22); SN_STEP(20,22); SN_STEP(15,26); SN_STEP(15,24); SN_STEP(19,21); 
        SN_STEP(22,24); SN_STEP(17,20); SN_STEP(5,26); SN_STEP(11,26); SN_STEP(15,22); 
        
        SN_STEP(20,21); SN_STEP(0,19); SN_STEP(15,21); SN_STEP(1,15); SN_STEP(17,19); 
        SN_STEP(19,20); SN_STEP(11,19); SN_STEP(1,11); SN_STEP(5,15); SN_STEP(8,20); 
        SN_STEP(12,24); SN_STEP(13,22); SN_STEP(7,13); SN_STEP(3,11); SN_STEP(2,17); 
        SN_STEP(14,21); SN_STEP(14,17); SN_STEP(7,11); SN_STEP(4,12); SN_STEP(9,15); 
        
        SN_STEP(18,19); SN_STEP(6,14); SN_STEP(10,17); SN_STEP(4,8); SN_STEP(16,20); 
        SN_STEP(13,15); SN_STEP(4,6); SN_STEP(12,16); SN_STEP(10,14); SN_STEP(8,10); 
        SN_STEP(13,18); SN_STEP(5,7); SN_STEP(12,14); SN_STEP(1,2); SN_STEP(15,18); 
        SN_STEP(16,17); SN_STEP(15,16); SN_STEP(13,14); SN_STEP(9,11); SN_STEP(7,8); 
        
        SN_STEP(9,10); SN_STEP(5,6); SN_STEP(11,12); SN_STEP(3,4); SN_STEP(17,18);
    }
};

template<>
struct PartialSortingNetwork<20, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 28)
        Func func
    ) {
        // 67
        SN_STEP(19,21); SN_STEP(22,25); SN_STEP(20,24); SN_STEP(19,23); SN_STEP(26,27); 
        SN_STEP(24,25); SN_STEP(19,26); SN_STEP(20,22); SN_STEP(19,20); SN_STEP(23,27); 
        SN_STEP(22,26); SN_STEP(20,22); SN_STEP(23,24); SN_STEP(22,23); SN_STEP(25,27); 
        SN_STEP(20,22); SN_STEP(24,25); SN_STEP(14,25); SN_STEP(4,27); SN_STEP(24,26); 
        
        SN_STEP(3,19); SN_STEP(14,26); SN_STEP(9,22); SN_STEP(23,24); SN_STEP(10,20); 
        SN_STEP(1,9); SN_STEP(0,10); SN_STEP(11,19); SN_STEP(5,26); SN_STEP(8,23); 
        SN_STEP(7,24); SN_STEP(8,10); SN_STEP(14,20); SN_STEP(16,23); SN_STEP(17,22); 
        SN_STEP(2,8); SN_STEP(12,27); SN_STEP(6,14); SN_STEP(12,16); SN_STEP(12,14); 
        
        SN_STEP(15,24); SN_STEP(15,19); SN_STEP(6,8); SN_STEP(18,20); SN_STEP(7,11); 
        SN_STEP(1,3); SN_STEP(16,18); SN_STEP(5,9); SN_STEP(9,11); SN_STEP(5,7); 
        SN_STEP(13,26); SN_STEP(13,17); SN_STEP(4,10); SN_STEP(13,15); SN_STEP(4,6); 
        SN_STEP(12,13); SN_STEP(2,3); SN_STEP(6,7); SN_STEP(17,19); SN_STEP(0,1); 
        
        SN_STEP(18,19); SN_STEP(8,10); SN_STEP(16,17); SN_STEP(14,15); SN_STEP(10,11); 
        SN_STEP(4,5); SN_STEP(8,9);
    }
};

template<>
struct PartialSortingNetwork<21, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 29)
        Func func
    ) {
        // 69
        SN_STEP(23,28); SN_STEP(21,25); SN_STEP(22,27); SN_STEP(25,28); SN_STEP(24,26); 
        SN_STEP(26,27); SN_STEP(21,23); SN_STEP(27,28); SN_STEP(5,28); SN_STEP(23,26); 
        SN_STEP(22,24); SN_STEP(24,25); SN_STEP(23,24); SN_STEP(21,22); SN_STEP(19,25); 
        SN_STEP(19,26); SN_STEP(22,23); SN_STEP(19,27); SN_STEP(1,22); SN_STEP(23,24); 
        
        SN_STEP(10,23); SN_STEP(19,24); SN_STEP(19,22); SN_STEP(0,21); SN_STEP(26,27); 
        SN_STEP(6,27); SN_STEP(7,26); SN_STEP(14,27); SN_STEP(8,21); SN_STEP(16,24); 
        SN_STEP(4,16); SN_STEP(2,10); SN_STEP(3,19); SN_STEP(12,21); SN_STEP(13,28); 
        SN_STEP(20,21); SN_STEP(11,19); SN_STEP(7,11); SN_STEP(4,8); SN_STEP(12,16); 
        
        SN_STEP(9,22); SN_STEP(6,10); SN_STEP(6,8); SN_STEP(18,23); SN_STEP(14,18); 
        SN_STEP(17,22); SN_STEP(15,26); SN_STEP(5,9); SN_STEP(16,20); SN_STEP(13,17); 
        SN_STEP(15,19); SN_STEP(5,7); SN_STEP(7,8); SN_STEP(9,11); SN_STEP(17,19); 
        SN_STEP(18,20); SN_STEP(1,4); SN_STEP(14,16); SN_STEP(13,15); SN_STEP(2,4); 
        
        SN_STEP(5,6); SN_STEP(10,12); SN_STEP(15,16); SN_STEP(9,10); SN_STEP(13,14); 
        SN_STEP(19,20); SN_STEP(11,12); SN_STEP(3,4); SN_STEP(17,18);
    }
};

template<>
struct PartialSortingNetwork<22, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 30)
        Func func
    ) {
        // 73
        SN_STEP(21,22); SN_STEP(21,25); SN_STEP(28,29); SN_STEP(23,26); SN_STEP(24,27); 
        SN_STEP(25,27); SN_STEP(24,28); SN_STEP(21,23); SN_STEP(23,28); SN_STEP(21,24); 
        SN_STEP(1,21); SN_STEP(26,29); SN_STEP(25,26); SN_STEP(23,24); SN_STEP(23,25); 
        SN_STEP(24,25); SN_STEP(5,24); SN_STEP(27,29); SN_STEP(19,24); SN_STEP(26,28); 
        
        SN_STEP(27,28); SN_STEP(4,23); SN_STEP(14,29); SN_STEP(26,27); SN_STEP(8,27); 
        SN_STEP(8,23); SN_STEP(12,23); SN_STEP(18,26); SN_STEP(17,28); SN_STEP(6,14); 
        SN_STEP(16,27); SN_STEP(0,8); SN_STEP(18,25); SN_STEP(15,25); SN_STEP(17,21); 
        SN_STEP(13,19); SN_STEP(21,25); SN_STEP(19,21); SN_STEP(2,18); SN_STEP(6,18); 
        
        SN_STEP(9,19); SN_STEP(20,23); SN_STEP(16,20); SN_STEP(10,18); SN_STEP(11,17); 
        SN_STEP(15,17); SN_STEP(17,19); SN_STEP(9,13); SN_STEP(14,18); SN_STEP(7,15); 
        SN_STEP(0,4); SN_STEP(3,11); SN_STEP(14,16); SN_STEP(7,11); SN_STEP(10,12); 
        SN_STEP(2,4); SN_STEP(9,11); SN_STEP(3,5); SN_STEP(18,20); SN_STEP(16,17); 
        
        SN_STEP(13,15); SN_STEP(5,7); SN_STEP(4,5); SN_STEP(0,1); SN_STEP(12,13); 
        SN_STEP(2,3); SN_STEP(10,11); SN_STEP(20,21); SN_STEP(6,8); SN_STEP(18,19); 
        SN_STEP(6,7); SN_STEP(14,15); SN_STEP(8,9);
    }
};

template<>
struct PartialSortingNetwork<23, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 31)
        Func func
    ) {
        // 75
        SN_STEP(25,26); SN_STEP(23,29); SN_STEP(26,29); SN_STEP(24,30); SN_STEP(27,28); 
        SN_STEP(28,30); SN_STEP(23,25); SN_STEP(25,28); SN_STEP(24,27); SN_STEP(23,24); 
        SN_STEP(29,30); SN_STEP(19,29); SN_STEP(26,27); SN_STEP(8,23); SN_STEP(14,23); 
        SN_STEP(24,25); SN_STEP(27,28); SN_STEP(19,27); SN_STEP(27,28); SN_STEP(25,26); 
        
        SN_STEP(13,27); SN_STEP(24,25); SN_STEP(7,30); SN_STEP(9,24); SN_STEP(1,9); 
        SN_STEP(22,23); SN_STEP(17,27); SN_STEP(21,24); SN_STEP(20,25); SN_STEP(19,26); 
        SN_STEP(18,20); SN_STEP(11,19); SN_STEP(3,11); SN_STEP(7,11); SN_STEP(0,8); 
        SN_STEP(10,18); SN_STEP(4,26); SN_STEP(6,28); SN_STEP(20,26); SN_STEP(12,20); 
        
        SN_STEP(4,8); SN_STEP(3,9); SN_STEP(5,13); SN_STEP(2,10); SN_STEP(15,30); 
        SN_STEP(16,28); SN_STEP(16,22); SN_STEP(16,18); SN_STEP(6,10); SN_STEP(2,4); 
        SN_STEP(12,14); SN_STEP(20,22); SN_STEP(10,12); SN_STEP(15,19); SN_STEP(5,9); 
        SN_STEP(13,21); SN_STEP(18,20); SN_STEP(11,13); SN_STEP(6,8); SN_STEP(15,21); 
        
        SN_STEP(3,4); SN_STEP(7,9); SN_STEP(7,8); SN_STEP(17,21); SN_STEP(14,16); 
        SN_STEP(15,16); SN_STEP(1,2); SN_STEP(13,14); SN_STEP(17,18); SN_STEP(19,21); 
        SN_STEP(9,10); SN_STEP(19,20); SN_STEP(11,12); SN_STEP(5,6); SN_STEP(21,22);
    }
};

template<>
struct PartialSortingNetwork<24, 8> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 32)
        Func func
    ) {
        // 79
        SN_STEP(29,31); SN_STEP(25,28); SN_STEP(25,29); SN_STEP(22,29); SN_STEP(26,27); 
        SN_STEP(27,28); SN_STEP(24,30); SN_STEP(30,31); SN_STEP(27,30); SN_STEP(24,26); 
        SN_STEP(24,25); SN_STEP(22,26); SN_STEP(22,27); SN_STEP(15,24); SN_STEP(22,25); 
        SN_STEP(28,31); SN_STEP(26,28); SN_STEP(1,22); SN_STEP(9,22); SN_STEP(25,27); 
        
        SN_STEP(28,30); SN_STEP(26,28); SN_STEP(26,27); SN_STEP(19,27); SN_STEP(8,15); 
        SN_STEP(10,25); SN_STEP(18,26); SN_STEP(7,31); SN_STEP(4,19); SN_STEP(6,30); 
        SN_STEP(14,30); SN_STEP(17,25); SN_STEP(16,22); SN_STEP(0,8); SN_STEP(21,30); 
        SN_STEP(20,28); SN_STEP(3,18); SN_STEP(11,18); SN_STEP(22,31); SN_STEP(20,26); 
        
        SN_STEP(5,20); SN_STEP(19,22); SN_STEP(2,10); SN_STEP(23,24); SN_STEP(21,25); 
        SN_STEP(20,21); SN_STEP(12,19); SN_STEP(3,9); SN_STEP(6,10); SN_STEP(12,15); 
        SN_STEP(10,12); SN_STEP(13,20); SN_STEP(13,16); SN_STEP(22,23); SN_STEP(21,23); 
        SN_STEP(18,22); SN_STEP(7,11); SN_STEP(14,17); SN_STEP(4,8); SN_STEP(2,4); 
        
        SN_STEP(17,19); SN_STEP(20,22); SN_STEP(5,9); SN_STEP(3,4); SN_STEP(11,13); 
        SN_STEP(6,8); SN_STEP(7,9); SN_STEP(1,2); SN_STEP(16,18); SN_STEP(9,10); 
        SN_STEP(21,22); SN_STEP(14,15); SN_STEP(5,6); SN_STEP(11,12); SN_STEP(13,14); 
        SN_STEP(15,16); SN_STEP(7,8); SN_STEP(19,20); SN_STEP(17,18);
    }
};


template<>
struct PartialSortingNetwork<1, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 7)
        Func func
    ) {
        // 6
        SN_STEP(4, 5); SN_STEP(0, 1); SN_STEP(0, 4); SN_STEP(3, 6); SN_STEP(2, 3); 
        SN_STEP(0, 2); 
    }
};

template<>
struct PartialSortingNetwork<2, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 8)
        Func func
    ) {
        // 12
        SN_STEP(2, 4); SN_STEP(3, 5); SN_STEP(6, 7); SN_STEP(1, 5); SN_STEP(0, 4); 
        SN_STEP(0, 3); SN_STEP(2, 6); SN_STEP(6, 7); SN_STEP(0, 6); SN_STEP(1, 2); 
        SN_STEP(1, 3); SN_STEP(0, 1); 
    }
};

template<>
struct PartialSortingNetwork<3, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 9)
        Func func
    ) {
        // 14
        SN_STEP(6, 7); SN_STEP(4, 8); SN_STEP(3, 5); SN_STEP(0, 3); SN_STEP(4, 6); 
        SN_STEP(0, 4); SN_STEP(5, 8); SN_STEP(5, 7); SN_STEP(2, 3); SN_STEP(1, 5); 
        SN_STEP(2, 6); SN_STEP(1, 4); SN_STEP(2, 4); SN_STEP(1, 2); 
    }
};

template<>
struct PartialSortingNetwork<4, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 10)
        Func func
    ) {
        // 19
        SN_STEP(5, 7); SN_STEP(4, 6); SN_STEP(4, 5); SN_STEP(8, 9); SN_STEP(6, 7); 
        SN_STEP(4, 8); SN_STEP(5, 9); SN_STEP(1, 7); SN_STEP(5, 8); SN_STEP(1, 5); 
        SN_STEP(0, 6); SN_STEP(2, 6); SN_STEP(3, 9); SN_STEP(0, 4); SN_STEP(3, 4); 
        SN_STEP(2, 8); SN_STEP(1, 3); SN_STEP(2, 5); SN_STEP(2, 3); 
    }
};

template<>
struct PartialSortingNetwork<5, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 11)
        Func func
    ) {
        // 21
        SN_STEP(7, 10); SN_STEP(6, 9); SN_STEP(6, 8); SN_STEP(5, 7); SN_STEP(8, 10); 
        SN_STEP(7, 9); SN_STEP(9, 10); SN_STEP(7, 8); SN_STEP(5, 6); SN_STEP(1, 7); 
        SN_STEP(3, 9); SN_STEP(4, 5); SN_STEP(2, 8); SN_STEP(0, 6); SN_STEP(3, 7); 
        SN_STEP(0, 4); SN_STEP(1, 4); SN_STEP(2, 6); SN_STEP(2, 4); SN_STEP(3, 6); 

        SN_STEP(3, 4); 
    }
};

template<>
struct PartialSortingNetwork<6, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 12)
        Func func
    ) {
        // 24
        SN_STEP(8, 10); SN_STEP(6, 11); SN_STEP(10, 11); SN_STEP(6, 8); SN_STEP(6, 7); 
        SN_STEP(9, 10); SN_STEP(5, 6); SN_STEP(2, 11); SN_STEP(7, 10); SN_STEP(0, 9); 
        SN_STEP(7, 9); SN_STEP(1, 8); SN_STEP(3, 9); SN_STEP(4, 10); SN_STEP(0, 5); 
        SN_STEP(5, 7); SN_STEP(2, 7); SN_STEP(4, 8); SN_STEP(1, 5); SN_STEP(3, 4); 

        SN_STEP(2, 5); SN_STEP(4, 7); SN_STEP(3, 5); SN_STEP(4, 5); 
    }
};

template<>
struct PartialSortingNetwork<7, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 13)
        Func func
    ) {
        // 26
        SN_STEP(9, 11); SN_STEP(8, 12); SN_STEP(7, 10); SN_STEP(7, 9); SN_STEP(6, 7); 
        SN_STEP(6, 8); SN_STEP(10, 11); SN_STEP(10, 12); SN_STEP(0, 6); SN_STEP(2, 8); 
        SN_STEP(3, 12); SN_STEP(9, 10); SN_STEP(1, 9); SN_STEP(3, 9); SN_STEP(4, 10); 
        SN_STEP(2, 6); SN_STEP(4, 6); SN_STEP(8, 11); SN_STEP(6, 8); SN_STEP(3, 6); 

        SN_STEP(5, 9); SN_STEP(5, 8); SN_STEP(1, 2); SN_STEP(2, 4); SN_STEP(3, 4); 
        SN_STEP(5, 6); 
    }
};

template<>
struct PartialSortingNetwork<8, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 14)
        Func func
    ) {
        // 29
        SN_STEP(10, 13); SN_STEP(8, 12); SN_STEP(12, 13); SN_STEP(9, 11); SN_STEP(8, 10); 
        SN_STEP(8, 9); SN_STEP(11, 13); SN_STEP(11, 12); SN_STEP(3, 8); SN_STEP(0, 11); 
        SN_STEP(0, 3); SN_STEP(9, 10); SN_STEP(1, 9); SN_STEP(10, 12); SN_STEP(10, 11); 
        SN_STEP(6, 10); SN_STEP(6, 9); SN_STEP(2, 13); SN_STEP(2, 6); SN_STEP(4, 11); 

        SN_STEP(7, 12); SN_STEP(1, 3); SN_STEP(4, 6); SN_STEP(2, 3); SN_STEP(5, 9); 
        SN_STEP(7, 8); SN_STEP(5, 7); SN_STEP(4, 5); SN_STEP(6, 7); 
    }
};

template<>
struct PartialSortingNetwork<9, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 15)
        Func func
    ) {
        // 31
        SN_STEP(10, 14); SN_STEP(11, 12); SN_STEP(12, 13); SN_STEP(9, 11); SN_STEP(11, 14); 
        SN_STEP(13, 14); SN_STEP(10, 12); SN_STEP(11, 12); SN_STEP(7, 14); SN_STEP(12, 13); 
        SN_STEP(9, 10); SN_STEP(0, 9); SN_STEP(8, 9); SN_STEP(10, 11); SN_STEP(11, 12); 
        SN_STEP(3, 10); SN_STEP(5, 12); SN_STEP(6, 11); SN_STEP(1, 5); SN_STEP(4, 13); 

        SN_STEP(4, 8); SN_STEP(2, 6); SN_STEP(2, 4); SN_STEP(6, 8); SN_STEP(7, 10); 
        SN_STEP(5, 7); SN_STEP(1, 3); SN_STEP(7, 8); SN_STEP(1, 2); SN_STEP(3, 4); 
        SN_STEP(5, 6); 
    }
};

template<>
struct PartialSortingNetwork<10, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 16)
        Func func
    ) {
        // 33
        SN_STEP(13, 14); SN_STEP(10, 15); SN_STEP(10, 12); SN_STEP(14, 15); SN_STEP(11, 13); 
        SN_STEP(12, 13); SN_STEP(13, 15); SN_STEP(12, 14); SN_STEP(4, 15); SN_STEP(10, 11); 
        SN_STEP(11, 14); SN_STEP(8, 12); SN_STEP(13, 14); SN_STEP(5, 14); SN_STEP(1, 10); 
        SN_STEP(8, 11); SN_STEP(0, 8); SN_STEP(2, 13); SN_STEP(7, 11); SN_STEP(3, 7); 

        SN_STEP(6, 13); SN_STEP(9, 10); SN_STEP(2, 8); SN_STEP(4, 8); SN_STEP(5, 9); 
        SN_STEP(7, 9); SN_STEP(3, 5); SN_STEP(4, 5); SN_STEP(6, 8); SN_STEP(0, 1); 
        SN_STEP(2, 3); SN_STEP(6, 7); SN_STEP(8, 9); 
    }
};

template<>
struct PartialSortingNetwork<11, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 17)
        Func func
    ) {
        // 35
        SN_STEP(14, 16); SN_STEP(11, 13); SN_STEP(12, 15); SN_STEP(11, 14); SN_STEP(14, 15); 
        SN_STEP(13, 16); SN_STEP(12, 13); SN_STEP(11, 12); SN_STEP(9, 14); SN_STEP(15, 16); 
        SN_STEP(13, 15); SN_STEP(5, 16); SN_STEP(0, 12); SN_STEP(9, 13); SN_STEP(9, 12); 
        SN_STEP(10, 11); SN_STEP(6, 15); SN_STEP(0, 10); SN_STEP(8, 12); SN_STEP(7, 13); 

        SN_STEP(4, 8); SN_STEP(1, 9); SN_STEP(5, 9); SN_STEP(3, 7); SN_STEP(7, 9); 
        SN_STEP(2, 10); SN_STEP(6, 10); SN_STEP(8, 10); SN_STEP(1, 2); SN_STEP(7, 8); 
        SN_STEP(4, 6); SN_STEP(3, 5); SN_STEP(5, 6); SN_STEP(3, 4); SN_STEP(9, 10); 
        
    }
};

template<>
struct PartialSortingNetwork<12, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 18)
        Func func
    ) {
        // 37
        SN_STEP(12, 17); SN_STEP(13, 15); SN_STEP(14, 16); SN_STEP(12, 13); SN_STEP(13, 14); 
        SN_STEP(15, 16); SN_STEP(15, 17); SN_STEP(12, 13); SN_STEP(1, 13); SN_STEP(16, 17); 
        SN_STEP(0, 15); SN_STEP(6, 17); SN_STEP(10, 15); SN_STEP(11, 12); SN_STEP(14, 16); 
        SN_STEP(3, 11); SN_STEP(10, 14); SN_STEP(10, 13); SN_STEP(7, 16); SN_STEP(8, 14); 

        SN_STEP(4, 8); SN_STEP(0, 3); SN_STEP(2, 10); SN_STEP(6, 10); SN_STEP(1, 3); 
        SN_STEP(9, 13); SN_STEP(8, 10); SN_STEP(5, 9); SN_STEP(4, 6); SN_STEP(7, 11); 
        SN_STEP(5, 7); SN_STEP(4, 5); SN_STEP(9, 11); SN_STEP(10, 11); SN_STEP(2, 3); 
        SN_STEP(8, 9); SN_STEP(6, 7); 
    }
};

template<>
struct PartialSortingNetwork<13, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 19)
        Func func
    ) {
        // 40
        SN_STEP(14, 16); SN_STEP(15, 18); SN_STEP(14, 15); SN_STEP(16, 18); SN_STEP(13, 17); 
        SN_STEP(15, 16); SN_STEP(13, 14); SN_STEP(14, 16); SN_STEP(15, 17); SN_STEP(11, 14); 
        SN_STEP(17, 18); SN_STEP(11, 15); SN_STEP(16, 17); SN_STEP(10, 15); SN_STEP(9, 16); 
        SN_STEP(12, 13); SN_STEP(6, 10); SN_STEP(3, 11); SN_STEP(8, 17); SN_STEP(4, 8); 

        SN_STEP(5, 9); SN_STEP(0, 12); SN_STEP(8, 12); SN_STEP(7, 18); SN_STEP(10, 12); 
        SN_STEP(7, 11); SN_STEP(4, 8); SN_STEP(9, 11); SN_STEP(1, 5); SN_STEP(5, 7); 
        SN_STEP(2, 6); SN_STEP(6, 8); SN_STEP(1, 3); SN_STEP(7, 8); SN_STEP(5, 6); 
        SN_STEP(11, 12); SN_STEP(2, 4); SN_STEP(1, 2); SN_STEP(9, 10); SN_STEP(3, 4); 

        
    }
};

template<>
struct PartialSortingNetwork<14, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 20)
        Func func
    ) {
        // 43
        SN_STEP(14, 17); SN_STEP(16, 19); SN_STEP(17, 18); SN_STEP(15, 17); SN_STEP(17, 19); 
        SN_STEP(14, 16); SN_STEP(14, 15); SN_STEP(18, 19); SN_STEP(8, 19); SN_STEP(16, 17); 
        SN_STEP(0, 14); SN_STEP(13, 14); SN_STEP(8, 13); SN_STEP(12, 15); SN_STEP(1, 8); 
        SN_STEP(17, 18); SN_STEP(12, 16); SN_STEP(11, 17); SN_STEP(11, 16); SN_STEP(3, 11); 

        SN_STEP(10, 16); SN_STEP(7, 11); SN_STEP(2, 10); SN_STEP(4, 12); SN_STEP(9, 18); 
        SN_STEP(5, 9); SN_STEP(6, 10); SN_STEP(12, 13); SN_STEP(9, 12); SN_STEP(1, 4); 
        SN_STEP(2, 4); SN_STEP(6, 9); SN_STEP(5, 8); SN_STEP(10, 13); SN_STEP(3, 5); 
        SN_STEP(11, 12); SN_STEP(12, 13); SN_STEP(10, 11); SN_STEP(7, 8); SN_STEP(4, 5); 

        SN_STEP(2, 3); SN_STEP(6, 7); SN_STEP(8, 9); 
    }
};

template<>
struct PartialSortingNetwork<15, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 21)
        Func func
    ) {
        // 45
        SN_STEP(16, 19); SN_STEP(15, 17); SN_STEP(17, 19); SN_STEP(15, 16); SN_STEP(14, 18); 
        SN_STEP(16, 17); SN_STEP(14, 20); SN_STEP(19, 20); SN_STEP(13, 17); SN_STEP(16, 19); 
        SN_STEP(14, 15); SN_STEP(13, 15); SN_STEP(13, 16); SN_STEP(0, 14); SN_STEP(15, 19); 
        SN_STEP(11, 15); SN_STEP(1, 13); SN_STEP(11, 13); SN_STEP(3, 11); SN_STEP(9, 20); 

        SN_STEP(7, 9); SN_STEP(7, 11); SN_STEP(6, 14); SN_STEP(10, 19); SN_STEP(2, 10); 
        SN_STEP(10, 14); SN_STEP(12, 16); SN_STEP(2, 6); SN_STEP(5, 13); SN_STEP(8, 12); 
        SN_STEP(12, 14); SN_STEP(4, 8); SN_STEP(1, 2); SN_STEP(9, 13); SN_STEP(4, 6); 
        SN_STEP(8, 10); SN_STEP(5, 7); SN_STEP(2, 4); SN_STEP(9, 11); SN_STEP(7, 8); 

        SN_STEP(9, 10); SN_STEP(13, 14); SN_STEP(5, 6); SN_STEP(11, 12); SN_STEP(3, 4); 
        
    }
};

template<>
struct PartialSortingNetwork<16, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 22)
        Func func
    ) {
        // 48
        SN_STEP(16, 18); SN_STEP(19, 21); SN_STEP(18, 21); SN_STEP(19, 20); SN_STEP(16, 17); 
        SN_STEP(16, 19); SN_STEP(17, 20); SN_STEP(15, 16); SN_STEP(20, 21); SN_STEP(17, 19); 
        SN_STEP(18, 19); SN_STEP(19, 20); SN_STEP(17, 18); SN_STEP(13, 18); SN_STEP(7, 13); 
        SN_STEP(1, 15); SN_STEP(2, 19); SN_STEP(14, 17); SN_STEP(0, 14); SN_STEP(8, 14); 

        SN_STEP(9, 15); SN_STEP(11, 20); SN_STEP(12, 19); SN_STEP(12, 15); SN_STEP(10, 21); 
        SN_STEP(4, 10); SN_STEP(5, 11); SN_STEP(11, 14); SN_STEP(2, 8); SN_STEP(4, 8); 
        SN_STEP(10, 13); SN_STEP(6, 12); SN_STEP(14, 15); SN_STEP(13, 14); SN_STEP(6, 8); 
        SN_STEP(3, 7); SN_STEP(10, 11); SN_STEP(14, 15); SN_STEP(5, 9); SN_STEP(3, 5); 

        SN_STEP(11, 12); SN_STEP(2, 3); SN_STEP(7, 9); SN_STEP(8, 9); SN_STEP(0, 1); 
        SN_STEP(6, 7); SN_STEP(10, 11); SN_STEP(4, 5); 
    }
};

template<>
struct PartialSortingNetwork<17, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 23)
        Func func
    ) {
        // 49
        SN_STEP(17, 22); SN_STEP(19, 21); SN_STEP(21, 22); SN_STEP(18, 20); SN_STEP(17, 19); 
        SN_STEP(17, 18); SN_STEP(19, 21); SN_STEP(16, 17); SN_STEP(18, 21); SN_STEP(20, 22); 
        SN_STEP(19, 20); SN_STEP(15, 22); SN_STEP(18, 19); SN_STEP(14, 19); SN_STEP(20, 21); 
        SN_STEP(10, 14); SN_STEP(2, 10); SN_STEP(0, 16); SN_STEP(13, 20); SN_STEP(1, 13); 

        SN_STEP(4, 21); SN_STEP(11, 18); SN_STEP(3, 11); SN_STEP(1, 3); SN_STEP(8, 16); 
        SN_STEP(12, 21); SN_STEP(2, 8); SN_STEP(12, 16); SN_STEP(7, 15); SN_STEP(9, 13); 
        SN_STEP(5, 9); SN_STEP(15, 18); SN_STEP(13, 15); SN_STEP(6, 10); SN_STEP(4, 8); 
        SN_STEP(3, 4); SN_STEP(10, 12); SN_STEP(14, 16); SN_STEP(15, 16); SN_STEP(7, 11); 

        SN_STEP(5, 7); SN_STEP(1, 2); SN_STEP(6, 8); SN_STEP(5, 6); SN_STEP(9, 11); 
        SN_STEP(11, 12); SN_STEP(13, 14); SN_STEP(7, 8); SN_STEP(9, 10); 
    }
};

template<>
struct PartialSortingNetwork<18, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 24)
        Func func
    ) {
        // 51
        SN_STEP(19, 20); SN_STEP(18, 23); SN_STEP(20, 22); SN_STEP(18, 21); SN_STEP(21, 23); 
        SN_STEP(19, 20); SN_STEP(18, 19); SN_STEP(20, 21); SN_STEP(19, 20); SN_STEP(22, 23); 
        SN_STEP(20, 22); SN_STEP(21, 22); SN_STEP(16, 19); SN_STEP(8, 16); SN_STEP(0, 8); 
        SN_STEP(2, 21); SN_STEP(13, 22); SN_STEP(1, 18); SN_STEP(12, 23); SN_STEP(11, 20); 

        SN_STEP(3, 11); SN_STEP(9, 18); SN_STEP(3, 9); SN_STEP(4, 12); SN_STEP(2, 8); 
        SN_STEP(4, 8); SN_STEP(10, 21); SN_STEP(5, 13); SN_STEP(5, 9); SN_STEP(7, 11); 
        SN_STEP(0, 1); SN_STEP(17, 18); SN_STEP(6, 10); SN_STEP(12, 16); SN_STEP(7, 9); 
        SN_STEP(10, 12); SN_STEP(6, 8); SN_STEP(14, 21); SN_STEP(13, 17); SN_STEP(15, 20); 

        SN_STEP(8, 9); SN_STEP(2, 3); SN_STEP(11, 13); SN_STEP(10, 11); SN_STEP(6, 7); 
        SN_STEP(12, 13); SN_STEP(15, 17); SN_STEP(4, 5); SN_STEP(14, 16); SN_STEP(14, 15); 
        SN_STEP(16, 17); 
    }
};

template<>
struct PartialSortingNetwork<19, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 25)
        Func func
    ) {
        // 53
        SN_STEP(19, 20); SN_STEP(20, 24); SN_STEP(21, 23); SN_STEP(22, 23); SN_STEP(23, 24); 
        SN_STEP(20, 22); SN_STEP(19, 21); SN_STEP(21, 22); SN_STEP(19, 20); SN_STEP(20, 21); 
        SN_STEP(21, 23); SN_STEP(17, 20); SN_STEP(1, 17); SN_STEP(13, 24); SN_STEP(18, 19); 
        SN_STEP(0, 21); SN_STEP(22, 23); SN_STEP(16, 21); SN_STEP(13, 17); SN_STEP(15, 22); 

        SN_STEP(3, 15); SN_STEP(9, 17); SN_STEP(8, 16); SN_STEP(12, 16); SN_STEP(4, 8); 
        SN_STEP(2, 18); SN_STEP(7, 15); SN_STEP(11, 15); SN_STEP(10, 18); SN_STEP(6, 23); 
        SN_STEP(6, 10); SN_STEP(15, 17); SN_STEP(5, 13); SN_STEP(14, 23); SN_STEP(14, 18); 
        SN_STEP(16, 18); SN_STEP(0, 2); SN_STEP(4, 6); SN_STEP(9, 13); SN_STEP(7, 9); 

        SN_STEP(3, 5); SN_STEP(12, 14); SN_STEP(3, 4); SN_STEP(11, 13); SN_STEP(8, 10); 
        SN_STEP(9, 10); SN_STEP(11, 12); SN_STEP(13, 14); SN_STEP(5, 6); SN_STEP(7, 8); 
        SN_STEP(15, 16); SN_STEP(17, 18); SN_STEP(1, 2); 
    }
};

template<>
struct PartialSortingNetwork<20, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 26)
        Func func
    ) {
        // 57
        SN_STEP(19, 20); SN_STEP(23, 25); SN_STEP(21, 24); SN_STEP(19, 22); SN_STEP(24, 25); 
        SN_STEP(22, 24); SN_STEP(24, 25); SN_STEP(19, 21); SN_STEP(21, 22); SN_STEP(19, 23); 
        SN_STEP(22, 23); SN_STEP(21, 22); SN_STEP(23, 24); SN_STEP(11, 21); SN_STEP(14, 25); 
        SN_STEP(12, 23); SN_STEP(14, 21); SN_STEP(7, 14); SN_STEP(9, 12); SN_STEP(0, 19); 

        SN_STEP(1, 11); SN_STEP(3, 11); SN_STEP(13, 22); SN_STEP(13, 19); SN_STEP(2, 13); 
        SN_STEP(10, 24); SN_STEP(6, 10); SN_STEP(17, 22); SN_STEP(6, 13); SN_STEP(5, 9); 
        SN_STEP(7, 11); SN_STEP(12, 19); SN_STEP(8, 12); SN_STEP(18, 21); SN_STEP(10, 13); 
        SN_STEP(14, 19); SN_STEP(12, 13); SN_STEP(15, 24); SN_STEP(15, 19); SN_STEP(9, 11); 

        SN_STEP(3, 5); SN_STEP(16, 23); SN_STEP(4, 8); SN_STEP(8, 10); SN_STEP(17, 19); 
        SN_STEP(9, 10); SN_STEP(4, 6); SN_STEP(16, 18); SN_STEP(16, 17); SN_STEP(5, 7); 
        SN_STEP(5, 6); SN_STEP(13, 14); SN_STEP(11, 12); SN_STEP(18, 19); SN_STEP(7, 8); 
        SN_STEP(3, 4); SN_STEP(1, 2); 
    }
};

template<>
struct PartialSortingNetwork<21, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 27)
        Func func
    ) {
        // 57
        SN_STEP(20, 24); SN_STEP(21, 23); SN_STEP(22, 25); SN_STEP(21, 22); SN_STEP(23, 25); 
        SN_STEP(20, 26); SN_STEP(0, 20); SN_STEP(0, 21); SN_STEP(20, 22); SN_STEP(25, 26); 
        SN_STEP(23, 25); SN_STEP(22, 25); SN_STEP(1, 23); SN_STEP(20, 21); SN_STEP(1, 20); 
        SN_STEP(12, 20); SN_STEP(16, 25); SN_STEP(2, 21); SN_STEP(22, 23); SN_STEP(4, 12); 

        SN_STEP(21, 22); SN_STEP(14, 22); SN_STEP(11, 21); SN_STEP(7, 26); SN_STEP(15, 26); 
        SN_STEP(16, 20); SN_STEP(6, 14); SN_STEP(2, 4); SN_STEP(8, 16); SN_STEP(19, 21); 
        SN_STEP(3, 11); SN_STEP(15, 19); SN_STEP(10, 14); SN_STEP(8, 12); SN_STEP(6, 8); 
        SN_STEP(13, 23); SN_STEP(14, 16); SN_STEP(7, 11); SN_STEP(10, 12); SN_STEP(9, 13); 

        SN_STEP(5, 9); SN_STEP(5, 7); SN_STEP(18, 22); SN_STEP(13, 15); SN_STEP(18, 20); 
        SN_STEP(5, 6); SN_STEP(17, 23); SN_STEP(17, 19); SN_STEP(9, 11); SN_STEP(19, 20); 
        SN_STEP(9, 10); SN_STEP(11, 12); SN_STEP(13, 14); SN_STEP(3, 4); SN_STEP(17, 18); 
        SN_STEP(7, 8); SN_STEP(15, 16); 
    }
};

template<>
struct PartialSortingNetwork<22, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 28)
        Func func
    ) {
        // 61
        SN_STEP(22, 24); SN_STEP(21, 22); SN_STEP(23, 25); SN_STEP(26, 27); SN_STEP(24, 27); 
        SN_STEP(21, 23); SN_STEP(21, 26); SN_STEP(20, 23); SN_STEP(25, 27); SN_STEP(20, 26); 
        SN_STEP(10, 27); SN_STEP(24, 25); SN_STEP(24, 26); SN_STEP(20, 24); SN_STEP(25, 26); 
        SN_STEP(17, 26); SN_STEP(0, 20); SN_STEP(14, 25); SN_STEP(5, 17); SN_STEP(14, 20); 

        SN_STEP(9, 21); SN_STEP(6, 10); SN_STEP(11, 24); SN_STEP(12, 20); SN_STEP(16, 27); 
        SN_STEP(2, 14); SN_STEP(13, 21); SN_STEP(17, 21); SN_STEP(16, 20); SN_STEP(8, 12); 
        SN_STEP(6, 14); SN_STEP(4, 8); SN_STEP(19, 24); SN_STEP(10, 14); SN_STEP(18, 25); 
        SN_STEP(1, 9); SN_STEP(13, 17); SN_STEP(15, 19); SN_STEP(15, 17); SN_STEP(8, 10); 

        SN_STEP(3, 11); SN_STEP(19, 21); SN_STEP(7, 11); SN_STEP(3, 9); SN_STEP(5, 9); 
        SN_STEP(11, 13); SN_STEP(18, 20); SN_STEP(7, 9); SN_STEP(10, 11); SN_STEP(8, 9); 
        SN_STEP(16, 17); SN_STEP(0, 1); SN_STEP(4, 6); SN_STEP(6, 7); SN_STEP(20, 21); 
        SN_STEP(18, 19); SN_STEP(12, 14); SN_STEP(12, 13); SN_STEP(2, 3); SN_STEP(4, 5); 

        SN_STEP(14, 15); 
    }
};

template<>
struct PartialSortingNetwork<23, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 29)
        Func func
    ) {
        // 63
        SN_STEP(23, 27); SN_STEP(24, 26); SN_STEP(23, 24); SN_STEP(25, 28); SN_STEP(26, 28); 
        SN_STEP(26, 27); SN_STEP(23, 25); SN_STEP(24, 25); SN_STEP(24, 26); SN_STEP(27, 28); 
        SN_STEP(25, 27); SN_STEP(22, 23); SN_STEP(25, 26); SN_STEP(15, 26); SN_STEP(13, 28); 
        SN_STEP(18, 25); SN_STEP(21, 24); SN_STEP(1, 21); SN_STEP(8, 22); SN_STEP(20, 27); 

        SN_STEP(6, 18); SN_STEP(20, 25); SN_STEP(10, 18); SN_STEP(13, 21); SN_STEP(2, 6); 
        SN_STEP(4, 20); SN_STEP(12, 20); SN_STEP(11, 15); SN_STEP(14, 18); SN_STEP(7, 11); 
        SN_STEP(0, 8); SN_STEP(5, 13); SN_STEP(15, 21); SN_STEP(3, 7); SN_STEP(17, 28); 
        SN_STEP(9, 15); SN_STEP(9, 13); SN_STEP(7, 9); SN_STEP(17, 21); SN_STEP(1, 8); 

        SN_STEP(4, 8); SN_STEP(6, 8); SN_STEP(16, 22); SN_STEP(20, 22); SN_STEP(18, 20); 
        SN_STEP(19, 26); SN_STEP(19, 21); SN_STEP(7, 8); SN_STEP(12, 16); SN_STEP(10, 12); 
        SN_STEP(3, 5); SN_STEP(2, 4); SN_STEP(3, 4); SN_STEP(5, 6); SN_STEP(19, 20); 
        SN_STEP(14, 16); SN_STEP(11, 13); SN_STEP(21, 22); SN_STEP(17, 18); SN_STEP(15, 16); 

        SN_STEP(11, 12); SN_STEP(9, 10); SN_STEP(13, 14); 
    }
};

template<>
struct PartialSortingNetwork<24, 6> {
    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 30)
        Func func
    ) {
        // 66
        SN_STEP(27, 28); SN_STEP(24, 26); SN_STEP(25, 26); SN_STEP(24, 27); SN_STEP(28, 29); 
        SN_STEP(25, 28); SN_STEP(24, 25); SN_STEP(26, 29); SN_STEP(22, 27); SN_STEP(0, 24); 
        SN_STEP(26, 28); SN_STEP(22, 28); SN_STEP(23, 24); SN_STEP(22, 26); SN_STEP(15, 23); 
        SN_STEP(22, 25); SN_STEP(17, 25); SN_STEP(3, 26); SN_STEP(7, 26); SN_STEP(20, 26); 

        SN_STEP(11, 28); SN_STEP(19, 28); SN_STEP(21, 25); SN_STEP(14, 22); SN_STEP(5, 29); 
        SN_STEP(1, 14); SN_STEP(10, 17); SN_STEP(4, 11); SN_STEP(3, 14); SN_STEP(12, 20); 
        SN_STEP(18, 29); SN_STEP(8, 15); SN_STEP(17, 18); SN_STEP(19, 23); SN_STEP(11, 15); 
        SN_STEP(16, 20); SN_STEP(9, 14); SN_STEP(18, 22); SN_STEP(2, 10); SN_STEP(13, 17); 

        SN_STEP(12, 14); SN_STEP(4, 8); SN_STEP(20, 22); SN_STEP(2, 4); SN_STEP(5, 9); 
        SN_STEP(21, 23); SN_STEP(6, 10); SN_STEP(13, 15); SN_STEP(17, 19); SN_STEP(7, 9); 
        SN_STEP(3, 4); SN_STEP(14, 15); SN_STEP(12, 13); SN_STEP(6, 8); SN_STEP(22, 23); 
        SN_STEP(20, 21); SN_STEP(16, 18); SN_STEP(18, 19); SN_STEP(10, 11); SN_STEP(16, 17); 

        SN_STEP(7, 8); SN_STEP(13, 14); SN_STEP(11, 12); SN_STEP(5, 6); SN_STEP(9, 10); 
        SN_STEP(1, 2); 
    }
};

#undef SN_STEP


// sorting networks for `n worthy candidates` approach

#define SN_STEPW(IDXA, IDXB) { func(distances_##IDXA, indices_##IDXA, distances_##IDXB, indices_##IDXB); }

template<size_t T, size_t N>
struct PartialSortingNetworkW {};

template<>
struct PartialSortingNetworkW<16, 6> {
    static constexpr size_t SN_T = 16;
    static constexpr size_t SN_N = 6;

    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 16)
        Func func
    ) {
        // 41
        SN_STEPW(5,6); SN_STEPW(12,14); SN_STEPW(0,7); SN_STEPW(2,3); SN_STEPW(10,15); 
        SN_STEPW(8,13); SN_STEPW(9,12); SN_STEPW(1,11); SN_STEPW(1,8); SN_STEPW(0,2); 
        SN_STEPW(3,7); SN_STEPW(2,12); SN_STEPW(6,13); SN_STEPW(6,10); SN_STEPW(2,6); 
        SN_STEPW(14,15); SN_STEPW(7,10); SN_STEPW(6,8); SN_STEPW(4,7); SN_STEPW(4,11); 
        
        SN_STEPW(0,9); SN_STEPW(5,6); SN_STEPW(4,9); SN_STEPW(7,12); SN_STEPW(3,14); 
        SN_STEPW(3,9); SN_STEPW(6,9); SN_STEPW(2,4); SN_STEPW(0,1); SN_STEPW(7,15); 
        SN_STEPW(3,5); SN_STEPW(1,6); SN_STEPW(13,14); SN_STEPW(1,4); SN_STEPW(6,7); 
        SN_STEPW(5,6); SN_STEPW(6,13); SN_STEPW(6,11); SN_STEPW(6,8); SN_STEPW(5,6); 
        
        SN_STEPW(4,6);
    }
};

template<>
struct PartialSortingNetworkW<16, 8> {
    static constexpr size_t SN_T = 16;
    static constexpr size_t SN_N = 8;

    template<typename DistancesEngineT, typename IndicesEngineT, typename Func>
    static __attribute__((always_inline)) inline void sort(
REPEAT_1D(GENERATE_INPUTS, 16)
        Func func
    ) {
        // 44
        SN_STEPW(5,13); SN_STEPW(6,9); SN_STEPW(1,10); SN_STEPW(3,7); SN_STEPW(8,14); 
        SN_STEPW(5,11); SN_STEPW(2,4); SN_STEPW(9,13); SN_STEPW(1,2); SN_STEPW(3,5); 
        SN_STEPW(5,14); SN_STEPW(2,9); SN_STEPW(0,15); SN_STEPW(10,14); SN_STEPW(7,15); 
        SN_STEPW(0,2); SN_STEPW(7,10); SN_STEPW(6,8); SN_STEPW(4,9); SN_STEPW(12,13); 

        SN_STEPW(2,5); SN_STEPW(8,12); SN_STEPW(3,6); SN_STEPW(9,14); SN_STEPW(0,6); 
        SN_STEPW(4,6); SN_STEPW(1,2); SN_STEPW(5,6); SN_STEPW(10,13); SN_STEPW(9,15); 
        SN_STEPW(6,12); SN_STEPW(6,9); SN_STEPW(4,7); SN_STEPW(2,7); SN_STEPW(7,10); 
        SN_STEPW(7,9); SN_STEPW(2,8); SN_STEPW(5,6); SN_STEPW(5,7); SN_STEPW(7,11); 

        SN_STEPW(6,10); SN_STEPW(6,8); SN_STEPW(6,11); SN_STEPW(7,8);
    }
};


#undef SN_STEPW

#undef GENERATE_INPUTS

}  // namespace smalltopk

#include <smalltopk/utils/macro_repeat_undefine.h>
