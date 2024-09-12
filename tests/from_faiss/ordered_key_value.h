// reworked from Faiss

#pragma once

#include <limits>

namespace smalltopk {
namespace from_faiss {

template <typename T_, typename TI_>
struct CMax {
    using T = T_;
    using TI = TI_;
    inline static bool cmp(T a, T b) {
        return a > b;
    }
    // Similar to cmp(), but also breaks ties
    // by comparing the second pair of arguments.
    inline static bool cmp2(T a1, T b1, TI a2, TI b2) {
        return (a1 > b1) || ((a1 == b1) && (a2 > b2));
    }
    inline static T neutral() {
        return std::numeric_limits<T>::max();
    }
};

}
}
