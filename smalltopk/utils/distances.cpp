#include <smalltopk/utils/distances.h>

#include <cstdint>
#include <cstddef>

namespace smalltopk {

float l2_sqr(const float* const x, const size_t dim) {
    float output = 0;

    for (size_t i = 0; i < dim; i++) {
        output += x[i] * x[i];
    }

    return output;
}

}  // namespace smalltopk
