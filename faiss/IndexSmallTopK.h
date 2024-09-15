#pragma once

#include <cstdint>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

namespace faiss {
namespace cppcontrib {
namespace smalltopk {

enum class SmallTopKKernel : uint32_t {
    DEFAULT = 0,
    FP32 = 1,
    FP16 = 2,
    FP32HACK = 3,
    FP32HACK_AVX512_AMX = 4,
    FP32HACK_APPROX = 5
};

struct SmallTopKSearchParameters : faiss::SearchParameters {
    SmallTopKKernel kernel = SmallTopKKernel::DEFAULT;
};

// these are the indices that speed up building other indices

struct IndexFlatL2SmallTopK : faiss::IndexFlat {
    SmallTopKKernel smalltopk_kernel = SmallTopKKernel::DEFAULT;

    IndexFlatL2SmallTopK();

    explicit IndexFlatL2SmallTopK(faiss::idx_t d);

    void search(
            faiss::idx_t n,
            const float* x,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override;
};

struct IndexFlatL2SmallTopKFactory : faiss::ProgressiveDimIndexFactory {
    bool verbose = false;
    SmallTopKKernel smalltopk_kernel = SmallTopKKernel::DEFAULT;

    IndexFlatL2SmallTopKFactory(SmallTopKKernel kernel_in = SmallTopKKernel::DEFAULT);

    faiss::Index* operator()(int dim) override;
};

}  // namespace smalltopk
}  // namespace cppcontrib
}  // namespace faiss

