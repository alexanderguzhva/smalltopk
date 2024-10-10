#include "IndexSmallTopK.h"

#include <cstdio>

extern "C" {
    #include "../smalltopk/smalltopk.h"
}

namespace faiss {
namespace cppcontrib {
namespace smalltopk {

//
IndexFlatL2SmallTopK::IndexFlatL2SmallTopK() = default;

//  
IndexFlatL2SmallTopK::IndexFlatL2SmallTopK(faiss::idx_t d) : faiss::IndexFlat(d, faiss::MetricType::METRIC_L2) {}

void IndexFlatL2SmallTopK::search(
        faiss::idx_t n,
        const float* x,
        faiss::idx_t k,
        float* distances,
        faiss::idx_t* labels,
        const faiss::SearchParameters* params_in
) const {
    bool succeeded = false;

    if (params_in == nullptr || (params_in != nullptr && params_in->sel == nullptr)) {
        if (verbose) {
            printf("Evaluating n=%zd, k=%zd, ntotal=%zd, dim=%zd\n", size_t(n), size_t(k), size_t(ntotal), size_t(d));
        }

        KnnL2sqrParameters p;
        if (auto params = dynamic_cast<const SmallTopKSearchParameters*>(params_in)) {
            p.kernel = (uint32_t)params->kernel;
        } else {
            p.kernel = (uint32_t)smalltopk_kernel;
        }

        succeeded = knn_L2sqr_fp32(
            x,
            (const float*)this->codes.data(),
            this->d,
            n,
            this->ntotal,
            k,
            nullptr,
            nullptr,
            distances,
            labels,
            &p
        );
    }
    
    if (succeeded) {
        return;
    }
    
    // invoke a default version
    IndexFlat::search(n, x, k, distances, labels, params_in);
}

//
IndexFlatL2SmallTopKFactory::IndexFlatL2SmallTopKFactory(SmallTopKKernel kernel_in) :
    smalltopk_kernel{kernel_in} {}

//
faiss::Index* IndexFlatL2SmallTopKFactory::operator()(int dim) {
    if (verbose) {
        printf("New factory of %d\n", dim);
    }

    IndexFlatL2SmallTopK* index = new IndexFlatL2SmallTopK(dim);
    index->smalltopk_kernel = this->smalltopk_kernel;
    return index;
}

}
}
}

