#include <faiss/index_factory.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/IndexPQ.h>
#include <faiss/impl/IDSelector.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>
#include <unordered_set>

#include "../IndexSmallTopK.h"

using namespace faiss::cppcontrib;

std::vector<float> generate_dataset(const size_t n, const size_t d, uint64_t seed) {
    std::default_random_engine rng(seed);
    std::uniform_real_distribution<float> u(-1, 1);

    std::vector<float> data(n * d);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = u(rng);
    }

    return data;
}

float get_recall_rate(
    const size_t nq,
    const size_t k,
    const std::vector<faiss::idx_t>& baseline, 
    const std::vector<faiss::idx_t>& candidate
) {
    size_t n = 0;
    for (size_t i = 0; i < nq; i++) {
        std::unordered_set<faiss::idx_t> a_set(k * 4);
        
        for (size_t j = 0; j < k; j++) {
            a_set.insert(baseline[i * k + j]);
        }

        for (size_t j = 0; j < k; j++) {
            auto itr = a_set.find(candidate[i * k + j]);
            if (itr != a_set.cend()) {
                n += 1;
            }
        }
    }

    return (float)n / candidate.size();
}

struct StopWatch {
    using timepoint_t = std::chrono::time_point<std::chrono::steady_clock>;
    timepoint_t Start;

    StopWatch() {
        Start = std::chrono::steady_clock::now();
    }

    double elapsed() const {
        const auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = now - Start;
        return elapsed.count();
    }
};

void benchmark_pq(
    // number of training samples
    const size_t nt, 
    // dim
    const size_t d, 
    // number of subquantizers for PQ
    const size_t nsubq, 
    // number of query samples
    const size_t nq, 
    // topk to look for
    const size_t k,
    // use smalltopk kernel; defaults to a regular faiss if undefined
    const std::optional<smalltopk::SmallTopKKernel> smalltopk_kernel
) {
    std::vector<float> xt = generate_dataset(nt, d, 123);

    // create an baseline
    std::unique_ptr<faiss::Index> baseline_index(
        faiss::index_factory(d, "Flat", faiss::MetricType::METRIC_L2)
    );

    baseline_index->train(nt, xt.data());
    baseline_index->add(nt, xt.data());

    // create a PQ index
    std::unique_ptr<faiss::IndexPQ> pq_index(new faiss::IndexPQ(d, nsubq, 8, faiss::MetricType::METRIC_L2));

    // some params
    pq_index->do_polysemous_training = false;

    std::unique_ptr<smalltopk::IndexFlatL2SmallTopKFactory> index_assign_factory;
    if (smalltopk_kernel.has_value()) {
        index_assign_factory = std::make_unique<smalltopk::IndexFlatL2SmallTopKFactory>(smalltopk_kernel.value());
        pq_index->pq.assign_index = index_assign_factory->operator()(d / nsubq);
    }

    // training PQ
    StopWatch sw_pq_trn;
    pq_index->train(nt, xt.data());
    const double pq_trn_elapsed = sw_pq_trn.elapsed();

    // adding points to PQ
    StopWatch sw_pq_add;
    pq_index->add(nt, xt.data());
    const double pq_trn_add = sw_pq_add.elapsed();

    // generate a query dataset
    std::vector<float> xq = generate_dataset(nq, d, 123);

    // a seed
    std::default_random_engine rng(789);

    // perform a baseline search
    std::vector<float> baseline_dis(k * nq, -1);
    std::vector<faiss::idx_t> baseline_ids(k * nq, -1);

    StopWatch sw_baseline;
    baseline_index->search(
        nq, 
        xq.data(), 
        k, 
        baseline_dis.data(), 
        baseline_ids.data());
    const double baseline_elapsed = sw_baseline.elapsed();


    // perform an pq search
    std::vector<float> pq_dis(k * nq, -1);
    std::vector<faiss::idx_t> pq_ids(k * nq, -1);

    StopWatch sw_pq;
    pq_index->search(
        nq, 
        xq.data(),
        k, 
        pq_dis.data(), 
        pq_ids.data());
    const double pq_elapsed = sw_pq.elapsed();


    // compute the recall rate
    const float recall_pq = get_recall_rate(nq, k, baseline_ids, pq_ids);
 
    printf("PQ, smalltopk kernel=%d, d=%zd, nsubq=%zd, nt=%zd, nq=%zd, recall_pq=%f, trn time=%f, add time=%f, search time=%f\n",
        (smalltopk_kernel.has_value()) ? ((int32_t)smalltopk_kernel.value()) : -1,
        d, nsubq, nt, nq, recall_pq,
        pq_trn_elapsed,
        pq_trn_add,
        pq_elapsed
    );
}

void benchmark_pq_kernel(const std::optional<smalltopk::SmallTopKKernel> smalltopk_kernel) {
    const size_t nt = 65536 * 2;
    const size_t dim = 192;
    const size_t nq = 128;
    benchmark_pq(nt, dim, dim / 2, nq, 64, smalltopk_kernel);
    benchmark_pq(nt, dim, dim / 3, nq, 64, smalltopk_kernel);
    benchmark_pq(nt, dim, dim / 4, nq, 64, smalltopk_kernel);
    benchmark_pq(nt, dim, dim / 8, nq, 64, smalltopk_kernel);
    benchmark_pq(nt, dim, dim / 16, nq, 64, smalltopk_kernel);
}

void benchmark_pq() {
    // test FAISS
    benchmark_pq_kernel(std::nullopt);
    // test kernels
    benchmark_pq_kernel(smalltopk::SmallTopKKernel::FP32);
    benchmark_pq_kernel(smalltopk::SmallTopKKernel::FP16);
    benchmark_pq_kernel(smalltopk::SmallTopKKernel::FP32HACK);
    benchmark_pq_kernel(smalltopk::SmallTopKKernel::FP32HACK_AVX512_AMX);
    benchmark_pq_kernel(smalltopk::SmallTopKKernel::FP32HACK_APPROX);
}

//
void benchmark_prq(
    // number of training samples
    const size_t nt, 
    // dim
    const size_t d, 
    // number of subquantizers for PRQ, similar to PQ
    const size_t nsubq,
    // number of residual layers
    const size_t nrq, 
    // number of query samples
    const size_t nq, 
    // topk to look for
    const size_t k,
    // use smalltopk kernel; defaults to a regular faiss if undefined
    const std::optional<smalltopk::SmallTopKKernel> smalltopk_kernel
) {
    std::vector<float> xt = generate_dataset(nt, d, 123);

    // create an baseline
    std::unique_ptr<faiss::Index> baseline_index(
        faiss::index_factory(d, "Flat", faiss::MetricType::METRIC_L2)
    );

    baseline_index->train(nt, xt.data());
    baseline_index->add(nt, xt.data());

    // create a PRQ index
    std::unique_ptr<faiss::IndexProductResidualQuantizer> prq_index(
        new faiss::IndexProductResidualQuantizer(d, nsubq, nrq, 8, faiss::MetricType::METRIC_L2));

    // some params
    std::unique_ptr<smalltopk::IndexFlatL2SmallTopKFactory> index_assign_factory;
    if (smalltopk_kernel.has_value()) {
        index_assign_factory = std::make_unique<smalltopk::IndexFlatL2SmallTopKFactory>(smalltopk_kernel.value());
    }

    for (auto* aq : prq_index->prq.quantizers) {
        faiss::ResidualQuantizer* rq = dynamic_cast<faiss::ResidualQuantizer*>(aq);
        if (smalltopk_kernel.has_value()) {
            rq->assign_index_factory = index_assign_factory.get();    
        }

        rq->use_beam_LUT = 1;
        rq->max_mem_distances = 64 * 1048576;

        // rq->niter_codebook_refine = 5;
        // rq->cp.niter = 10;
        // rq->cp.max_points_per_centroid = 2048;
        rq->max_beam_size = 16;

        rq->train_type = faiss::ResidualQuantizer::Train_default;
    }

    prq_index->prq.max_mem_distances = 64 * 1048576;

    // training PRQ
    StopWatch sw_prq_trn;
    prq_index->train(nt, xt.data());
    const double prq_trn_elapsed = sw_prq_trn.elapsed();

    // adding points to PRQ
    StopWatch sw_prq_add;
    prq_index->add(nt, xt.data());
    const double prq_trn_add = sw_prq_add.elapsed();

    // generate a query dataset
    std::vector<float> xq = generate_dataset(nq, d, 123);

    // a seed
    std::default_random_engine rng(789);

    // perform evaluation with a different level of filtering

    // perform a baseline search
    std::vector<float> baseline_dis(k * nq, -1);
    std::vector<faiss::idx_t> baseline_ids(k * nq, -1);

    StopWatch sw_baseline;
    baseline_index->search(
        nq, 
        xq.data(), 
        k, 
        baseline_dis.data(), 
        baseline_ids.data());
    const double baseline_elapsed = sw_baseline.elapsed();


    // perform an prq search
    std::vector<float> prq_dis(k * nq, -1);
    std::vector<faiss::idx_t> prq_ids(k * nq, -1);

    StopWatch sw_prq;
    prq_index->search(
        nq, 
        xq.data(),
        k, 
        prq_dis.data(), 
        prq_ids.data());
    const double prq_elapsed = sw_prq.elapsed();


    // compute the recall rate
    const float recall_prq = get_recall_rate(nq, k, baseline_ids, prq_ids);
 
    printf("PRQ, smalltopk kernel=%d, d=%zd, nrq=%zd, nsubq=%zd, nt=%zd, nq=%zd, recall_prq=%f, trn time=%f, add time=%f, search time=%f\n",
        (smalltopk_kernel.has_value()) ? ((int32_t)smalltopk_kernel.value()) : -1,
        d, nrq, nsubq, nt, nq, recall_prq,
        prq_trn_elapsed,
        prq_trn_add,
        prq_elapsed
    );
}

void benchmark_prq_kernel(const std::optional<smalltopk::SmallTopKKernel> smalltopk_kernel) {
    benchmark_prq(65536, 128, 4, 16, 128, 64, smalltopk_kernel);
    benchmark_prq(65536, 128, 8, 8, 128, 64, smalltopk_kernel);
    benchmark_prq(65536, 128, 16, 4, 128, 64, smalltopk_kernel);
    benchmark_prq(65536, 128, 32, 2, 128, 64, smalltopk_kernel);

    benchmark_prq(65536, 128, 4, 8, 128, 64, smalltopk_kernel);
    benchmark_prq(65536, 128, 8, 4, 128, 64, smalltopk_kernel);
    benchmark_prq(65536, 128, 16, 2, 128, 64, smalltopk_kernel);
}

void benchmark_prq() {
    // test FAISS
    benchmark_prq_kernel(std::nullopt);
    // test kernels
    benchmark_prq_kernel(smalltopk::SmallTopKKernel::FP32);
    benchmark_prq_kernel(smalltopk::SmallTopKKernel::FP16);
    benchmark_prq_kernel(smalltopk::SmallTopKKernel::FP32HACK);
    benchmark_prq_kernel(smalltopk::SmallTopKKernel::FP32HACK_AVX512_AMX);
    benchmark_prq_kernel(smalltopk::SmallTopKKernel::FP32HACK_APPROX);
}

int main() {
    benchmark_pq();
    return 0;
}

