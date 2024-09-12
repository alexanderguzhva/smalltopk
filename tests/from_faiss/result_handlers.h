#pragma once

#include <cstddef>
#include <cstdint>

#include "heap.h"

namespace smalltopk {
namespace from_faiss {

using idx_t = int64_t;

/** Encapsulates a set of ids to handle. */
struct IDSelector {
    virtual bool is_member(idx_t id) const = 0;
    virtual ~IDSelector() {}
};


template <class C, bool use_sel = false>
struct BlockResultHandler {
    size_t nq; // number of queries for which we search
    const IDSelector* sel;

    explicit BlockResultHandler(size_t nq, const IDSelector* sel = nullptr)
            : nq(nq), sel(sel) {
        assert(!use_sel || sel);
    }

    // currently handled query range
    size_t i0 = 0, i1 = 0;

    // start collecting results for queries [i0, i1)
    virtual void begin_multiple(size_t i0_2, size_t i1_2) {
        this->i0 = i0_2;
        this->i1 = i1_2;
    }

    // add results for queries [i0, i1) and database [j0, j1)
    virtual void add_results(size_t, size_t, const typename C::T*) {}

    // series of results for queries i0..i1 is done
    virtual void end_multiple() {}

    virtual ~BlockResultHandler() {}

    bool is_in_selection(idx_t i) const {
        return !use_sel || sel->is_member(i);
    }
};

// handler for a single query
template <class C>
struct ResultHandler {
    // if not better than threshold, then not necessary to call add_result
    typename C::T threshold = C::neutral();

    // return whether threshold was updated
    virtual bool add_result(typename C::T dis, typename C::TI idx) = 0;

    virtual ~ResultHandler() {}
};
/*****************************************************************
 * Single best result handler.
 * Tracks the only best result, thus avoiding storing
 * some temporary data in memory.
 *****************************************************************/

template <class C, bool use_sel = false>
struct Top1BlockResultHandler : BlockResultHandler<C, use_sel> {
    using T = typename C::T;
    using TI = typename C::TI;
    using BlockResultHandler<C, use_sel>::i0;
    using BlockResultHandler<C, use_sel>::i1;

    // contains exactly nq elements
    T* dis_tab;
    // contains exactly nq elements
    TI* ids_tab;

    Top1BlockResultHandler(
            size_t nq,
            T* dis_tab,
            TI* ids_tab,
            const IDSelector* sel = nullptr)
            : BlockResultHandler<C, use_sel>(nq, sel),
              dis_tab(dis_tab),
              ids_tab(ids_tab) {}

    struct SingleResultHandler : ResultHandler<C> {
        Top1BlockResultHandler& hr;
        using ResultHandler<C>::threshold;

        TI min_idx;
        size_t current_idx = 0;

        explicit SingleResultHandler(Top1BlockResultHandler& hr) : hr(hr) {}

        /// begin results for query # i
        void begin(const size_t current_idx_2) {
            this->current_idx = current_idx_2;
            threshold = C::neutral();
            min_idx = -1;
        }

        /// add one result for query i
        bool add_result(T dis, TI idx) final {
            if (C::cmp(this->threshold, dis)) {
                threshold = dis;
                min_idx = idx;
                return true;
            }
            return false;
        }

        /// series of results for query i is done
        void end() {
            hr.dis_tab[current_idx] = threshold;
            hr.ids_tab[current_idx] = min_idx;
        }
    };

    /// begin
    void begin_multiple(size_t i0, size_t i1) final {
        this->i0 = i0;
        this->i1 = i1;

        for (size_t i = i0; i < i1; i++) {
            this->dis_tab[i] = C::neutral();
        }
    }

    /// add results for query i0..i1 and j0..j1
    void add_results(size_t j0, size_t j1, const T* dis_tab_2) final {
        for (int64_t i = i0; i < i1; i++) {
            const T* dis_tab_i = dis_tab_2 + (j1 - j0) * (i - i0) - j0;

            auto& min_distance = this->dis_tab[i];
            auto& min_index = this->ids_tab[i];

            for (size_t j = j0; j < j1; j++) {
                const T distance = dis_tab_i[j];

                if (C::cmp(min_distance, distance)) {
                    min_distance = distance;
                    min_index = j;
                }
            }
        }
    }

    void add_result(const size_t i, const T dis, const TI idx) {
        auto& min_distance = this->dis_tab[i];
        auto& min_index = this->ids_tab[i];

        if (C::cmp(min_distance, dis)) {
            min_distance = dis;
            min_index = idx;
        }
    }
};

/*****************************************************************
 * Heap based result handler
 *****************************************************************/

template <class C, bool use_sel = false>
struct HeapBlockResultHandler : BlockResultHandler<C, use_sel> {
    using T = typename C::T;
    using TI = typename C::TI;
    using BlockResultHandler<C, use_sel>::i0;
    using BlockResultHandler<C, use_sel>::i1;

    T* heap_dis_tab;
    TI* heap_ids_tab;

    int64_t k; // number of results to keep

    HeapBlockResultHandler(
            size_t nq,
            T* heap_dis_tab,
            TI* heap_ids_tab,
            size_t k,
            const IDSelector* sel = nullptr)
            : BlockResultHandler<C, use_sel>(nq, sel),
              heap_dis_tab(heap_dis_tab),
              heap_ids_tab(heap_ids_tab),
              k(k) {}

    /******************************************************
     * API for 1 result at a time (each SingleResultHandler is
     * called from 1 thread)
     */

    struct SingleResultHandler : ResultHandler<C> {
        HeapBlockResultHandler& hr;
        using ResultHandler<C>::threshold;
        size_t k;

        T* heap_dis;
        TI* heap_ids;

        explicit SingleResultHandler(HeapBlockResultHandler& hr)
                : hr(hr), k(hr.k) {}

        /// begin results for query # i
        void begin(size_t i) {
            heap_dis = hr.heap_dis_tab + i * k;
            heap_ids = hr.heap_ids_tab + i * k;
            heap_heapify<C>(k, heap_dis, heap_ids);
            threshold = heap_dis[0];
        }

        /// add one result for query i
        bool add_result(T dis, TI idx) final {
            if (C::cmp(threshold, dis)) {
                heap_replace_top<C>(k, heap_dis, heap_ids, dis, idx);
                threshold = heap_dis[0];
                return true;
            }
            return false;
        }

        /// series of results for query i is done
        void end() {
            heap_reorder<C>(k, heap_dis, heap_ids);
        }
    };

    /******************************************************
     * API for multiple results (called from 1 thread)
     */

    /// begin
    void begin_multiple(size_t i0_2, size_t i1_2) final {
        this->i0 = i0_2;
        this->i1 = i1_2;
        for (size_t i = i0; i < i1; i++) {
            heap_heapify<C>(k, heap_dis_tab + i * k, heap_ids_tab + i * k);
        }
    }

    /// add results for query i0..i1 and j0..j1
    void add_results(size_t j0, size_t j1, const T* dis_tab) final {
#pragma omp parallel for
        for (int64_t i = i0; i < i1; i++) {
            T* heap_dis = heap_dis_tab + i * k;
            TI* heap_ids = heap_ids_tab + i * k;
            const T* dis_tab_i = dis_tab + (j1 - j0) * (i - i0) - j0;
            T thresh = heap_dis[0];
            for (size_t j = j0; j < j1; j++) {
                T dis = dis_tab_i[j];
                if (C::cmp(thresh, dis)) {
                    heap_replace_top<C>(k, heap_dis, heap_ids, dis, j);
                    thresh = heap_dis[0];
                }
            }
        }
    }

    /// series of results for queries i0..i1 is done
    void end_multiple() final {
        // maybe parallel for
        for (size_t i = i0; i < i1; i++) {
            heap_reorder<C>(k, heap_dis_tab + i * k, heap_ids_tab + i * k);
        }
    }
};

}
}
