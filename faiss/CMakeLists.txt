option(SMALLTOPK_FAISS_WRAPPER "Whether to compile a helper facilities for Faiss" OFF)
option(SMALLTOPK_FAISS_BUILD_BENCHMARK "Whether to build a benchmark what tests PQ and PRQ" OFF)

set(SMALLTOPK_FAISS_LIB "faiss" CACHE STRING "Specifies a library for Faiss (for example, faiss or faiss_avx2)")

if (SMALLTOPK_FAISS_WRAPPER)
    set(SMALLTOPK_FAISS_SRCS IndexSmallTopK.cpp)

    add_library(smalltopk_faiss SHARED ${SMALLTOPK_FAISS_SRCS})
    target_link_libraries(smalltopk_faiss PRIVATE smalltopk ${SMALLTOPK_FAISS_LIB})

    if (SMALLTOPK_FAISS_BUILD_BENCHMARK)
        set(SMALLTOPK_FAISS_BENCHMARK_SRCS benchmark/benchmark.cpp)

        add_executable(smalltopk_faiss_benchmark ${SMALLTOPK_FAISS_BENCHMARK_SRCS})
        target_link_libraries(smalltopk_faiss_benchmark PRIVATE smalltopk_faiss ${SMALLTOPK_FAISS_LIB})
    endif()
endif()
