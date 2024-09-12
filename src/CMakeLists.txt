set(SMALLTOPK_SRCS
    dummy.cpp
    smalltopk_dispatch.cpp
    utils/distances.cpp
    utils/env.cpp
    utils/norms.cpp
    utils/transpose.cpp
)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp")

# set default flags
OPTION(SMALLTOPK_ENABLE_FP32 "Whether to enable fp32 kernel" OFF)
OPTION(SMALLTOPK_ENABLE_FP32HACK "Whether to enable fp32hack kernel" ON)
OPTION(SMALLTOPK_ENABLE_FP16 "Whether to enable fp16 kernel" OFF)
OPTION(SMALLTOPK_ENABLE_AVX512_FP32HACK_AMX "Whether to enable fp32hack kernel with AMX" OFF)
OPTION(SMALLTOPK_ENABLE_FP32HACK_APPROX "Whether to enable fp32hack kernel with 'fixed number of worthy candidates' approach" OFF)

# files
if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")

    # I don't care about a debug version
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native -O3 -ffast-math")

    # this file is always needed
    list(APPEND SMALLTOPK_SRCS x86/x86_instruction_set.cpp)

    # FP32 is not very useful
    if (SMALLTOPK_ENABLE_FP32)
        message(STATUS "including fp32 kernel")

        list(APPEND SMALLTOPK_SRCS x86/avx512_sorting_fp32.cpp)
        set_source_files_properties(x86/avx512_sorting_fp32.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512cd")
    else()
        message(STATUS "not including fp32 kernel")

        list(APPEND SMALLTOPK_SRCS x86/avx512_sorting_fp32_dummy.cpp)
    endif()

    # FP32 hack
    if (SMALLTOPK_ENABLE_FP32HACK)
        message(STATUS "including fp32hack kernel")

        list(APPEND SMALLTOPK_SRCS x86/avx512_sorting_fp32hack.cpp)
        set_source_files_properties(x86/avx512_sorting_fp32hack.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512cd")
    else()
        message(STATUS "not including fp32hack kernel")

        list(APPEND SMALLTOPK_SRCS x86/avx512_sorting_fp32hack_dummy.cpp)
    endif()

    # FP16 is not widely available
    if (SMALLTOPK_ENABLE_FP16)
        message(STATUS "including fp16 kernel")

        list(APPEND SMALLTOPK_SRCS x86/avx512_sorting_fp16.cpp)
        set_source_files_properties(x86/avx512_sorting_fp16.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512cd -mavx512fp16 -mf16c")
    else()
        message(STATUS "not including fp16 kernel")

        list(APPEND SMALLTOPK_SRCS x86/avx512_sorting_fp16_dummy.cpp)
    endif()

    # FP32 hack + AMX
    if (SMALLTOPK_ENABLE_AVX512_FP32HACK_AMX)
        message(STATUS "including fp32hack AMX kernel")

        list(APPEND SMALLTOPK_SRCS 
            x86/amx_init.cpp
            x86/avx512_sorting_fp32hack_amx.cpp
        )
        set_source_files_properties(x86/avx512_sorting_fp32hack_amx.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512cd -mf16c -mamx-bf16 -mavx512bf16")
    else()
        message(STATUS "not including fp32hack AMX kernel")

        list(APPEND SMALLTOPK_SRCS 
            x86/amx_init_dummy.cpp
            x86/avx512_sorting_fp32hack_amx_dummy.cpp
        )
    endif()

    # FP32 hack + APPROX
    if (SMALLTOPK_ENABLE_FP32HACK_APPROX)
        message(STATUS "including fp32hack approx kernel")

        list(APPEND SMALLTOPK_SRCS x86/avx512_sorting_fp32hack_approx.cpp)
        set_source_files_properties(x86/avx512_sorting_fp32hack_approx.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512cd")
    else()
        message(STATUS "not including fp32hack approx kernel")

        list(APPEND SMALLTOPK_SRCS x86/avx512_sorting_fp32hack_approx_dummy.cpp)
    endif()

elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm*")

    # I don't care about a debug version
    # Sometimes, one needs to add -msve-vector-bits=256, bcz otherwise 
    #   gcc triggers an internal compiler error.
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native -O3 -ffast-math -mcpu=neoverse-v1")

    # this file is always needed
    list(APPEND SMALLTOPK_SRCS arm/arm_instruction_set.cpp)

    # FP32 is not very useful
    if (SMALLTOPK_ENABLE_FP32)
        message(STATUS "including fp32 kernel")

        list(APPEND SMALLTOPK_SRCS arm/sve_sorting_fp32.cpp)
        # set_source_files_properties(arm/sve_sorting_fp32.cpp PROPERTIES COMPILE_FLAGS "")
    else()
        message(STATUS "not including fp32 kernel")

        list(APPEND SMALLTOPK_SRCS arm/sve_sorting_fp32_dummy.cpp)
    endif()

    # FP32 hack
    if (SMALLTOPK_ENABLE_FP32HACK)
        message(STATUS "including fp32hack kernel")

        list(APPEND SMALLTOPK_SRCS arm/sve_sorting_fp32hack.cpp)
        # set_source_files_properties(arm/sve_sorting_fp32hack.cpp PROPERTIES COMPILE_FLAGS "")
    else()
        message(STATUS "not including fp32hack kernel")

        list(APPEND SMALLTOPK_SRCS arm/sve_sorting_fp32hack_dummy.cpp)
    endif()

    # FP16 is not widely available
    if (SMALLTOPK_ENABLE_FP16)
        message(STATUS "including fp16 kernel")

        list(APPEND SMALLTOPK_SRCS arm/sve_sorting_fp16.cpp)
        # set_source_files_properties(arm/sve_sorting_fp16.cpp PROPERTIES COMPILE_FLAGS "")
    else()
        message(STATUS "not including fp16 kernel")

        list(APPEND SMALLTOPK_SRCS arm/sve_sorting_fp16_dummy.cpp)
    endif()

    # FP32 hack + APPROX
    if (SMALLTOPK_ENABLE_FP32HACK_APPROX)
        message(STATUS "including fp32hack approx kernel")

        list(APPEND SMALLTOPK_SRCS arm/sve_sorting_fp32hack_approx.cpp)
        # set_source_files_properties(arm/sve_sorting_fp32hack_approx.cpp PROPERTIES COMPILE_FLAGS "")
    else()
        message(STATUS "not including fp32hack approx kernel")

        list(APPEND SMALLTOPK_SRCS arm/sve_sorting_fp32hack_approx_dummy.cpp)
    endif()
endif()


add_library(smalltopk SHARED ${SMALLTOPK_SRCS})
