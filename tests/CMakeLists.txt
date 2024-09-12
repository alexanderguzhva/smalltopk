include_directories(${CMAKE_HOME_DIRECTORY}/src)

set(SMALLTOPK_TEST_SRCS
    test_functionality.cpp
)

find_package(GTest REQUIRED)
find_package(BLAS REQUIRED)

add_executable(tests ${SMALLTOPK_TEST_SRCS})

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp")

target_link_libraries(tests 
    GTest::gtest_main 
    pthread 
    smalltopk
    ${BLAS_LIBRARIES}
)
