include_directories(${CMAKE_HOME_DIRECTORY})

find_package(GTest REQUIRED)
find_package(BLAS REQUIRED)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp")

add_executable(test_knn test_functionality.cpp)
target_link_libraries(test_knn 
    GTest::gtest_main 
    pthread 
    smalltopk
    ${BLAS_LIBRARIES}
)

add_executable(test_getmink test_getmink.cpp)
target_link_libraries(test_getmink 
    GTest::gtest_main 
    pthread 
    smalltopk
    ${BLAS_LIBRARIES}
)
