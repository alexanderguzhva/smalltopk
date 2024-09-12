cmake_minimum_required(VERSION 3.5)

project(smalltopk C CXX)

set (CMAKE_C_STANDARD 11)
set (CMAKE_CXX_STANDARD 17)

# # enable for ASAN in clang
# add_compile_options(-fno-stack-protector -fno-omit-frame-pointer -fno-var-tracking -fsanitize=address -g)
# add_link_options(-fno-stack-protector -fno-omit-frame-pointer -fno-var-tracking -fsanitize=address -g)

# add_compile_options(-march=native -g)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(faiss)
