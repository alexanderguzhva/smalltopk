CC=clang-19 CXX=clang++-19 cmake -B build_rel -G Ninja -DSMALLTOPK_ENABLE_AVX512_FP32HACK_AMX=1 -DSMALLTOPK_ENABLE_FP32HACK_APPROX=1 -DSMALLTOPK_ENABLE_FP32=1 -DSMALLTOPK_ENABLE_FP16=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .
cd build_rel
ninja
