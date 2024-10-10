#pragma once

namespace smalltopk {

struct InstructionSet {
    InstructionSet();

    static InstructionSet& get_instance() {
        static InstructionSet singleton;
        return singleton;
    }

    bool is_avx512f_supported = false;
    bool is_avx512cd_supported = false;
    bool is_avx512bw_supported = false;
    bool is_avx512dq_supported = false;
    bool is_avx512vl_supported = false;
    bool is_avx512fp16_supported = false;
    bool is_avx512bf16_supported = false;
    bool is_avx512amxbf16_supported = false;

    // intel skylake capabilities
    bool is_avx512_cap_skylake = false;
};

}  // namespace smalltopk
