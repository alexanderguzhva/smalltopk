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
};

}  // namespace smalltopk
