#pragma once

namespace smalltopk {

struct InstructionSet {
    InstructionSet();

    static InstructionSet& get_instance() {
        static InstructionSet singleton;
        return singleton;
    }

    bool is_sve_supported = false;
};

}  // namespace smalltopk
