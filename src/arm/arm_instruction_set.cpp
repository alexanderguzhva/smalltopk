#include "arm_instruction_set.h"


namespace smalltopk {

InstructionSet::InstructionSet() {
#ifdef __ARM_FEATURE_SVE
    is_sve_supported = true;
#else
    is_sve_supported = false;
#endif
}

}  // namespace smalltopk
