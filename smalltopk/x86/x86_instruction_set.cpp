#include <smalltopk/x86/x86_instruction_set.h>

#include <cpuid.h>

#include <array>
#include <bitset>
#include <vector>

namespace smalltopk {

InstructionSet::InstructionSet() {
    std::array<int, 4> cpui = {0, 0, 0, 0};

    __cpuid(0, cpui[0], cpui[1], cpui[2], cpui[3]);
    int n_ids = cpui[0];

    std::vector<std::array<int, 4>> data;

    for (int i = 0; i <= n_ids; i++) {
        __cpuid_count(i, 0, cpui[0], cpui[1], cpui[2], cpui[3]);
        data.push_back(cpui);
    }

    if (n_ids >= 7) {
        {
            std::bitset<32> ebx = data[7][1];
            is_avx512f_supported = ebx[16];
            is_avx512dq_supported = ebx[17];
            is_avx512cd_supported = ebx[28];
            is_avx512bw_supported = ebx[30];
            is_avx512vl_supported = ebx[31];

            std::bitset<32> edx = data[7][3];
            is_avx512fp16_supported = edx[23];
            is_avx512amxbf16_supported = edx[22];
        }

        is_avx512_cap_skylake = 
            is_avx512f_supported &&
            is_avx512dq_supported &&
            is_avx512cd_supported &&
            is_avx512bw_supported &&
            is_avx512vl_supported;

        if (is_avx512_cap_skylake) {
            // check bf16, fp16 and amx
            std::array<int, 4> cpui_7_1 = {0, 0, 0, 0};
            
            __cpuid_count(7, 1, cpui_7_1[0], cpui_7_1[1], cpui_7_1[2], cpui_7_1[3]);

            std::bitset<32> eax = cpui_7_1[0];
            is_avx512bf16_supported = eax[5];
        }
    }
}

}  // namespace smalltopk
