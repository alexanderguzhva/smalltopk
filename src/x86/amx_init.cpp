#include "amx_init.h"

#include <sys/syscall.h>
#include <unistd.h>

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

namespace smalltopk {

bool init_amx() {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) {
        // failed
        return false;
    }

    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILECFG) != 0) {
        // failed
        return false;
    }

    // success
    return true;
}

}  // namespace smalltopk
