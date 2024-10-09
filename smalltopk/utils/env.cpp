#include "env.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace smalltopk {

//
std::string to_lower(const std::string& src) {
    std::string dst;
    dst.resize(src.size());

    std::transform(src.begin(), src.end(), dst.begin(), ::tolower);

    return dst;    
}

//
std::optional<std::string> get_env(const std::string& name) {
    std::string v;
    if (const char* env_v = std::getenv(name.c_str())) {
        v = std::string(env_v);
    } else {
        return std::nullopt;
    }

    v = to_lower(v);

    return v;
}

}  // namespace smalltopk
