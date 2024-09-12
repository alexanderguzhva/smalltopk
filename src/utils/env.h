#pragma once

#include <optional>
#include <string>

namespace smalltopk {

//
std::string to_lower(const std::string& src);

//
std::optional<std::string> get_env(const std::string& name);

}  // namespace smalltopk
