// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "serial/headers/2dot9/gna_model_header.hpp"

namespace ov {
namespace intel_gna {
namespace header_latest {

using ModelHeader = header_2_dot_9::ModelHeader;
using RuntimeEndPoint = header_2_dot_9::RuntimeEndPoint;

template <typename A, typename B>
bool IsFirstVersionLower(const A& first, const B& second) {
    return first.major < second.major || (first.major == second.major && first.minor < second.minor);
}

}  // namespace header_latest
}  // namespace intel_gna
}  // namespace ov
