// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T>
void copy(const T* arg, T* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        out[i] = arg[i];
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
