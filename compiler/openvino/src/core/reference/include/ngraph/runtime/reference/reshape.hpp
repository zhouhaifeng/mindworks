// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_vector.hpp"
#include "ngraph/shape.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
void reshape(const char* arg,
             char* out,
             const Shape& in_shape,
             const AxisVector& in_axis_order,
             const Shape& out_shape,
             size_t elem_size);
}
}  // namespace runtime
}  // namespace ngraph
