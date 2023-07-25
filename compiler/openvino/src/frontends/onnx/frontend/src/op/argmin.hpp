// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
/// \brief Convert ONNX ArgMin operation to an nGraph node.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing an Ngraph node which produces the output
///         of an ONNX ArgMin operation.
OutputVector argmin(const Node& node);

}  // namespace set_1

namespace set_12 {
/// \brief Convert ONNX ArgMin operation to an nGraph node.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing an Ngraph node which produces the output
///         of an ONNX ArgMax operation.
OutputVector argmin(const Node& node);

}  // namespace set_12

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
