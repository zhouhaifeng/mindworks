// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <numeric>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/shape.hpp"
#include "onnx_import/core/node.hpp"
#include "openvino/core/deprecated.hpp"
#include "utils/common.hpp"

namespace ngraph {
namespace onnx_import {
namespace variadic {
/// \brief Create an nGraph version of an ONNX variadic operation.
///        This creates a subgraph with a series of binary operations.
///
/// \param node Incoming ONNX opearation.
///
/// \tparam T   Class of an nGraph binary operation (e.g. Add, Minimum, Maximum)
///
/// \return nGraph node equivalent of the ONNX operation
OPENVINO_SUPPRESS_DEPRECATED_START
template <class T>
inline OutputVector make_ng_variadic_op(
    const Node& node,
    const ngraph::op::AutoBroadcastSpec& auto_broadcast = ngraph::op::AutoBroadcastType::NUMPY) {
    const OutputVector ng_inputs{node.get_ng_inputs()};

    // Templated binary operation - Creates Add, Minimum, Maximum, etc.
    const auto binary_operation = [&auto_broadcast](const Output<ngraph::Node>& arg0,
                                                    const Output<ngraph::Node>& arg1) {
        return std::make_shared<T>(arg0, arg1, auto_broadcast);
    };

    // Create a result node as a series of binary operations
    auto result = std::accumulate(std::next(std::begin(ng_inputs)),  // First operand value - the second input
                                  std::end(ng_inputs),               // Last value - final input
                                  ng_inputs.front(),                 // Initial value - first input
                                  binary_operation);

    if (ng_inputs.size() == 1) {
        common::mark_as_optimized_out(result);
    }

    return {result};
}
OPENVINO_SUPPRESS_DEPRECATED_END

}  // namespace variadic

}  // namespace  onnx_import

}  // namespace  ngraph
