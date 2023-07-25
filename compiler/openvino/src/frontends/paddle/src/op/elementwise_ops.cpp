// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "elementwise_ops.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

//
NamedOutputs elementwise_add(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Add>(node_context);
}

NamedOutputs elementwise_sub(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Subtract>(node_context);
}

NamedOutputs elementwise_mul(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Multiply>(node_context);
}

NamedOutputs elementwise_div(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Divide>(node_context);
}

NamedOutputs elementwise_min(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Minimum>(node_context);
}

NamedOutputs elementwise_max(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Maximum>(node_context);
}

NamedOutputs elementwise_pow(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Power>(node_context);
}

NamedOutputs elementwise_equal(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Equal>(node_context);
}

NamedOutputs elementwise_greater_equal(const NodeContext& node_context) {
    return elementwise_ops<default_opset::GreaterEqual>(node_context);
}

NamedOutputs elementwise_not_equal(const NodeContext& node_context) {
    return elementwise_ops<default_opset::NotEqual>(node_context);
}

NamedOutputs elementwise_floordiv(const NodeContext& node_context) {
    auto x = node_context.get_input("X");
    auto y = node_context.get_input("Y");
    auto axis = -1;
    if (node_context.has_attribute("axis")) {
        axis = node_context.get_attribute<int>("axis");
    }
    return node_context.default_single_output_mapping(
        {std::make_shared<default_opset::Divide>(x,
                                                 y,
                                                 false,
                                                 ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, axis))},
        {"Out"});
}

NamedOutputs elementwise_mod(const NodeContext& node_context) {
    return elementwise_ops<default_opset::FloorMod>(node_context);
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
