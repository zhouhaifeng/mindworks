// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_binary_op(const NodeContext& node,
                                 const std::function<Output<Node>(Output<Node>&, Output<Node>&)>& create_binary_op) {
    auto ng_lhs = node.get_input(0);
    auto ng_rhs = node.get_input(1);
    auto ng_node = create_binary_op(ng_lhs, ng_rhs);
    set_node_name(node.get_name(), ng_node.get_node_shared_ptr());
    return {ng_node};
}

OutputVector translate_floor_div_op(const NodeContext& node) {
    auto floordiv_fn = [](const Output<Node>& x, const Output<Node>& y) {
        return make_shared<Floor>(make_shared<Divide>(x, y));
    };
    return translate_binary_op(node, floordiv_fn);
}

template <typename T>
OutputVector translate_binary_op(const NodeContext& node) {
    return translate_binary_op(node, [](Output<Node>& ng_lhs, Output<Node>& ng_rhs) {
        return make_shared<T>(ng_lhs, ng_rhs);
    });
}

template OutputVector translate_binary_op<Add>(const NodeContext& node);
template OutputVector translate_binary_op<Equal>(const NodeContext& node);
template OutputVector translate_binary_op<FloorMod>(const NodeContext& node);
template OutputVector translate_binary_op<Greater>(const NodeContext& node);
template OutputVector translate_binary_op<GreaterEqual>(const NodeContext& node);
template OutputVector translate_binary_op<Less>(const NodeContext& node);
template OutputVector translate_binary_op<LessEqual>(const NodeContext& node);
template OutputVector translate_binary_op<LogicalAnd>(const NodeContext& node);
template OutputVector translate_binary_op<LogicalOr>(const NodeContext& node);
template OutputVector translate_binary_op<LogicalXor>(const NodeContext& node);
template OutputVector translate_binary_op<Maximum>(const NodeContext& node);
template OutputVector translate_binary_op<Minimum>(const NodeContext& node);
template OutputVector translate_binary_op<Multiply>(const NodeContext& node);
template OutputVector translate_binary_op<Mod>(const NodeContext& node);
template OutputVector translate_binary_op<NotEqual>(const NodeContext& node);
template OutputVector translate_binary_op<Power>(const NodeContext& node);
template OutputVector translate_binary_op<PRelu>(const NodeContext& node);
template OutputVector translate_binary_op<Divide>(const NodeContext& node);
template OutputVector translate_binary_op<SquaredDifference>(const NodeContext& node);
template OutputVector translate_binary_op<Subtract>(const NodeContext& node);

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
