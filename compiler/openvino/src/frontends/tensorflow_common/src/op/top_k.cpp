// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset11.hpp"

using namespace std;
using namespace ov::opset11;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
NamedOutputVector translate_top_k_base_op(const NodeContext& node,
                                          const ov::Output<ov::Node>& k_input,
                                          int min_input_size) {
    default_op_checks(node, min_input_size, {"TopK", "TopKV2"});
    auto input = node.get_input(0);

    // retrieve k attribute
    bool sorted = node.get_attribute<bool>("sorted", true);
    auto top_k = make_shared<TopK>(input,
                                   k_input,
                                   -1,
                                   ov::op::v1::TopK::Mode::MAX,
                                   sorted ? TopK::SortType::SORT_VALUES : TopK::SortType::SORT_INDICES,
                                   ov::element::i32,
                                   true);
    set_node_name(node.get_name(), top_k);
    return {{"values", top_k->output(0)}, {"indices", top_k->output(1)}};
}
NamedOutputVector translate_top_k_op(const NodeContext& node) {
    // retrieve k attribute
    auto k = node.get_attribute<int64_t>("k");
    auto k_input = make_shared<Constant>(ov::element::i64, Shape{}, std::vector<int64_t>({k}));
    return translate_top_k_base_op(node, k_input, 1);
}

NamedOutputVector translate_top_k_v2_op(const NodeContext& node) {
    default_op_checks(node, 2, {"TopKV2"});
    auto k_input = node.get_input(1);
    return translate_top_k_base_op(node, k_input, 1);
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
