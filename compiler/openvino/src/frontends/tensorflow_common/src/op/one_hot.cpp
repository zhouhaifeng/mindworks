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

OutputVector translate_one_hot_op(const NodeContext& node) {
    default_op_checks(node, 4, {"OneHot"});
    auto indices = node.get_input(0);
    auto depth = node.get_input(1);
    auto on_value = node.get_input(2);
    auto off_value = node.get_input(3);

    auto axis = node.get_attribute<int64_t>("axis", -1);
    auto one_hot = make_shared<OneHot>(indices, depth, on_value, off_value, axis);
    set_node_name(node.get_name(), one_hot);
    return {one_hot};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
