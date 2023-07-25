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

OutputVector translate_squeeze_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Squeeze", "SQUEEZE"});
    auto input = node.get_input(0);
    std::vector<int64_t> axis;
    if (node.has_attribute("axis")) {
        axis = node.get_attribute<std::vector<int64_t>>("axis", {});
    } else {
        // check deprecated name
        axis = node.get_attribute<std::vector<int64_t>>("squeeze_dims", {});
    }
    auto axis_const = make_shared<Constant>(element::i32, Shape{axis.size()}, axis);
    auto squeeze = make_shared<Squeeze>(input, axis_const);
    set_node_name(node.get_name(), squeeze);
    return {squeeze};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
