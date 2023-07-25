// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_rsqrt_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Rsqrt", "RSQRT"});
    auto input = node.get_input(0);
    auto exponent = create_same_type_const_scalar<float>(input, -0.5f);
    auto rsqrt = make_shared<Power>(input, exponent);
    set_node_name(node.get_name(), rsqrt);
    return {rsqrt};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
