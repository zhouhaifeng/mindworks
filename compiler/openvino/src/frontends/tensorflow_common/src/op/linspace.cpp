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
OutputVector translate_linspace_op(const NodeContext& node) {
    // The operation is simple that generates a range [start, ..., stop]
    // with num elements staying in the same distance between each other
    default_op_checks(node, 3, {"LinSpace"});
    auto start = node.get_input(0);
    auto stop = node.get_input(1);
    auto num = node.get_input(2);

    // compute delta value, i.e. distance between neighbor values of the result
    auto const_one = create_same_type_const_scalar<int32_t>(num, 1);
    Output<Node> num_minus_one = make_shared<Subtract>(num, const_one);
    num_minus_one = make_shared<ConvertLike>(num_minus_one, start);
    Output<Node> delta = make_shared<Subtract>(stop, start);
    delta = make_shared<Divide>(delta, num_minus_one);

    // generate a range of numbers [0, 1, ..., num)
    // to have exact numbers of elements equal to num
    auto const_zero = create_same_type_const_scalar<int32_t>(num, 0);
    Output<Node> range0_n = make_shared<Range>(const_zero, num, const_one, ov::element::f32);
    range0_n = make_shared<ConvertLike>(range0_n, start);

    // compute the result
    Output<Node> linspace = make_shared<Multiply>(range0_n, delta);
    linspace = make_shared<Add>(linspace, start);
    set_node_name(node.get_name(), linspace.get_node_shared_ptr());
    return {linspace};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
