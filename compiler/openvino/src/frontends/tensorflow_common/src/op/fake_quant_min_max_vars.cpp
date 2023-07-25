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
OutputVector translate_fake_quant_op(const NodeContext& node) {
    default_op_checks(node, 2, {"FakeQuantWithMinMaxVars", "FakeQuantWithMinMaxVarsPerChannel"});
    auto inputs = node.get_input(0);
    auto min = node.get_input(1);
    auto max = node.get_input(2);

    // retrieve attributes
    auto narrow_range = node.get_attribute<bool>("narrow_range", false);
    auto num_bits = node.get_attribute<int64_t>("num_bits", 8);

    size_t levels = static_cast<size_t>(pow(2, num_bits));
    levels = narrow_range ? levels - 1 : levels;

    // compute real min and max values
    Output<Node> minimum = make_shared<Minimum>(min, max);
    Output<Node> maximum = make_shared<Maximum>(min, max);

    // adjust min and max so that min <= 0
    auto zero = create_same_type_const_scalar<float>(min, 0);
    auto min_greater_zero = make_shared<Greater>(minimum, zero);
    Output<Node> max_minus_min = make_shared<Subtract>(maximum, minimum);
    minimum = make_shared<Select>(min_greater_zero, zero, minimum);
    maximum = make_shared<Select>(min_greater_zero, max_minus_min, maximum);

    // adjust min and max so that 0 <= max
    auto max_less_zero = make_shared<Less>(maximum, zero);
    auto min_minus_max = make_shared<Subtract>(minimum, maximum);
    minimum = make_shared<Select>(max_less_zero, min_minus_max, minimum);
    maximum = make_shared<Select>(max_less_zero, zero, maximum);

    // adjust min and max so that scale = (max - min) / (2^num_bits - 1),
    // min_adj = scale * round(min / scale) and max_adj = max + min_adj - min
    max_minus_min = make_shared<Subtract>(maximum, minimum);
    auto const_levels = make_shared<Constant>(element::f32, Shape{}, static_cast<float>(levels - 1));
    auto scale = make_shared<Divide>(max_minus_min, const_levels);
    auto descaled_min = make_shared<Divide>(minimum, scale);
    auto rounded_descaled_min = make_shared<Round>(descaled_min, Round::RoundMode::HALF_TO_EVEN);
    auto min_adj = make_shared<Multiply>(scale, rounded_descaled_min);
    auto adjustment = make_shared<Subtract>(min_adj, minimum);
    auto max_adj = make_shared<Add>(maximum, adjustment);

    auto fake_quantize = make_shared<FakeQuantize>(inputs, min_adj, max_adj, min_adj, max_adj, levels);
    set_node_name(node.get_name(), fake_quantize);
    return {fake_quantize};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
