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

OutputVector translate_identity_op(const NodeContext& node) {
    vector<string> supported_ops = {"Identity",
                                    "PreventGradient",
                                    "Snapshot",
                                    "StopGradient",
                                    "ReadVariableOp",
                                    "ShardedFilename",
                                    "MergeV2Checkpoints"};
    default_op_checks(node, 1, supported_ops);
    auto input = node.get_input(0);

    // set only tensor names
    // no need to change node name since Identity node is skipped
    set_out_name(node.get_name(), input);
    set_out_name(node.get_name() + ":" + "0", input);
    return {input};
}

OutputVector translate_identity_n_op(const NodeContext& node) {
    default_op_checks(node, 1, {"IdentityN"});
    auto input_size = static_cast<int>(node.get_input_size());
    auto node_name = node.get_name();

    OutputVector result;
    for (int input_idx = 0; input_idx < input_size; ++input_idx) {
        auto input = node.get_input(input_idx);
        set_out_name(node_name + ":" + std::to_string(input_idx), input);
        result.push_back(input);
    }

    return result;
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
