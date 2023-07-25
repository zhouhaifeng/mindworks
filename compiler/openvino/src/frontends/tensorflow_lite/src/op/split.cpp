// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector split(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = get_decoder(node);
    std::map<std::string, ov::Any> attrs{
        {"num_split", static_cast<int64_t>(decoder->get_attribute(&tflite::SplitOptions::num_splits))}};
    return attribute_helper(node, attrs, ov::frontend::tensorflow::op::translate_split_op);
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
