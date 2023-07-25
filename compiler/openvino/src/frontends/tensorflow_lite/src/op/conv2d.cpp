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

OutputVector conv2d(const ov::frontend::tensorflow_lite::NodeContext& node) {
    auto decoder = get_conv_decoder_map<tflite::Conv2DOptions>("Conv2D", node);
    FRONT_END_GENERAL_CHECK(node.get_input_size() >= 2,
                            "Unexpected number of input in node of type=",
                            node.get_op_type(),
                            " name=",
                            node.get_name());
    OutputVector output;
    get_conv(output, node, decoder, &ov::frontend::tensorflow::op::translate_conv_2d_op);
    get_bias(output, node, decoder);
    get_activation(output, decoder);
    output[0].get_node_shared_ptr()->set_friendly_name(node.get_name());
    return output;
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
