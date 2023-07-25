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

OutputVector translate_depthwise_conv_2d_native_op(const NodeContext& node) {
    default_op_checks(node, 2, {"DepthwiseConv2dNative"});
    auto input = node.get_input(0);
    auto filter = node.get_input(1);

    // retrive mandatory attributes for DepthwiseConv2dNative
    auto tf_strides = node.get_attribute<std::vector<int64_t>>("strides");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    ov::op::PadType auto_pad = convert_tf_padding(node, tf_padding_type);

    // retrieve optional attributes
    auto tf_data_format = node.get_attribute<std::string>("data_format", "NHWC");
    auto tf_dilations = node.get_attribute<std::vector<int64_t>>("dilations", {1, 1, 1, 1});

    TENSORFLOW_OP_VALIDATION(node,
                             auto_pad != ov::op::PadType::EXPLICIT,
                             "Explicit padding for DepthwiseConv2dNative is not supported.");
    TENSORFLOW_OP_VALIDATION(node,
                             tf_data_format == "NHWC" || tf_data_format == "NCHW",
                             "DepthwiseConv2dNative data format is neither NHWC nor NCHW");

    bool is_nhwc = (tf_data_format == "NHWC");

    Strides strides(2);
    Strides dilations(2);
    convert_nhwc_to_hw(is_nhwc, tf_strides, strides);
    convert_nhwc_to_hw(is_nhwc, tf_dilations, dilations);

    Shape ng_image_shape(2);
    Shape ng_kernel_shape(2);

    convert_nhwc_to_nchw(is_nhwc, input, ov::Rank(4));

    // prepare filter to have a number of groups equal to CIN
    auto unsqueeze_filter =
        make_shared<Unsqueeze>(filter, make_shared<Constant>(element::i64, Shape{1}, std::vector<int64_t>{3}));
    auto transposed_filter =
        make_shared<Transpose>(unsqueeze_filter,
                               make_shared<Constant>(element::i64, Shape{5}, std::vector<int64_t>{2, 4, 3, 0, 1}));

    ov::Output<ov::Node> group_conv = make_shared<GroupConvolution>(input,
                                                                    transposed_filter,
                                                                    strides,
                                                                    CoordinateDiff({}),
                                                                    CoordinateDiff({}),
                                                                    dilations,
                                                                    auto_pad);
    ov::frontend::tensorflow::convert_nchw_to_nhwc(is_nhwc, group_conv, ov::Rank(4));
    ov::frontend::tensorflow::set_node_name(node.get_name(), group_conv.get_node_shared_ptr());
    return {group_conv};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
