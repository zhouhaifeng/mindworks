// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "generate_proposals.hpp"

#include "default_opset.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

namespace {
void validate_generate_proposals_inputs(const OutputVector& inputs) {
    OPENVINO_ASSERT(inputs.size() == 4, "GenerateProposals operator expects 4 inputs, got ", inputs.size());

    const auto scores_rank = inputs[0].get_partial_shape().rank();
    OPENVINO_ASSERT(scores_rank.compatible(4), "GenerateProposals input scores rank should be 4, is ", scores_rank);

    const auto& anchors_shape = inputs[3].get_partial_shape();
    const auto anchors_rank = anchors_shape.rank();
    OPENVINO_ASSERT(anchors_rank == Rank(2), "GenerateProposals input anchors rank should be 2, is ", anchors_rank);
    OPENVINO_ASSERT(anchors_shape[1].compatible(4),
                    "GenerateProposals input anchors shape should be {A, 4}, is ",
                    anchors_shape);
}
}  // namespace

OutputVector generate_proposals(const Node& node) {
    const auto inputs = node.get_ng_inputs();
    validate_generate_proposals_inputs(inputs);

    const auto& scores = inputs[0];   // shape [N, A, H, W]
    const auto& deltas = inputs[1];   // shape [N, A*4, H, W]
    const auto& im_info = inputs[2];  // shape [N, 3] or [N, 4]
    const auto& anchors = inputs[3];  // shape [A, 4]

    ov::op::v9::GenerateProposals::Attributes attrs;
    attrs.min_size = node.get_attribute_value<float>("min_size", 1.f);
    attrs.nms_threshold = node.get_attribute_value<float>("nms_thresh", 0.7f);
    attrs.pre_nms_count = node.get_attribute_value<int64_t>("pre_nms_topN", 6000);
    attrs.post_nms_count = node.get_attribute_value<int64_t>("post_nms_topN", 300);
    attrs.normalized = !node.get_attribute_value<int64_t>("legacy_plus_one", true);

    // Broadcast anchors from [A, 4] to [H, W, A, 4] where [H, W] is taken from scores shape.
    const auto zero = default_opset::Constant::create(element::i64, Shape{1}, {0});
    const auto scores_shape = std::make_shared<default_opset::ShapeOf>(scores);
    const auto anchors_shape = std::make_shared<default_opset::ShapeOf>(anchors);
    const auto scores_shape_tail = default_opset::Constant::create(element::i64, Shape{2}, {2, 3});
    const auto new_anchors_shape_front = std::make_shared<default_opset::Gather>(scores_shape, scores_shape_tail, zero);
    const auto new_anchors_shape =
        std::make_shared<default_opset::Concat>(OutputVector{new_anchors_shape_front, anchors_shape}, 0);
    const auto new_anchors = std::make_shared<default_opset::Broadcast>(anchors, new_anchors_shape);

    const auto proposals = std::make_shared<ov::op::v9::GenerateProposals>(im_info, new_anchors, deltas, scores, attrs);

    return proposals->outputs();
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
