// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/transpose.hpp"
#include "ngraph/op/constant.hpp"

#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/reorder.hpp"

namespace ov {
namespace intel_gpu {

static void CreateTransposeOp(Program& p, const std::shared_ptr<ngraph::op::v1::Transpose>& op) {
    validate_inputs_count(op, {1, 2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    std::vector<uint16_t> order;
    if (op->get_input_size() == 2) {
        auto order_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
        OPENVINO_ASSERT(order_constant != nullptr, "[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
        order = order_constant->cast_vector<uint16_t>();
    }

    auto is_convert_color_type_impl = [](const std::shared_ptr<ov::Node> &node) {
        return ngraph::is_type<ngraph::op::v8::NV12toRGB>(node) ||
               ngraph::is_type<ngraph::op::v8::NV12toBGR>(node) ||
               ngraph::is_type<ngraph::op::v8::I420toRGB>(node) ||
               ngraph::is_type<ngraph::op::v8::I420toBGR>(node);
    };

    auto is_convert_color_type = [&is_convert_color_type_impl](const std::shared_ptr<ov::Node> &node) {
        if (ngraph::is_type<ngraph::op::v0::Convert>(node)) {
            return is_convert_color_type_impl(node->get_input_node_shared_ptr(0));
        }
        return is_convert_color_type_impl(node);
    };

    // Handle Transpose operation related to ConvertColor operation:
    // In case of ConvertColor operation we have NHWC (byxf) input format which should be converted to
    // NCHW (bfyx) by this Permute, so we replace Permute with Reorder (to bfyx) primitve
    auto input = op->get_input_size() > 0 ? op->get_input_node_shared_ptr(0) : nullptr;
    // Handle the case ConvertColor -> FakeQuantize -> Permute
    auto input1 = input ? (input->get_input_size() > 0 ? input->get_input_node_shared_ptr(0) : nullptr) : nullptr;
    if (((input && is_convert_color_type(input)) || (input1 && is_convert_color_type(input1)))
            && order == std::vector<uint16_t>{0, 3, 1, 2}) {
        auto precision = input->get_element_type();
        auto reorder_prim = cldnn::reorder(layerName,
                                      inputs[0],
                                      cldnn::format::bfyx,
                                      cldnn::element_type_to_data_type(precision),
                                      std::vector<float>(),
                                      cldnn::reorder_mean_mode::none);
        p.add_primitive(*op, reorder_prim);
        return;
    }

    int rank = std::max(4, static_cast<int>(op->get_input_partial_shape(0).size()));
    if (order.empty()) {
        // if order size is less than 4 - fill the rest with just copy
        for (int o = rank - 1; o >= 0; o--)
            order.push_back((uint16_t)o);
    }

    auto permutePrim = cldnn::permute(layerName,
                                      inputs[0],
                                      order);

    p.add_primitive(*op, permutePrim);
}

REGISTER_FACTORY_IMPL(v1, Transpose);

}  // namespace intel_gpu
}  // namespace ov
