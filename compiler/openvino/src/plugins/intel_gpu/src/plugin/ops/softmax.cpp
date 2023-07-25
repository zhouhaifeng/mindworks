// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/softmax.hpp"
#include "ngraph/op/log_softmax.hpp"

#include "intel_gpu/primitives/softmax.hpp"
#include "intel_gpu/primitives/activation.hpp"

namespace ov {
namespace intel_gpu {

static void CreateSoftmaxOp(Program& p, const std::shared_ptr<ngraph::op::v1::Softmax>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);
    auto softmaxPrim = cldnn::softmax(layerName,
                                      inputs[0],
                                      op->get_axis());
    p.add_primitive(*op, softmaxPrim);
}

static void CreateSoftmaxOp(Program& p, const std::shared_ptr<ngraph::op::v8::Softmax>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    OPENVINO_SUPPRESS_DEPRECATED_START
    int64_t axis = ov::normalize_axis(op.get(), op->get_axis(), op->get_input_partial_shape(0).rank());
    OPENVINO_SUPPRESS_DEPRECATED_END

    auto softmaxPrim = cldnn::softmax(layerName,
                                      inputs[0],
                                      axis);
    p.add_primitive(*op, softmaxPrim);
}

static void CreateLogSoftmaxOp(Program& p, const std::shared_ptr<ngraph::op::v5::LogSoftmax>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);
    std::string layerNameSoftmax = layer_type_name_ID(op) + "_softmax";

    OPENVINO_SUPPRESS_DEPRECATED_START
    int64_t axis = ov::normalize_axis(op.get(), op->get_axis(), op->get_input_partial_shape(0).rank());
    OPENVINO_SUPPRESS_DEPRECATED_END

    auto softmaxPrim = cldnn::softmax(layerNameSoftmax,
                                      inputs[0],
                                      axis);

    auto logPrim = cldnn::activation(layerName, cldnn::input_info(layerNameSoftmax), cldnn::activation_func::log, {(0.0F), (0.0F)});

    p.add_primitive(*op, softmaxPrim);
    p.add_primitive(*op, logPrim);
}

REGISTER_FACTORY_IMPL(v1, Softmax);
REGISTER_FACTORY_IMPL(v8, Softmax);
REGISTER_FACTORY_IMPL(v5, LogSoftmax);

}  // namespace intel_gpu
}  // namespace ov
