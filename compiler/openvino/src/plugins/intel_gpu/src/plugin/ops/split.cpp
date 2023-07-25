// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/split.hpp"
#include "ngraph/op/variadic_split.hpp"

#include "intel_gpu/primitives/crop.hpp"

namespace ov {
namespace intel_gpu {

static void CreateCommonSplitOp(Program& p, const std::shared_ptr<ngraph::Node>& op) {
    auto get_layer_name = [&](size_t idx)->std::string {
        return layer_type_name_ID(op) + ((op->get_output_size() == 1)? "" : ".out" + std::to_string(idx));
    };

    auto inputs = p.GetInputInfo(op);
    if (p.use_new_shape_infer() || op->is_dynamic()) {
        std::vector<cldnn::tensor> offsets;

        if (!op->is_dynamic()) {
            auto input_pshape = op->get_input_partial_shape(0);
            InferenceEngine::SizeVector start_offset(input_pshape.size());
            for (size_t i = 0; i < op->get_output_size(); i++) {
                const auto outPartialShape = op->get_output_partial_shape(i);
                auto offsetTensor = tensor_from_dims(start_offset, 0);
                offsets.push_back(offsetTensor);

                for (size_t idx = 0; idx < input_pshape.size(); idx++) {
                    if (outPartialShape[idx] != input_pshape[idx]) {
                        start_offset[idx] += outPartialShape.to_shape()[idx];
                    }
                }
            }
        }

        cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
        auto num_splits = static_cast<size_t>(1);
        if (ngraph::is_type<ngraph::op::v1::Split>(op)) {
            num_splits = ngraph::as_type_ptr<ngraph::op::v1::Split>(op)->get_num_splits();
            op_mode = cldnn::crop_ngraph_op_mode::split;
        }

        for (size_t i = 0; i < op->get_output_size(); i++) {
            auto cropPrim = cldnn::crop(get_layer_name(i),
                                        inputs,
                                        cldnn::tensor(1),
                                        (op->is_dynamic() ? cldnn::tensor(0) : offsets[i]),
                                        op_mode,
                                        static_cast<int>(i),
                                        num_splits);
            p.add_primitive(*op, cropPrim);
        }
    } else {
        auto input_pshape = op->get_input_partial_shape(0);
        InferenceEngine::SizeVector start_offset(input_pshape.size());
        for (size_t i = 0; i < op->get_output_size(); i++) {
            const auto outPartialShape = op->get_output_partial_shape(i);
            NGRAPH_SUPPRESS_DEPRECATED_START
            if (outPartialShape.size() != start_offset.size()) {
                OPENVINO_THROW("Invalid dimesions in split layer: ", op->get_friendly_name(),
                               " output: ", ov::descriptor::get_ov_tensor_legacy_name(op->get_output_tensor(i)));
            }
            for (size_t idx = 0; idx < input_pshape.size(); idx++) {
                if ((outPartialShape[idx].get_length() + static_cast<ov::Dimension::value_type>(start_offset[idx])) > input_pshape[idx].get_length()) {
                    OPENVINO_THROW("Invalid dimesions in split layer: ", op->get_friendly_name(),
                                   " output: ", ov::descriptor::get_ov_tensor_legacy_name(op->get_output_tensor(idx)));
                }
            }
            NGRAPH_SUPPRESS_DEPRECATED_END

            auto offsetTensor = tensor_from_dims(start_offset, 0);
            auto outTensor = tensor_from_dims(op->get_output_shape(i), 1);
            auto cropPrim = cldnn::crop(get_layer_name(i), inputs[0], outTensor, offsetTensor);
            p.add_primitive(*op, cropPrim);

            for (size_t idx = 0; idx < input_pshape.size(); idx++) {
                if (outPartialShape[idx] != input_pshape[idx]) {
                    start_offset[idx] += outPartialShape.to_shape()[idx];
                }
            }
        }
    }
}

static void CreateSplitOp(Program& p, const std::shared_ptr<ngraph::op::v1::Split>& op) {
    validate_inputs_count(op, {2});
    CreateCommonSplitOp(p, op);
}

static void CreateVariadicSplitOp(Program& p, const std::shared_ptr<ngraph::op::v1::VariadicSplit>& op) {
    validate_inputs_count(op, {3});
    CreateCommonSplitOp(p, op);
}

REGISTER_FACTORY_IMPL(v1, Split);
REGISTER_FACTORY_IMPL(v1, VariadicSplit);

}  // namespace intel_gpu
}  // namespace ov
