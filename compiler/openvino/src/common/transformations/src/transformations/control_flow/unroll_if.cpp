// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/control_flow/unroll_if.hpp"

#include <memory>
#include <ngraph/graph_util.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/result.hpp"
#include "transformations/utils/utils.hpp"

bool ov::pass::UnrollIf::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(UnrollIf);
    bool is_applicable = false;
    for (const auto& op : f->get_ordered_ops()) {
        auto multisubgraph_op = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp>(op);
        if (multisubgraph_op) {
            for (size_t i = 0; i < multisubgraph_op->get_internal_subgraphs_size(); ++i) {
                run_on_model(multisubgraph_op->get_function(static_cast<int>(i)));
            }
        }
        auto if_node = std::dynamic_pointer_cast<ov::op::v8::If>(op);
        if (!if_node || transformation_callback(if_node)) {
            continue;
        }
        Output<Node> cond = if_node->input_value(0);
        OPENVINO_SUPPRESS_DEPRECATED_START
        const auto cond_is_const = ngraph::get_constant_from_source(cond);
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (!cond_is_const) {
            continue;
        }

        auto cond_value = cond_is_const->cast_vector<bool>();
        auto body = (cond_value[0]) ? if_node->get_then_body() : if_node->get_else_body();
        auto input_descriptions = if_node->get_input_descriptions(static_cast<int>(!cond_value[0]));
        auto output_descriptions = if_node->get_output_descriptions(static_cast<int>(!cond_value[0]));

        // connect inputs instead of body parameters
        for (const auto& input_descr : input_descriptions) {
            auto in_data = if_node->input_value(input_descr->m_input_index);
            auto& param = body->get_parameters()[input_descr->m_body_parameter_index];
            for (const auto& input : param->output(0).get_target_inputs()) {
                input.replace_source_output(in_data);
            }
        }
        for (const auto& output_desc : output_descriptions) {
            std::shared_ptr<ov::op::v0::Result> result = body->get_results()[output_desc->m_body_value_index];
            const auto& in_value = result->input_value(0);

            // set output name to Tensor to store it for ngraph to cnn conversion
            NGRAPH_SUPPRESS_DEPRECATED_START
            ov::descriptor::set_ov_tensor_legacy_name(
                in_value.get_tensor(),
                op::util::create_ie_output_name(if_node->output(output_desc->m_output_index)));
            NGRAPH_SUPPRESS_DEPRECATED_END
            for (const auto& input : if_node->output(output_desc->m_output_index).get_target_inputs()) {
                input.replace_source_output(result->get_input_source_output(0));
            }
        }
        is_applicable = true;
        f->add_sinks(body->get_sinks());
        copy_runtime_info(if_node, body->get_ops());
    }
    return is_applicable;
}
