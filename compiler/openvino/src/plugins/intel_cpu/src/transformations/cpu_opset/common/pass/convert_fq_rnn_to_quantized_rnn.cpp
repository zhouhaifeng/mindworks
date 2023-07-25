// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "convert_fq_rnn_to_quantized_rnn.hpp"

#include <algorithm>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/rt_info.hpp>
#include "ngraph/except.hpp"
#include "ngraph/node_output.hpp"
#include "ngraph/type/element_type.hpp"
#include <ov_ops/type_relaxed.hpp>

#include "ie_common.h"
#include "itt.hpp"
#include "openvino/core/type/element_type.hpp"

#include <stdexcept>
#include <vector>
#include <cstdlib>

ov::intel_cpu::ConvertFqRnnToQuantizedRnn::ConvertFqRnnToQuantizedRnn() {
    MATCHER_SCOPE(ConvertFqRnnToQuantizedRnn);

    auto X_m = ngraph::pattern::any_input();
    auto convert_X = ngraph::pattern::wrap_type<ngraph::opset9::Convert>({X_m});
    auto input_shift_X = ngraph::pattern::wrap_type<ngraph::opset9::Constant>();
    auto subtract_X = ngraph::pattern::wrap_type<ngraph::opset9::Subtract>({convert_X, input_shift_X});
    auto input_scale_X = ngraph::pattern::wrap_type<ngraph::opset9::Constant>();

    auto deq_X = std::make_shared<ngraph::pattern::op::Or>(
        OutputVector{
            ngraph::pattern::wrap_type<ngraph::opset9::Multiply>({convert_X, input_scale_X}),
            ngraph::pattern::wrap_type<ngraph::opset9::Multiply>({subtract_X, input_scale_X}),
        });

    auto H_m = ngraph::pattern::any_input();
    auto convert_H = ngraph::pattern::wrap_type<ngraph::opset9::Convert>({H_m});
    auto input_shift_H = ngraph::pattern::wrap_type<ngraph::opset9::Constant>();
    auto subtract_H = ngraph::pattern::wrap_type<ngraph::opset9::Subtract>({convert_H, input_shift_H});
    auto input_scale_H = ngraph::pattern::wrap_type<ngraph::opset9::Constant>();

    auto deq_H = std::make_shared<ngraph::pattern::op::Or>(
        OutputVector{
            ngraph::pattern::wrap_type<ngraph::opset9::Multiply>({convert_H, input_scale_H}),
            ngraph::pattern::wrap_type<ngraph::opset9::Multiply>({subtract_H, input_scale_H}),
        });

    auto H_as_const = ngraph::pattern::wrap_type<ngraph::opset9::Constant>();
    auto H_in = std::make_shared<ngraph::pattern::op::Or>(
        OutputVector {
            deq_H,
            H_as_const
        });

    auto cell_state_m = ngraph::pattern::any_input(); // for LSTM
    auto sequence_length_m = ngraph::pattern::any_input(); // for Sequences

    auto W_m = ngraph::pattern::wrap_type<ngraph::opset9::Constant>();
    auto convert_W = ngraph::pattern::wrap_type<ngraph::opset9::Convert>({W_m});
    auto weights_scale_W = ngraph::pattern::wrap_type<ngraph::opset9::Constant>();
    auto deq_W = ngraph::pattern::wrap_type<ngraph::opset9::Multiply>({convert_W, weights_scale_W});

    auto R_m = ngraph::pattern::wrap_type<ngraph::opset9::Constant>();
    auto convert_R = ngraph::pattern::wrap_type<ngraph::opset9::Convert>({R_m});
    auto weights_scale_R = ngraph::pattern::wrap_type<ngraph::opset9::Constant>();
    auto deq_R = ngraph::pattern::wrap_type<ngraph::opset9::Multiply>({convert_R, weights_scale_R});

    const auto B_m = ngraph::pattern::wrap_type<ngraph::opset9::Constant>();

    auto lstm_seq_m  = ngraph::pattern::wrap_type<ngraph::opset9::LSTMSequence>({deq_X, H_in, cell_state_m, sequence_length_m, deq_W, deq_R, B_m});
    auto gru_seq_m   = ngraph::pattern::wrap_type<ngraph::opset9::GRUSequence> ({deq_X, H_in,               sequence_length_m, deq_W, deq_R, B_m});

    auto rnn_pattern = std::make_shared<ngraph::pattern::op::Or>(
        OutputVector {
            lstm_seq_m,
            gru_seq_m
        });

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto rnn = m.get_match_root();

        if (!rnn || transformation_callback(rnn))
            return false;

        const auto& pattern_map  = m.get_pattern_value_map();
        const auto& activation   = pattern_map.at(X_m);
        const auto  hidden_state_it = pattern_map.find(H_m);

        ngraph::Output<ngraph::Node> hidden_state;
        if (hidden_state_it != pattern_map.end()) { // is it H(i8/u8) -> dequantized -> RNN pattern?
            hidden_state = hidden_state_it->second;
        } else {
            hidden_state = pattern_map.at(H_as_const); // if not, then it is just H (f32 const) -> RNN
        }

        const auto& weights      = pattern_map.at(W_m);
        const auto& r_weights    = pattern_map.at(R_m);
        const auto& bias         = pattern_map.at(B_m);

        std::shared_ptr<ngraph::Node> rnn_quantized;

        if (const auto lstm_seq = ngraph::as_type_ptr<ngraph::opset9::LSTMSequence>(rnn)) {
            const auto& cell_state = pattern_map.at(cell_state_m);
            const auto& sequence_length = pattern_map.at(sequence_length_m);

            // @todo prototype removal of unnecessary fq between two consecutive rnn nodes
            auto rnn_quantized_tr = std::make_shared<op::TypeRelaxed<ngraph::opset9::LSTMSequence>>(
                element::TypeVector{ element::f32, element::f32, element::f32, element::f32, element::f32, element::f32, element::f32 },
                element::TypeVector{ element::f32, element::f32, element::f32 },
                op::TemporaryReplaceOutputType(activation, element::f32).get(),
                op::TemporaryReplaceOutputType(hidden_state, element::f32).get(),
                op::TemporaryReplaceOutputType(cell_state, element::f32).get(),
                op::TemporaryReplaceOutputType(sequence_length, element::f32).get(),
                op::TemporaryReplaceOutputType(weights, element::f32).get(),
                op::TemporaryReplaceOutputType(r_weights, element::f32).get(),
                op::TemporaryReplaceOutputType(bias, element::f32).get(),
                lstm_seq->get_hidden_size(),
                lstm_seq->get_direction(),
                lstm_seq->get_activations_alpha(),
                lstm_seq->get_activations_beta(),
                lstm_seq->get_activations(),
                lstm_seq->get_clip());

            rnn_quantized_tr->set_overridden_output_type(hidden_state.get_element_type(), 1);
            rnn_quantized = rnn_quantized_tr;
        } else if (const auto gru_seq = ngraph::as_type_ptr<ngraph::opset9::GRUSequence>(rnn)) {
            const auto& sequence_length = pattern_map.at(sequence_length_m);

            auto rnn_quantized_tr = std::make_shared<op::TypeRelaxed<ngraph::opset9::GRUSequence>>(
                std::vector<ngraph::element::Type>{ element::f32, element::f32, element::f32, element::f32, element::f32, element::f32},
                std::vector<ngraph::element::Type>{ element::f32, element::f32 },
                op::TemporaryReplaceOutputType(activation, element::f32).get(),
                op::TemporaryReplaceOutputType(hidden_state, element::f32).get(),
                op::TemporaryReplaceOutputType(sequence_length, element::f32).get(),
                op::TemporaryReplaceOutputType(weights, element::f32).get(),
                op::TemporaryReplaceOutputType(r_weights, element::f32).get(),
                op::TemporaryReplaceOutputType(bias, element::f32).get(),
                gru_seq->get_hidden_size(),
                gru_seq->get_direction(),
                gru_seq->get_activations(),
                gru_seq->get_activations_alpha(),
                gru_seq->get_activations_beta(),
                gru_seq->get_clip(),
                gru_seq->get_linear_before_reset());

            rnn_quantized_tr->set_overridden_output_type(hidden_state.get_element_type(), 1);
            rnn_quantized = rnn_quantized_tr;
        } else {
            return false;
        }

        // input scales (Multiply per tensor) and weights_scales (Multiply per multiple dimensions) must be present
        const auto& input_scale_output   = pattern_map.at(input_scale_X);
        const auto& weights_scale_output = pattern_map.at(weights_scale_W);
        // extract constant values
        const auto input_scale_constant   = std::dynamic_pointer_cast<ngraph::opset9::Constant>(input_scale_output.get_node_shared_ptr());
        const auto weights_scale_constant = std::dynamic_pointer_cast<ngraph::opset9::Constant>(weights_scale_output.get_node_shared_ptr());

        if (!input_scale_constant || !weights_scale_constant)
            return false;

        const float* input_scale_ptr = input_scale_constant->get_data_ptr<float>();
        if (*input_scale_ptr == 0.f)
            OPENVINO_THROW("Cannot handle zero input scale");

        const float input_scale  = 1 / *input_scale_ptr;
        std::vector<float> weights_scales  = weights_scale_constant->get_vector<float>();

        // transform dequantization scales into quantization ones
        std::transform(weights_scales.begin(), weights_scales.end(), weights_scales.begin(), [](float& scale) { return 1 / scale; });

        auto& runtime_info = rnn_quantized->get_rt_info();

        // use runtime information to store input and weight scales
        runtime_info["inputScale"]    = input_scale;
        runtime_info["weightsScales"] = weights_scales;

        // input shift (Subtract) is optional
        const auto input_shift_it = pattern_map.find(input_shift_X);

        if (input_shift_it != pattern_map.end()) {
            const auto  input_shift_constant = std::dynamic_pointer_cast<ngraph::opset9::Constant>(input_shift_it->second.get_node_shared_ptr());
            const float* input_shift_ptr      = input_shift_constant->get_data_ptr<float>();
            runtime_info["inputShift"] = *input_shift_ptr;
        }

        auto H_outputs = rnn->output(1).get_target_inputs();
        rnn_quantized->set_friendly_name(rnn->get_friendly_name());
        ngraph::copy_runtime_info(rnn, rnn_quantized);
        ngraph::replace_node(rnn, rnn_quantized);

        /* in case of pattern:
         * H(u8,i8) -> dequantize -> RNN
         * dequantize has to be inserted after H output port since
         * oneDNN supports only equal data types on H in/out ports
         * either: u8u8, i8i8 or f32f32 */
        if (hidden_state_it != pattern_map.end()) {
            const auto& convert  = pattern_map.at(convert_H).get_node_shared_ptr();
            const auto  subtract_it = pattern_map.find(subtract_H);
            const auto& multiply = rnn->get_input_node_shared_ptr(1);

            auto new_convert  = convert->clone_with_new_inputs({rnn_quantized->output(1)});
            std::shared_ptr<Node> multiply_in = new_convert;
            // dequantize with subtract
            if (subtract_it != pattern_map.end()) {
                const auto subtract = std::dynamic_pointer_cast<ngraph::opset9::Subtract>(subtract_it->second.get_node_shared_ptr());
                auto new_subtract = subtract->clone_with_new_inputs({rnn_quantized->output(1), subtract->input_value(1)});
                multiply_in = new_subtract;
            }

            auto new_multiply = multiply->clone_with_new_inputs({multiply_in, multiply->input_value(1)});
            new_multiply->set_friendly_name(rnn_quantized->get_friendly_name() + ".1");

            for (auto output : H_outputs) {
                output.replace_source_output(new_multiply);
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(rnn_pattern, matcher_name);
    this->register_matcher(m, callback);
}
