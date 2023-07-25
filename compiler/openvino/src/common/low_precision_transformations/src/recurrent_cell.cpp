﻿// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/recurrent_cell.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset1.hpp>

#include <memory>
#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/or.hpp>

#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/skip_cleanup_attribute.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

RecurrentCellTransformation::RecurrentCellTransformation(const Params& params) : LayerTransformation(params) {
    const auto X = ngraph::pattern::any_input();
    const auto H = ngraph::pattern::any_input();
    const auto C = ngraph::pattern::any_input();
    const auto S = ngraph::pattern::any_input();
    const auto W = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto R = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto B = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();

    const auto H_as_const = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();

    const auto fq_X = wrap_fake_quantize(X);
    const auto fq_H = wrap_fake_quantize(H);
    const auto fq_W = wrap_fake_quantize(W);
    const auto fq_R = wrap_fake_quantize(R);

    const auto dequantization_X = wrap_dequantization(ngraph::pattern::any_input(), true);
    const auto dequantization_H = wrap_dequantization(ngraph::pattern::any_input(), true);
    const auto dequantization_W = wrap_dequantization(ngraph::pattern::any_input(), true);
    const auto dequantization_R = wrap_dequantization(ngraph::pattern::any_input(), true);

    const auto dequantization_without_subtract_X = wrap_dequantization(ngraph::pattern::any_input(), false);
    const auto dequantization_without_subtract_H = wrap_dequantization(ngraph::pattern::any_input(), false);
    const auto dequantization_without_subtract_W = wrap_dequantization(ngraph::pattern::any_input(), false);
    const auto dequantization_without_subtract_R = wrap_dequantization(ngraph::pattern::any_input(), false);

    auto X_in = std::make_shared<ngraph::pattern::op::Or>(
        OutputVector{
            fq_X, dequantization_X, dequantization_without_subtract_X
        });

    auto H_in = std::make_shared<ngraph::pattern::op::Or>(
        OutputVector{
            H_as_const, fq_H, dequantization_H, dequantization_without_subtract_H
        });

    auto W_in = std::make_shared<ngraph::pattern::op::Or>(
        OutputVector{
            fq_W, dequantization_W, dequantization_without_subtract_W
        });

    auto R_in = std::make_shared<ngraph::pattern::op::Or>(
        OutputVector{
            fq_R, dequantization_R, dequantization_without_subtract_R
        });

    const auto lstm_seq = ngraph::pattern::wrap_type<ngraph::opset5::LSTMSequence>(
        {X_in, H_in, C, S, W_in, R_in, B});
    const auto gru_seq  = ngraph::pattern::wrap_type<ngraph::opset5::GRUSequence>(
        {X_in, H_in,    S, W_in, R_in, B});

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }

        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(
        std::make_shared<pattern::op::Or>(
            OutputVector {
                lstm_seq,
                gru_seq
            }),
        "RecurrentCellTransformation");

    this->register_matcher(m, callback);
}

bool RecurrentCellTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    const auto lstm = m.get_match_root();
    if (!canBeTransformed(context, lstm)) {
        return false;
    }
    for (size_t parentIndex = 0ul; parentIndex < lstm->get_input_size(); parentIndex++) {
        auto lstm_parent = lstm->get_input_node_shared_ptr(parentIndex);
        if (is_type<ngraph::opset1::FakeQuantize>(lstm_parent)) {
            auto fq_parent = lstm_parent->get_input_node_shared_ptr(0);
            if (is_type<ngraph::opset5::Constant>(fq_parent)) {
                auto fq_node = as_type_ptr<ngraph::opset1::FakeQuantize>(lstm_parent);
                const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fq_node);
                const auto precisionsAttribute = getAttributeFromOutput<PrecisionsAttribute>(lstm_parent);
                const auto precisions = precisionsAttribute.empty()
                                            ? defaultPrecisions
                                            : precisionsAttribute.as<PrecisionsAttribute>().value();
                const DataPrecision dataPrecision = getDataPrecision(lstm_parent, quantizationDetails, precisions);
                auto QDQ = NetworkHelper::decomposeFakeQuantize(fq_node,
                                                                  dataPrecision.precision,
                                                                  dataPrecision.min,
                                                                  dataPrecision.max,
                                                                  dataPrecision.hasZeroPoint,
                                                                  updatePrecisions);
                std::shared_ptr<ngraph::Node> new_fq = std::get<0>(QDQ);
                std::shared_ptr<ngraph::Node> deq_multiply = std::get<1>(QDQ);
                if (deq_multiply == nullptr || new_fq == nullptr) {
                    return false;
                }

                std::shared_ptr<ngraph::Node> convert;
                auto multiply_parent = deq_multiply->get_input_node_shared_ptr(0);
                if (is_type<ngraph::opset1::Subtract>(multiply_parent)) {
                    convert = multiply_parent->get_input_node_shared_ptr(0);
                } else {
                    convert = multiply_parent;
                }
                ov::disable_constant_folding(convert);
                propagateSkipCleanupAttribute(deq_multiply);

                this->register_new_node(new_fq);
                updateOutput(context, deq_multiply, new_fq);
            } else {
                continue;
            }
        } else {
            if (is_type<ngraph::opset1::Multiply>(lstm_parent)) {
                auto multiply = lstm_parent->get_input_node_shared_ptr(0);
                ov::disable_constant_folding(multiply);
                propagateSkipCleanupAttribute(lstm_parent);
            }
            continue;
        }
    }
    return true;
}

bool RecurrentCellTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> lstm) const {
    std::shared_ptr<ov::Node> W, R;

    if (is_type<opset5::LSTMSequence>(lstm)) {
        W = lstm->get_input_node_shared_ptr(4);
        R = lstm->get_input_node_shared_ptr(5);
    } else if (is_type<opset5::GRUSequence>(lstm)) {
        W = lstm->get_input_node_shared_ptr(3);
        R = lstm->get_input_node_shared_ptr(4);
    } else {
        return false;
    }

    return true;
}

bool RecurrentCellTransformation::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return true;
}

void RecurrentCellTransformation::propagateSkipCleanupAttribute(std::shared_ptr<Node> multiply) {
    SkipCleanupAttribute::create(multiply);
    auto multiply_parent = multiply->get_input_node_shared_ptr(0);
    SkipCleanupAttribute::create(multiply_parent);
    if (is_type<ngraph::opset1::Subtract>(multiply_parent)) {
        auto subtract_parent = multiply_parent->get_input_node_shared_ptr(0);
        SkipCleanupAttribute::create(subtract_parent);
    }
}

std::shared_ptr<ov::Node> RecurrentCellTransformation::wrap_fake_quantize(
    const std::shared_ptr<ov::Node> parameter) {
    const auto input_low = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto input_high = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto output_low = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto output_high = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    return ngraph::pattern::wrap_type<opset1::FakeQuantize>({
        parameter,
        input_low,
        input_high,
        output_low,
        output_high});
}

std::shared_ptr<ov::Node> RecurrentCellTransformation::wrap_quantization(
    const std::shared_ptr<ov::Node> parameter) {
    const auto quantization_fake_quantize = wrap_fake_quantize(parameter);
    const auto quantization_convert = ngraph::pattern::wrap_type<ngraph::opset1::Convert>(
        {quantization_fake_quantize});
    return quantization_convert;
}

std::shared_ptr<ov::Node> RecurrentCellTransformation::wrap_dequantization(
    const std::shared_ptr<ov::Node> parameter,
    const bool with_subtract) {
    const auto dequantization_convert = ngraph::pattern::wrap_type<ngraph::opset1::Convert>({parameter});
    const auto subtract_constant = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto dequantization_subtract = ngraph::pattern::wrap_type<ngraph::opset1::Subtract>(
        {dequantization_convert, subtract_constant});
    const auto multiply_constant = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto multiply_parent = with_subtract ? dequantization_subtract : dequantization_convert;
    const auto dequantization_multiply = ngraph::pattern::wrap_type<ngraph::opset1::Multiply>(
        {multiply_parent, multiply_constant});
    return dequantization_multiply;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
