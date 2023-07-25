// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/insert_transpose_after_convolution_or_pooling.hpp"

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <numeric>
#include <openvino/cc/ngraph/itt.hpp>

#include "log/debug.hpp"
#include "log/log.hpp"

using namespace ov::intel_gna::pass;

bool InsertTransposeAfterConvOrPool::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(InsertTransposeAfterConvOrPool);
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ngraph::opset7::Convolution>(node) == nullptr &&
            std::dynamic_pointer_cast<ngraph::opset7::MaxPool>(node) == nullptr) {
            continue;
        }

        auto next_node = node->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
        if (std::dynamic_pointer_cast<ngraph::opset7::Reshape>(next_node) == nullptr) {
            continue;
        }

        bool found_reshape_to_1d = false;
        std::shared_ptr<ngraph::Node> reshape_node = next_node;
        std::shared_ptr<ngraph::Node> transpose_node = nullptr;
        while ((reshape_node != nullptr || transpose_node != nullptr) && next_node->get_output_size() == 1) {
            auto input_shape = next_node->get_input_shape(0);
            auto output_shape = next_node->get_output_shape(0);
            if (input_shape[1] > 1 && output_shape.back() == std::accumulate(std::begin(output_shape),
                                                                             std::end(output_shape),
                                                                             size_t(1),
                                                                             std::multiplies<size_t>())) {
                found_reshape_to_1d = true;
                break;
            }
            next_node = next_node->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
            reshape_node = std::dynamic_pointer_cast<ngraph::opset7::Reshape>(next_node);
            transpose_node = std::dynamic_pointer_cast<ngraph::opset7::Transpose>(next_node);
        }

        if (!found_reshape_to_1d)
            continue;

        // Search for a convolution after this reshape
        bool found_next_conv_or_pool = false;
        while (next_node->get_output_size() > 0 && next_node->output(0).get_target_inputs().size() > 0 &&
               std::dynamic_pointer_cast<ngraph::opset7::MatMul>(next_node) == nullptr &&
               std::dynamic_pointer_cast<ngraph::op::util::BinaryElementwiseArithmetic>(next_node) == nullptr) {
            next_node = next_node->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
            if (std::dynamic_pointer_cast<ngraph::opset7::Convolution>(next_node) != nullptr ||
                std::dynamic_pointer_cast<ngraph::opset7::MaxPool>(next_node) != nullptr) {
                found_next_conv_or_pool = true;
                break;
            }
        }

        if (!found_next_conv_or_pool)
            continue;

        // check if transpose is supported by GNA
        auto output_shape = node->get_output_shape(0);
        if (output_shape.size() < 3)
            continue;
        std::vector<size_t> transpose_ids;
        for (size_t ix = 0; ix < output_shape.size(); ++ix) {
            if (output_shape[ix] > 1) {
                transpose_ids.push_back(ix);
            }
        }
        if (transpose_ids.size() == 1) {
            continue;
        }
        if (transpose_ids.size() != 2) {
            THROW_GNA_EXCEPTION << "Unable to insert transpose after: " << node->get_friendly_name()
                                << " number of dimensions to transpose: " << transpose_ids.size();
        }
        size_t min, max;
        std::tie(min, max) = std::minmax(output_shape[transpose_ids[0]], output_shape[transpose_ids[1]]);
        if (min > 8 || max % 8 != 0) {
            THROW_GNA_EXCEPTION << "Unable to insert transpose after: " << node->get_friendly_name()
                                << " min dimension size: " << min << " max dimension size: " << max;
        }

        log::debug() << "Insert Transpose after " << node->get_friendly_name() << "\n";

        auto consumers = node->output(0).get_target_inputs();

        ngraph::Shape transposeInShape = output_shape;
        std::swap(transposeInShape[transpose_ids[0]], transposeInShape[transpose_ids[1]]);
        auto reshapeConstBefore = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                             ngraph::Shape{transposeInShape.size()},
                                                                             transposeInShape);
        auto reshapeBefore = std::make_shared<ngraph::opset7::Reshape>(node, reshapeConstBefore, false);
        reshapeBefore->set_friendly_name(node->get_friendly_name() + "/reshape_out");
        ngraph::copy_runtime_info(node, {reshapeBefore, reshapeConstBefore});

        auto transpose_order = transposeInShape.size() == 3 ? ngraph::Shape{0, 2, 1} : ngraph::Shape{0, 3, 1, 2};
        auto transpose_order_const = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                                      ngraph::Shape{transpose_order.size()},
                                                                      transpose_order);
        auto transpose = std::make_shared<ngraph::opset7::Transpose>(reshapeBefore, transpose_order_const);
        transpose->set_friendly_name(node->get_friendly_name() + "/transpose_out");
        ngraph::copy_runtime_info(node, {transpose, transpose_order_const});

        for (auto& input : consumers) {
            input.replace_source_output(transpose);
        }
        is_graph_modfied = true;
    }

    return is_graph_modfied;
}
