// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/move_fake_quantize_function.hpp"
#include <low_precision/relu.hpp>

#include <ngraph/opsets/opset1.hpp>
#include "ov_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"

#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> MoveFakeQuantize::get(
    const ngraph::element::Type inputPrecision,
    const std::vector<ngraph::PartialShape>& inputShapes,
    const size_t concatInputsCount,
    const std::vector<FakeQuantizeOnDataWithConstant>& fqOnDataBefore,
    const DequantizationOperations::Convert& convertBefore,
    const DequantizationOperations& dequantizationBefore,
    const std::string& operation,
    const FakeQuantizeOnDataWithConstant& fqOnDataAfter,
    const DequantizationOperations::Convert& convertAfter,
    const DequantizationOperations& dequantizationAfter,
    const std::vector<ov::Any>& concatAttributes,
    const ngraph::element::Type precisionAfterOperation,
    const std::int64_t& axis,
    const bool oneInputWithSplit) {
    std::vector<std::shared_ptr<ngraph::opset1::Parameter>> inputs(oneInputWithSplit ? 1 : concatInputsCount);
    std::vector<ov::Output<ov::Node>> concatParents(concatInputsCount);
    if (oneInputWithSplit) {
        auto newInputShape = inputShapes[0];
        int channels = 0;
        bool channelsWasIdentified = false;
        for (const auto inputShape : inputShapes) {
            if (inputShape[axis].is_static()) {
                channels += inputShape[axis].get_length();
                channelsWasIdentified = true;
            }
        }

        if (channelsWasIdentified) {
            newInputShape[axis] = channels;
        }

        inputs[0] = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, newInputShape);
        inputs[0]->set_friendly_name("input");

        const auto axis_constant = std::make_shared<ngraph::opset1::Constant>(element::i32, Shape{}, std::vector<int64_t>({axis}));
        std::vector<int> split_lengths_values(inputShapes.size(), 1);
        split_lengths_values[split_lengths_values.size() - 1] = channels - (split_lengths_values.size() - 1);
        const auto split_lengths = std::make_shared<ngraph::opset1::Constant>(element::i32, Shape{split_lengths_values.size()}, split_lengths_values);
        const auto split = std::make_shared<ngraph::opset1::VariadicSplit>(inputs[0], axis_constant, split_lengths);
        for (size_t i = 0; i < concatInputsCount; i++) {
            // added unary op to avoid Split -> Concat pair elimination
            concatParents[i] = std::make_shared<ngraph::opset1::Sigmoid>(split->output(i));
        }
    } else {
        for (size_t i = 0; i < concatInputsCount; i++) {
            auto ind = 0;
            if (inputShapes.size() != 1) {
                ind = i;
            }
            inputs[i] = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShapes[ind]);
            inputs[i]->set_friendly_name(std::string("input") + "_" + std::to_string(i + 1));
            concatParents[i] = inputs[i];
        }
    }

    if (!fqOnDataBefore.empty()) {
        for (size_t i = 0; i < concatInputsCount; i++) {
            size_t ind = i;
            if (fqOnDataBefore.size() == 1) {
                ind = 0;
            }
            if (operation == "relu") {
                auto relu = std::make_shared<ngraph::opset1::Relu>(concatParents[i]);
                concatParents[i] = makeFakeQuantize(relu, inputPrecision, fqOnDataBefore[ind]);
            } else {
                concatParents[i] = makeFakeQuantize(concatParents[i], inputPrecision, fqOnDataBefore[ind]);
            }
            concatParents[i].get_node()->set_friendly_name(std::string("concat_fq") + "_" + std::to_string(i + 1));
            if (!convertBefore.empty()) {
                concatParents[i] = std::make_shared<opset1::Convert>(concatParents[i], convertBefore.outPrecision);
            }
            if (!dequantizationBefore.empty()) {
                concatParents[i] = makeDequantization(concatParents[i], dequantizationBefore);
            }
        }
    }

    const auto concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector(concatParents.begin(), concatParents.end()),
        axis);
    concat->set_friendly_name("concat");
    std::shared_ptr<ngraph::Node> parent = concat;
    addAttributes({ parent }, concatAttributes);
    if (!fqOnDataAfter.empty()) {
        std::shared_ptr<ngraph::Node> fq;
        if (operation == "relu") {
            auto relu = std::make_shared<ngraph::opset1::Relu>(concat->output(0));
            fq = makeFakeQuantize(relu, inputPrecision, fqOnDataAfter);
        } else {
            fq = makeFakeQuantize(concat, inputPrecision, fqOnDataAfter);
        }
        fq->set_friendly_name("fakeQuantizeAfter");
        parent = fq;
        if (!convertAfter.empty()) {
            parent = std::make_shared<opset1::Convert>(parent, convertAfter.outPrecision);
        }
        if (!dequantizationAfter.empty()) {
            parent = makeDequantization(parent, dequantizationAfter);
        }
    }
    parent->set_friendly_name("output");
    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(parent) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector(inputs.begin(), inputs.end()),
        "MoveFakeQuantize");
    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
