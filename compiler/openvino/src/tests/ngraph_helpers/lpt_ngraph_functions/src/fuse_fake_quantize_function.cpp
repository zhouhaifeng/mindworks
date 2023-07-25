// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/fuse_fake_quantize_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ov_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> FuseFakeQuantizeFunction::getOriginal(
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type precisionBeforeAdd,
    const Add& add,
    const ngraph::element::Type precisionBeforeDequantization,
    const DequantizationOperations& dequantization,
    const ngraph::element::Type precisionAfterDequantization,
    const ngraph::element::Type precisionFqOnData,
    const FakeQuantizeOnDataWithConstant& fqOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(add.empty() ? precisionBeforeDequantization : precisionBeforeAdd, inputShape);
    input->set_friendly_name("input");

    std::shared_ptr<Node> parent = input;
    if (!add.empty()) {
        parent = makeElementwise<ngraph::opset1::Add>(parent, add);
    }

    const std::shared_ptr<Node> lastDequantization = makeDequantization(parent, dequantization);

    const std::shared_ptr<Node> fakeQuantize = precisionAfterDequantization == precisionFqOnData ?
        makeFakeQuantize(lastDequantization, precisionFqOnData, fqOnData) :
        makeFakeQuantizeTypeRelaxed(lastDequantization, precisionFqOnData, fqOnData);
    fakeQuantize->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FuseFakeQuantizeFunction");
}

namespace {
std::shared_ptr<ngraph::opset1::Convolution> make_convolution(
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type precisionBefore,
    const std::shared_ptr<Node>& parent,
    const size_t index) {
    const ov::Shape shape = inputShape.to_shape();
    const ov::Shape weightsShape({ shape[1], shape[1], 1ull, 1ull });
    auto weightsConstant = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, weightsShape, std::vector<float>(9, 1.f));
    auto weights = makeFakeQuantize(
        weightsConstant,
        precisionBefore,
        FakeQuantizeOnData(
            255,
            ov::Shape({ shape[1], 1ull, 1ull, 1ull }),
            { -1.27f, -1.27f, -1.27f },
            { 1.28f, 1.28f, 1.28f },
            { -1.27f, -1.27f, -1.27f },
            { 1.28f, 1.28f, 1.28f },
            precisionBefore));

    auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        parent,
        weights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    convolution->set_friendly_name("convolution" + std::to_string(index));
    return convolution;
}
}  // namespace

    std::shared_ptr<ngraph::Function> FuseFakeQuantizeFunction::getReference(
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precisionBeforeAdd,
        const Add& add,
        const ngraph::element::Type precisionBeforeDequantization,
        const DequantizationOperations& dequantization,
        const ngraph::element::Type precisionAfterDequantization,
        const ngraph::element::Type precisionFqOnData,
        const FakeQuantizeOnDataWithConstant& fqOnData) {
        const auto input = std::make_shared<ngraph::opset1::Parameter>(add.empty() ? precisionBeforeDequantization : precisionBeforeAdd, inputShape);
        input->set_friendly_name("input");

        std::shared_ptr<Node> parent = input;
        if (!add.empty()) {
            parent = makeElementwise<ngraph::opset1::Add>(parent, add);
        }

        const std::shared_ptr<Node> lastDequantization = makeDequantization(parent, dequantization);

        auto fqOnDataCopy = fqOnData;
        fqOnDataCopy.outputHighValues = {255.f};
        fqOnDataCopy.outputPrecision = fqOnData.outputPrecision == element::undefined ? ngraph::element::u8 : fqOnData.outputPrecision;

        std::shared_ptr<Node> lastNode = makeFakeQuantizeTypeRelaxed(lastDequantization, precisionFqOnData, fqOnDataCopy);
        lastNode = makeDequantization(
            lastNode,
            {
                lastNode->output(0).get_element_type() != element::f32 ?
                    DequantizationOperations::Convert{element::f32} :
                    DequantizationOperations::Convert{},
                {},
                {{0.01f},
                precisionFqOnData}
            });
        lastNode->set_friendly_name("output");

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastNode) };
        return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FuseFakeQuantizeFunction");
    }

std::shared_ptr<ngraph::Function> FuseFakeQuantizeFunction::get(
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type precisionBefore,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const DequantizationOperations& dequantizationOperations2) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBefore, inputShape);
    input->set_friendly_name("input");

    std::shared_ptr<Node> parent = input;

    if (!fqOnData1.empty()) {
        parent = fqOnData1.outputPrecision == precisionBefore ?
            makeFakeQuantize(parent, precisionBefore, fqOnData1) :
            makeFakeQuantizeTypeRelaxed(parent, precisionBefore, fqOnData1);
        parent->set_friendly_name("fakeQuantize1");
    }

    const std::vector<size_t> kernel = { 3, 3 };
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ngraph::op::PadType padType = ngraph::op::PadType::NOTSET;
    const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;

    parent = std::make_shared<ngraph::opset1::MaxPool>(
        parent,
        stride,
        padBegin,
        padEnd,
        kernel,
        roundingType,
        padType);

    if (!fqOnData2.empty()) {
        parent = makeFakeQuantize(parent, precisionBefore, fqOnData2);
        parent->set_friendly_name("fakeQuantize2");
    }

    if (!dequantizationOperations2.empty()) {
        parent = makeDequantization(parent, dequantizationOperations2);
    }

    ngraph::ResultVector results{
        std::make_shared<ngraph::opset1::Result>(make_convolution(inputShape, precisionBefore, parent, 0)),
        std::make_shared<ngraph::opset1::Result>(make_convolution(inputShape, precisionBefore, parent, 1))
    };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FuseFakeQuantizeFunction");
}

std::shared_ptr<ngraph::Function> FuseFakeQuantizeFunction::get(
    const ngraph::PartialShape& inputShape,
    const std::vector<Branch>& branches,
    const ngraph::element::Type precisionFqOnData,
    const FakeQuantizeOnData& fqOnData) {
    if (branches.size() != 2ul) {
        throw std::runtime_error("unsupported branches count");
    }

    if (branches[0].dequantization.multiply.outPrecision != branches[1].dequantization.multiply.outPrecision) {
        throw std::runtime_error("branch precisions are not equal");
    }

    ngraph::ParameterVector inputs;
    std::vector<std::shared_ptr<Node>> lastDequantizations;
    for (const Branch& branch : branches) {
        const auto input = std::make_shared<ngraph::opset1::Parameter>(branch.precisionBeforeDequantization, inputShape);
        inputs.push_back(input);

        const std::shared_ptr<Node> lastDequantization = makeDequantization(input, branch.dequantization);
        lastDequantizations.push_back(lastDequantization);
    }

    std::shared_ptr<ngraph::opset1::Multiply> multiply = std::make_shared<ngraph::opset1::Multiply>(lastDequantizations[0], lastDequantizations[1]);

    const std::shared_ptr<Node> fakeQuantize = branches[0].dequantization.multiply.outPrecision == precisionFqOnData ?
        makeFakeQuantize(multiply, precisionFqOnData, fqOnData) :
        makeFakeQuantizeTypeRelaxed(multiply, precisionFqOnData, fqOnData);
    fakeQuantize->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };
    return std::make_shared<ngraph::Function>(results, inputs, "FuseFakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
