// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_propagation_convertion_function.hpp"
#include <assert.h>
#include <ngraph/opsets/opset1.hpp>

namespace ov {
namespace test {
namespace snippets {

namespace {
std::shared_ptr<ngraph::op::FakeQuantize> make_fake_quantize(
    const Output<Node>& parent,
    const ngraph::PartialShape& inputShape,
    const element::Type inputType,
    const std::vector<float>& fake_quantize_intervals) {
    auto generate = [](const ov::element::Type precision,
        const ngraph::Shape& shape,
        const float initialValue,
        const std::string& name) {
            const auto size = ngraph::shape_size(shape);
            std::vector<float> values(size);
            for (auto i = 0; i < size; ++i) {
                values[i] = static_cast<float>(initialValue + i);
            }
            auto constant = std::make_shared<ngraph::opset1::Constant>(precision, shape, values);
            constant->set_friendly_name(name);
            return constant;
    };

    const auto fakeQuantize = std::make_shared<ngraph::opset1::FakeQuantize>(
        parent,
        generate(inputType, {}, fake_quantize_intervals[0], "inputLow"),
        generate(inputType, {}, fake_quantize_intervals[1], "inputHigh"),
        generate(inputType, {}, fake_quantize_intervals[2], "outputLow"),
        generate(inputType, {}, fake_quantize_intervals[3], "outputHigh"),
        256ul);
    fakeQuantize->set_friendly_name("fakeQuantize");

    return fakeQuantize;
}
} // namespace

PrecisionPropagationConvertionFunction::PrecisionPropagationConvertionFunction(
    const std::vector<ov::PartialShape>& input_shapes,
    const element::Type input_type,
    const std::vector<float>& fake_quantize_intervals) :
    SnippetsFunctionBase(input_shapes, input_type),
    fake_quantize_intervals(fake_quantize_intervals) {
}

std::shared_ptr<ov::Model> PrecisionPropagationConvertionFunction::get(
    const std::vector<ov::PartialShape>& input_shapes,
    const element::Type input_type,
    const std::vector<float>& fake_quantize_intervals) {
    assert(2ull == input_shapes.size());
    assert(4ull == fake_quantize_intervals.size());
    const auto parameter1 = std::make_shared<ngraph::opset1::Parameter>(input_type, input_shapes[0]);
    parameter1->set_friendly_name("parameter1");

    const auto parameter2 = std::make_shared<ngraph::opset1::Parameter>(input_type, input_shapes[1]);
    parameter2->set_friendly_name("parameter2");

    std::shared_ptr<Node> parent = make_fake_quantize(
        parameter1,
        input_shapes[0],
        input_type,
        fake_quantize_intervals);
    parent->set_friendly_name("fakeQuantize");

    parent = std::make_shared<ngraph::opset1::Add>(parent, parameter2);
    parent->set_friendly_name("add");

    const auto result = std::make_shared<ngraph::opset1::Result>(parent);
    result->set_friendly_name("result");

    auto function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{ result },
        ParameterVector{ parameter1, parameter2 },
        "PrecisionPropagationConvertionFunction");
    return function;
}

std::shared_ptr<Model> PrecisionPropagationConvertionFunction::initOriginal() const {
    return get(input_shapes, precision, fake_quantize_intervals);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
