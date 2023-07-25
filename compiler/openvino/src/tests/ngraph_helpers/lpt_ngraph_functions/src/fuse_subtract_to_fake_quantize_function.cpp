// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/fuse_subtract_to_fake_quantize_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ov_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision/network_helper.hpp"

#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> FuseSubtractToFakeQuantizeFunction::get(
    const ngraph::PartialShape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData,
    const DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, inputShape);

    const auto fakeQuantize = makeFakeQuantize(input, ngraph::element::f32, fqOnData);
    const auto lastDequantization = makeDequantization(fakeQuantize, dequantization);
    lastDequantization->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastDequantization) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FuseSubtractToFakeQuantizeFunction");
}

std::shared_ptr<ngraph::Function> FuseSubtractToFakeQuantizeFunction::get(
    const ngraph::PartialShape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData,
    const DequantizationOperations& dequantization,
    const FakeQuantizeOnDataWithConstant& fqOnData2,
    const DequantizationOperations& dequantization2) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, inputShape);

    const auto axis = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{}, 1ul);
    const std::shared_ptr<Node> split = std::make_shared<ngraph::opset1::Split>(input, axis, 2ul);

    const auto fakeQuantize = makeFakeQuantize(split->output(0), ngraph::element::f32, fqOnData);
    fakeQuantize->set_friendly_name("fakeQuantize");
    const auto lastDequantization = makeDequantization(fakeQuantize, dequantization);
    lastDequantization->set_friendly_name("output");

    const auto fakeQuantize2 = makeFakeQuantize(split->output(1), ngraph::element::f32, fqOnData);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    const auto lastDequantization2 = makeDequantization(fakeQuantize2, dequantization);
    lastDequantization2->set_friendly_name("output2");

    ngraph::ResultVector results{
        std::make_shared<ngraph::opset1::Result>(lastDequantization),
        std::make_shared<ngraph::opset1::Result>(lastDequantization2)
    };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FuseSubtractToFakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
