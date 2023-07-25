// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/network_helper.hpp"
#include "low_precision/layer_transformation.hpp"

#include "ngraph/opsets/opset1.hpp"

#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "lpt_ngraph_functions/strided_slice_function.hpp"

using namespace ngraph::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> StridedSliceFunction::getOriginal(
    const ngraph::element::Type inputPrecision,
    const ngraph::PartialShape& inputShape,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization,
    const std::vector<int64_t>& begin,
    const std::vector<int64_t>& end,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& beginMask,
    const std::vector<int64_t>& endMask,
    const std::vector<int64_t>& newAxisMask,
    const std::vector<int64_t>& shrinkAxisMask,
    const std::vector<int64_t>& elipsisMask) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    input->set_friendly_name("input");
    const auto deq = makeDequantization(input, dequantization);

    const auto beginParam = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{ begin.size() }, begin);
    beginParam->set_friendly_name("begin");
    const auto endParam = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{ end.size() }, end);
    endParam->set_friendly_name("end");
    const auto stridesParam = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{ strides.size() }, strides);
    stridesParam->set_friendly_name("strides");

    const auto stridedSlice = std::make_shared<ngraph::opset1::StridedSlice>(
        deq, beginParam, endParam, stridesParam,
        beginMask, endMask, newAxisMask,
        shrinkAxisMask, elipsisMask);
    stridedSlice->set_friendly_name("StridedSlice");

    const auto res = std::make_shared<ngraph::opset1::Result>(stridedSlice);
    const auto function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{ res },
        ngraph::ParameterVector{ input },
        "StridedSliceTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> StridedSliceFunction::getOriginal(
    const ngraph::element::Type inputPrecision,
    const ngraph::PartialShape& inputShape,
    const ngraph::builder::subgraph::FakeQuantizeOnData& fakeQuantize,
    const std::vector<int64_t>& begin,
    const std::vector<int64_t>& end,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& beginMask,
    const std::vector<int64_t>& endMask,
    const std::vector<int64_t>& newAxisMask,
    const std::vector<int64_t>& shrinkAxisMask,
    const std::vector<int64_t>& elipsisMask) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    input->set_friendly_name("input");
    const auto fqOnData = makeFakeQuantize(input, inputPrecision, fakeQuantize);

    const auto beginParam = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{ begin.size() }, begin);
    beginParam->set_friendly_name("begin");
    const auto endParam = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{ end.size() }, end);
    endParam->set_friendly_name("end");
    const auto stridesParam = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{ strides.size() }, strides);
    stridesParam->set_friendly_name("strides");

    const auto stridedSlice = std::make_shared<ngraph::opset1::StridedSlice>(
        fqOnData, beginParam, endParam, stridesParam,
        beginMask, endMask, newAxisMask,
        shrinkAxisMask, elipsisMask);
    stridedSlice->set_friendly_name("StridedSlice");

    const auto res = std::make_shared<ngraph::opset1::Result>(stridedSlice);
    const auto function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{ res },
        ngraph::ParameterVector{ input },
        "StridedSliceTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> StridedSliceFunction::getReference(
    const ngraph::element::Type inputPrecision,
    const ngraph::PartialShape& inputShape,
    const std::vector<int64_t>& begin,
    const std::vector<int64_t>& end,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& beginMask,
    const std::vector<int64_t>& endMask,
    const std::vector<int64_t>& newAxisMask,
    const std::vector<int64_t>& shrinkAxisMask,
    const std::vector<int64_t>& elipsisMask,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    input->set_friendly_name("input");
    const auto deqBefore = makeDequantization(input, dequantizationBefore);

    const auto beginParam = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{ begin.size() }, begin);
    beginParam->set_friendly_name("begin");
    const auto endParam = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{ end.size() }, end);
    endParam->set_friendly_name("end");
    const auto stridesParam = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{ strides.size() }, strides);
    stridesParam->set_friendly_name("strides");

    const auto stridedSlice = std::make_shared<ngraph::opset1::StridedSlice>(
        deqBefore, beginParam, endParam, stridesParam,
        beginMask, endMask, newAxisMask,
        shrinkAxisMask, elipsisMask);

    const auto deqAfter = makeDequantization(stridedSlice, dequantizationAfter);
    deqAfter->set_friendly_name("StridedSlice");

    const auto res = std::make_shared<ngraph::opset1::Result>(deqAfter);
    const auto function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{ res },
        ngraph::ParameterVector{ input },
        "StridedSliceTransformation");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
