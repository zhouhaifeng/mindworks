// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// DEPRECATED, can't be removed currently due to arm and kmb-plugin dependency (#55568)

#include "shared_test_classes/single_layer/convolution_backprop_data.hpp"

namespace LayerTestsDefinitions {

std::string ConvolutionBackpropDataLayerTest::getTestCaseName(const testing::TestParamInfo<convBackpropDataLayerTestParamsSet>& obj) {
    convBackpropDataSpecificParams convBackpropDataParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::SizeVector outputShapes;
    std::string targetDevice;
    std::tie(convBackpropDataParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, outputShapes, targetDevice) = obj.param;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, outPadding) = convBackpropDataParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "OS=" << CommonTestUtils::vec2str(outputShapes) << "_";
    result << "K" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S" << CommonTestUtils::vec2str(stride) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
    result << "OP=" << CommonTestUtils::vec2str(outPadding) << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void ConvolutionBackpropDataLayerTest::SetUp() {
    convBackpropDataSpecificParams convBackpropDataParams;
    std::vector<size_t> inputShape;
    std::vector<size_t> outputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(convBackpropDataParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, outputShape, targetDevice) = this->GetParam();
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, outPadding) = convBackpropDataParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto convBackpropData = std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(
            ngraph::builder::makeConvolutionBackpropData(paramOuts[0], ngPrc, kernel, stride, padBegin,
                                                        padEnd, dilation, padType, convOutChannels, false, outPadding));
    if (!outputShape.empty()) {
        auto outShape = ngraph::opset3::Constant::create(ngraph::element::i64, {outputShape.size()}, outputShape);
        convBackpropData = std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(
        ngraph::builder::makeConvolutionBackpropData(paramOuts[0], outShape, ngPrc, kernel, stride, padBegin,
                                                        padEnd, dilation, padType, convOutChannels));
    }
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(convBackpropData)};
    function = std::make_shared<ngraph::Function>(results, params, "convolutionBackpropData");
}
}  // namespace LayerTestsDefinitions
