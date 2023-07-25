// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/fq_conv_fq_affine.hpp"

namespace SubgraphTestsDefinitions {

std::string FqConvFqAffineTest::getTestCaseName(const testing::TestParamInfo<FqConvFqAffineTestParamsSet>& obj) {
    FqSpecificParams fqParams;
    ConvParams convParams;
    bool permute;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(fqParams, convParams, permute, netPrecision, inputShapes, targetDevice, config) = obj.param;

    std::vector<size_t> levels;
    std::vector<float> inputArg;
    std::tie(levels, inputArg) = fqParams;

    std::vector<size_t> kernelShape;
    std::vector<size_t> strides;
    size_t inputChannels;
    size_t outputChannels;
    std::tie(kernelShape, strides, inputChannels, outputChannels) = convParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "LEVELS=" << CommonTestUtils::vec2str(levels) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "trgDev=" << targetDevice;
    for (auto const& configItem : config) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
     if (inputArg.size() == 3) {
        result << "_inputArg=" << inputArg[0] << "_" << inputArg[1] << "_" << inputArg[2];
    }
    result << "_KERNEL=" << CommonTestUtils::vec2str(kernelShape) << "_";
    result << "STRIDES=" << CommonTestUtils::vec2str(strides) << "_";
    result << "IC=" << inputChannels << "_";
    result << "OC=" << outputChannels << "_";
    result << "permute=" << permute << "\n";
    return result.str();
}

void FqConvFqAffineTest::SetUp() {
    FqSpecificParams fqParams;
    ConvParams convParams;
    bool permute;
    std::vector<size_t> inputShape;
    std::map<std::string, std::string> config;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(fqParams, convParams, permute, netPrecision, inputShape, targetDevice, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());

    std::vector<size_t> levels;
    std::vector<float> inputArg;
    std::tie(levels, inputArg) = fqParams;
    if (inputArg.size() == 3) {
        inputDataMin = inputArg[0];
        inputDataMax = inputArg[1];
        inputDataResolution = inputArg[2];
    }

    std::vector<size_t> kernelShape;
    std::vector<size_t> strides;
    size_t inputChannels;
    size_t outputChannels;
    std::tie(kernelShape, strides, inputChannels, outputChannels) = convParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    const int seed = 0;
    std::mt19937 gen(seed);

    auto inputFQNode = ngraph::builder::makeFakeQuantize(params[0], ngraph::element::f32, levels[0], std::vector<size_t>{},
        { inputDataMin }, { inputDataMax }, { inputDataMin }, { inputDataMax });
    auto inputFQ = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(inputFQNode);

    std::vector<size_t> convInputShape = {1, inputChannels, 1, inputShape[0] * inputShape[1] / inputChannels};
    auto reshapePattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, convInputShape);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(inputFQ, reshapePattern1, false);

    auto filterWeightsNode = ngraph::builder::makeConstant<float>(ngPrc, {outputChannels, inputChannels, kernelShape[0], kernelShape[1]},
                                                                  { 1.0f });

    auto convLowNode = ngraph::builder::makeConstant(ngraph::element::f32, std::vector<size_t>{ 1 }, std::vector<float>{inputDataMin});
    auto convHighNode = ngraph::builder::makeConstant(ngraph::element::f32, std::vector<size_t>{ 1 }, std::vector<float>{inputDataMax});
    auto convWeightsFQNode = std::make_shared<ngraph::opset1::FakeQuantize>(filterWeightsNode,
        convLowNode, convHighNode, convLowNode, convHighNode, levels[1]);
    auto convWeightsFQ = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(convWeightsFQNode);

    auto conv = std::make_shared<ngraph::opset1::Convolution>(reshape1, convWeightsFQ, strides, std::vector<ptrdiff_t>{ 0, 0 },
                                                              std::vector<ptrdiff_t>{ 0, 0 }, std::vector<size_t>{ 1, 1 },
                                                              ngraph::op::PadType::VALID);
    auto biasesWeightsNode = ngraph::builder::makeConstant(ngPrc, {}, std::vector<float>{ 0.0f });
    auto add = std::make_shared<ngraph::opset1::Add>(conv, biasesWeightsNode);

    auto widthAfterConv = (convInputShape[3] - kernelShape[1]) / strides[1] + 1;
    auto heightAfterConv = (convInputShape[2] - kernelShape[0]) / strides[0] + 1;
    std::vector<size_t> outFormShapes = {1,  outputChannels * widthAfterConv * heightAfterConv };

    ngraph::Output<ngraph::Node> nodeBeforeReshape;
    if (permute) {
        auto permuteOrder = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64,
                                                                       ngraph::Shape{4},
                                                                       ngraph::Shape{{0, 3, 2, 1}});
        auto transpose = std::make_shared<ngraph::opset1::Transpose>(add, permuteOrder);
        nodeBeforeReshape = transpose;
    } else {
        nodeBeforeReshape = add;
    }

    auto reshapePattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes);
    auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(nodeBeforeReshape, reshapePattern2, false);

    auto matMulWeightsNode = ngraph::builder::makeConstant<float>(ngPrc, {outFormShapes[1], outFormShapes[1]}, { 1.0f });
    auto matMulLowNode = ngraph::builder::makeConstant(ngraph::element::f32, std::vector<size_t>{ 1 }, std::vector<float>{inputDataMin});
    auto matMulHighNode = ngraph::builder::makeConstant(ngraph::element::f32, std::vector<size_t>{ 1 }, std::vector<float>{inputDataMax});
    auto matMulWeightsFQNode = std::make_shared<ngraph::opset1::FakeQuantize>(matMulWeightsNode,
        matMulLowNode, matMulHighNode, matMulLowNode, matMulHighNode, levels[1]);
    auto matMulWeightsFQ = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(matMulWeightsFQNode);

    auto matmul = std::make_shared<ngraph::opset1::MatMul>(reshape2, matMulWeightsFQ, false, true);

    function = std::make_shared<ngraph::Function>(matmul, params, "fqConvfqAffine");
}

InferenceEngine::Blob::Ptr FqConvFqAffineTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), inputDataMax - inputDataMin, inputDataMin, 1 / inputDataResolution,
                                            seed);
}
}  // namespace SubgraphTestsDefinitions
