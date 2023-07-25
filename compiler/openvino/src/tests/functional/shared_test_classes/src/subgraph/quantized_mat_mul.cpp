// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/quantized_mat_mul.hpp"
#include "ngraph_functions/builders.hpp"

namespace SubgraphTestsDefinitions {

using ngraph::helpers::QuantizationGranularity;

std::string QuantMatMulTest::getTestCaseName(const testing::TestParamInfo<QuantMatMulLayerTestParamsSet> &obj) {
    QuantParams quantParams0;
    QuantParams quantParams1;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape0;
    InferenceEngine::SizeVector inputShape1;
    QuantRange inputRange0;
    QuantRange inputRange1;
    QuantRange outputRange0;
    QuantRange outputRange1;
    std::string targetDevice;
    std::tie(quantParams0, quantParams1, netPrecision, inputShape0, inputShape1, targetDevice) = obj.param;

    size_t quantLevels0;
    size_t quantLevels1;
    QuantizationGranularity quantGranularity0;
    QuantizationGranularity quantGranularity1;
    InferenceEngine::Precision fqPrec0;
    InferenceEngine::Precision fqPrec1;
    std::tie(quantLevels0, inputRange0, outputRange0, quantGranularity0, fqPrec0) = quantParams0;
    std::tie(quantLevels1, inputRange1, outputRange1, quantGranularity1, fqPrec1) = quantParams1;

    std::ostringstream result;
    result << "IS0=" << CommonTestUtils::vec2str(inputShape0) << "_";
    result << "IS1=" << CommonTestUtils::vec2str(inputShape1) << "_";
    result << "Levels0=" << quantLevels0 << "_";
    result << "Levels1=" << quantLevels1 << "_";
    result << "inputRange0=" << inputRange0.first << "_" << inputRange0.second << "_";
    result << "outputRange0=" << outputRange0.first << "_" << outputRange0.second << "_";
    result << "inputRange1=" << inputRange1.first << "_" << inputRange1.second << "_";
    result << "outputRange1=" << outputRange1.first << "_" << outputRange1.second << "_";
    result << "QuantGranularity0=" << quantGranularity0 << "_";
    result << "QuantGranularity1=" << quantGranularity1 << "_";
    result << "fq0PRC=" << fqPrec0.name() << "_";
    result << "fq1PRC=" << fqPrec1.name() << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void QuantMatMulTest::SetUp() {
    QuantParams quantParams0;
    QuantParams quantParams1;
    InferenceEngine::SizeVector inputShape0;
    InferenceEngine::SizeVector inputShape1;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(quantParams0, quantParams1, netPrecision, inputShape0, inputShape1, targetDevice) = this->GetParam();

    size_t quantLevels0;
    size_t quantLevels1;
    QuantRange inputRange0;
    QuantRange inputRange1;
    QuantRange outputRange0;
    QuantRange outputRange1;
    QuantizationGranularity quantGranularity0;
    QuantizationGranularity quantGranularity1;
    InferenceEngine::Precision fqPrec0;
    InferenceEngine::Precision fqPrec1;
    std::tie(quantLevels0, inputRange0, outputRange0, quantGranularity0, fqPrec0) = quantParams0;
    std::tie(quantLevels1, inputRange1, outputRange1, quantGranularity1, fqPrec1) = quantParams1;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape0, inputShape1});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto makeFakeQuantizeNode = [ngPrc](size_t quantLevels, QuantRange inputRange, QuantRange outputRange,
            QuantizationGranularity quantGranularity, const ngraph::Output<ngraph::Node> &in, std::vector<size_t> inputShape,
            InferenceEngine::Precision prec) -> std::shared_ptr<ngraph::Node> {
        std::vector<size_t> dataFqConstShapes(inputShape.size(), 1);
        if (quantGranularity == ngraph::helpers::Perchannel)
            dataFqConstShapes[1] = inputShape[1];
        size_t constDataSize = ngraph::shape_size(dataFqConstShapes);
        std::vector<float> inputLowData(constDataSize), inputHighData(constDataSize), outputLowData(constDataSize), outputHighData(constDataSize);
        for (int i = 0; i < constDataSize; i++) {
            inputLowData[i] = inputRange.first;
            inputHighData[i] = inputRange.second;
            outputLowData[i] = outputRange.first;
            outputHighData[i] = outputRange.second;
        }
        return ngraph::builder::makeFakeQuantize(in, ngPrc, quantLevels, dataFqConstShapes, inputLowData, inputHighData, outputLowData, outputHighData);
    };

    auto dataFq0 = makeFakeQuantizeNode(quantLevels0, inputRange0, outputRange0, quantGranularity0, paramOuts[0], inputShape0, fqPrec0);
    auto dataFq1 = makeFakeQuantizeNode(quantLevels1, inputRange1, outputRange1, quantGranularity1, paramOuts[1], inputShape1, fqPrec1);

    auto MatMul = std::dynamic_pointer_cast<ngraph::opset3::MatMul>(
            ngraph::builder::makeMatMul(dataFq0, dataFq1));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(MatMul)};
    function = std::make_shared<ngraph::Function>(results, params, "QuantMatMul");
}
}  // namespace SubgraphTestsDefinitions
