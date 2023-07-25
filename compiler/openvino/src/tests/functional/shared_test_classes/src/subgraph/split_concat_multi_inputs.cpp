// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/split_concat_multi_inputs.hpp"
#include "ngraph_functions/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string SplitConcatMultiInputsTest::getTestCaseName(testing::TestParamInfo<SplitConcatMultiInputsParams> obj) {
    std::vector<size_t> inputShape;
    size_t splitsNum;
    std::map<std::string, std::string> config;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    bool withFC;
    std::tie(netPrecision, targetName, config, inputShape, splitsNum, withFC) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "SplitsN=" << splitsNum << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "trgDev=" << targetName << "_";
    result << "FC=" << withFC;
    for (auto const& configItem : config) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void SplitConcatMultiInputsTest::SetUp() {
    std::vector<size_t> inputShape;
    size_t splitsNum;
    std::map<std::string, std::string> tempConfig;
    InferenceEngine::Precision netPrecision;
    bool withFC;
    std::tie(netPrecision, targetDevice, tempConfig, inputShape, splitsNum, withFC) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    inputShape[1] *= splitsNum;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto split = ngraph::builder::makeSplit(params[0], ngPrc, splitsNum, 1);
    ngraph::OutputVector concatInputs = split->outputs();

    auto concat = std::make_shared<ngraph::opset7::Concat>(concatInputs, 1);

    if (withFC) {
        auto mul_const = ngraph::builder::makeConstant<float>(ngPrc, { 10, inputShape[1] },
            CommonTestUtils::generate_float_numbers(10 * inputShape[1], -0.2f, 0.2f), false);
        auto matmul = std::make_shared<ngraph::op::MatMul>(concat, mul_const, false, true);
        function = std::make_shared<ngraph::Function>(matmul, params, "SplitConcatMultiInputs");
    } else {
        function = std::make_shared<ngraph::Function>(concat, params, "SplitConcatMultiInputs");
    }
}

InferenceEngine::Blob::Ptr SplitConcatMultiInputsTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), inputDataMax - inputDataMin, inputDataMin, 1 / inputDataResolution,
                                            seed);
}
}  // namespace SubgraphTestsDefinitions
