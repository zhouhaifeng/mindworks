// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

#include "shared_test_classes/subgraph/strided_slice.hpp"

namespace SubgraphTestsDefinitions {

std::string StridedSliceTest::getTestCaseName(const testing::TestParamInfo<StridedSliceParams> &obj) {
    StridedSliceSpecificParams params;
    InferenceEngine::Precision netPrc;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::map<std::string, std::string> additionalConfig;
    std::tie(params, netPrc, inPrc, outPrc, inLayout, outLayout, targetName, additionalConfig) = obj.param;
    std::ostringstream result;
    result << "inShape=" << CommonTestUtils::vec2str(params.inputShape) << "_";
    result << "netPRC=" << netPrc.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "begin=" << CommonTestUtils::vec2str(params.begin) << "_";
    result << "end=" << CommonTestUtils::vec2str(params.end) << "_";
    result << "stride=" << CommonTestUtils::vec2str(params.strides) << "_";
    result << "begin_m=" << CommonTestUtils::vec2str(params.beginMask) << "_";
    result << "end_m=" << CommonTestUtils::vec2str(params.endMask) << "_";
    result << "new_axis_m=" << (params.newAxisMask.empty() ? "def" : CommonTestUtils::vec2str(params.newAxisMask)) << "_";
    result << "shrink_m=" << (params.shrinkAxisMask.empty() ? "def" : CommonTestUtils::vec2str(params.shrinkAxisMask)) << "_";
    result << "ellipsis_m=" << (params.ellipsisAxisMask.empty() ? "def" : CommonTestUtils::vec2str(params.ellipsisAxisMask)) << "_";
    result << "trgDev=" << targetName;
    for (auto const& configItem : additionalConfig) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void StridedSliceTest::SetUp() {
    StridedSliceSpecificParams ssParams;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additionalConfig;
    std::tie(ssParams, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice, additionalConfig) = this->GetParam();
    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {ssParams.inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto relu = std::make_shared<ngraph::opset1::Relu>(paramOuts[0]);
    auto ss = ngraph::builder::makeStridedSlice(relu, ssParams.begin, ssParams.end, ssParams.strides, ngPrc, ssParams.beginMask,
                                                ssParams.endMask, ssParams.newAxisMask, ssParams.shrinkAxisMask, ssParams.ellipsisAxisMask);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(ss)};
    function = std::make_shared<ngraph::Function>(results, params, "strided_slice");
}

}  // namespace SubgraphTestsDefinitions
