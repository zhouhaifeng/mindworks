// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/single_layer/batch_to_space.hpp"

namespace LayerTestsDefinitions {

std::string BatchToSpaceLayerTest::getTestCaseName(const testing::TestParamInfo<batchToSpaceParamsTuple> &obj) {
    std::vector<size_t> inShapes;
    std::vector<int64_t> blockShape, cropsBegin, cropsEnd;
    InferenceEngine::Precision  netPrc;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::tie(blockShape, cropsBegin, cropsEnd, inShapes, netPrc, inPrc, outPrc, inLayout, outLayout, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inShapes) << "_";
    result << "netPRC=" << netPrc.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "BS=" << CommonTestUtils::vec2str(blockShape) << "_";
    result << "CB=" << CommonTestUtils::vec2str(cropsBegin) << "_";
    result << "CE=" << CommonTestUtils::vec2str(cropsEnd) << "_";
    result << "trgDev=" << targetName << "_";
    return result.str();
}

void BatchToSpaceLayerTest::SetUp() {
    std::vector<size_t> inputShape;
    std::vector<int64_t> blockShape, cropsBegin, cropsEnd;
    InferenceEngine::Precision netPrecision;
    std::tie(blockShape, cropsBegin, cropsEnd, inputShape, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto b2s = ngraph::builder::makeBatchToSpace(paramOuts[0], ngPrc, blockShape, cropsBegin, cropsEnd);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(b2s)};
    function = std::make_shared<ngraph::Function>(results, params, "BatchToSpace");
}

}  // namespace LayerTestsDefinitions
