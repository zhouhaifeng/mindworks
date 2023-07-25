// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/single_layer/space_to_depth.hpp"

namespace LayerTestsDefinitions {

using namespace ngraph::opset3;

static inline std::string SpaceToDepthModeToString(const SpaceToDepth::SpaceToDepthMode& mode) {
    static std::map<SpaceToDepth::SpaceToDepthMode, std::string> names = {
        {SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, "BLOCKS_FIRST"},
        {SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, "DEPTH_FIRST"},
    };

    auto i = names.find(mode);
    if (i != names.end())
        return i->second;
    else
        throw std::runtime_error("Unsupported SpaceToDepthMode");
}

std::string SpaceToDepthLayerTest::getTestCaseName(const testing::TestParamInfo<spaceToDepthParamsTuple> &obj) {
    std::vector<size_t> inShape;
    SpaceToDepth::SpaceToDepthMode mode;
    std::size_t blockSize;
    InferenceEngine::Precision inputPrecision;
    std::string targetName;
    std::tie(inShape, inputPrecision, mode, blockSize, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inShape) << "_";
    result << "inPrc=" << inputPrecision.name() << "_";
    result << "M=" << SpaceToDepthModeToString(mode) << "_";
    result << "BS=" << blockSize << "_";
    result << "targetDevice=" << targetName << "_";
    return result.str();
}

void SpaceToDepthLayerTest::SetUp() {
    std::vector<size_t> inShape;
    SpaceToDepth::SpaceToDepthMode mode;
    std::size_t blockSize;
    InferenceEngine::Precision inputPrecision;
    std::tie(inShape, inputPrecision, mode, blockSize, targetDevice) = this->GetParam();
    auto inPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    auto params = ngraph::builder::makeParams(inPrc, {inShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto s2d = ngraph::builder::makeSpaceToDepth(paramOuts[0], mode, blockSize);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(s2d)};
    function = std::make_shared<ngraph::Function>(results, params, "SpaceToDepth");
}
}  // namespace LayerTestsDefinitions
