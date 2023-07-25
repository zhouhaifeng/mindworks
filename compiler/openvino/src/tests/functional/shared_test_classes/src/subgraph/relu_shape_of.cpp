// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/relu_shape_of.hpp"

namespace SubgraphTestsDefinitions {

    std::string ReluShapeOfSubgraphTest::getTestCaseName(const testing::TestParamInfo<LayerTestsDefinitions::shapeOfParamsCommon>& obj) {
        InferenceEngine::SizeVector inputShapes;
        InferenceEngine::Precision inputPrecision;
        std::string targetDevice;
        std::tie(inputPrecision, inputShapes, targetDevice) = obj.param;
        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        result << "Precision=" << inputPrecision.name() << "_";
        result << "TargetDevice=" << targetDevice;
        return result.str();
    }

    void ReluShapeOfSubgraphTest::SetUp() {
        InferenceEngine::SizeVector inputShapes;
        InferenceEngine::Precision inputPrecision;
        std::tie(inputPrecision, inputShapes, targetDevice) = this->GetParam();
        auto inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
        auto param = ngraph::builder::makeParams(inType, {inputShapes});
        auto relu = std::make_shared<ngraph::opset3::Relu>(param[0]);
        auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(relu, inType);
        const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(shapeOf)};
        function = std::make_shared<ngraph::Function>(results, param, "ReluShapeOf");
    }
}  // namespace SubgraphTestsDefinitions
