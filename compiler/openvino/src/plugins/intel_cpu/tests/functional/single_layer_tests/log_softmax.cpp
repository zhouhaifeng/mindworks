// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using logSoftmaxLayerTestParams = std::tuple<
        std::vector<InputShape>,               // inputShape
        Precision,                             // netPrecision
        int64_t>;                              // axis

class LogSoftmaxLayerCPUTest
        : public testing::WithParamInterface<logSoftmaxLayerTestParams>,
          public SubgraphBaseTest,
          public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<logSoftmaxLayerTestParams> obj) {
        std::vector<InputShape> inputShapes;
        Precision netPrecision;
        int64_t axis;
        std::tie(inputShapes, netPrecision, axis) = obj.param;

        std::ostringstream result;
        if (inputShapes.front().first.size() != 0) {
            result << "IS=(";
            for (const auto &shape : inputShapes) {
                result << CommonTestUtils::partialShape2str({shape.first}) << "_";
            }
            result.seekp(-1, result.cur);
            result << ")_";
        }
        result << "TS=";
        for (const auto &shape : inputShapes) {
            for (const auto &item : shape.second) {
                result << CommonTestUtils::vec2str(item) << "_";
            }
        }
        result << "netPRC=" << netPrecision.name();
        result << "Axis=" << axis;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        std::vector<InputShape> inputShapes;
        Precision netPrecision;
        int64_t axis;
        std::tie(inputShapes, netPrecision, axis) = this->GetParam();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        inType = outType = ngPrc;

        selectedType = std::string("unknown_") + netPrecision.name();
        init_input_shapes(inputShapes);

        const auto params = ngraph::builder::makeDynamicParams(ngPrc, {inputDynamicShapes.front()});
        const auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        const auto logSoftmax = std::make_shared<ngraph::op::v5::LogSoftmax>(paramOuts[0], axis);
        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(logSoftmax)};
        function = std::make_shared<ngraph::Function>(results, params, "logSoftmax");
    }
};

TEST_P(LogSoftmaxLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "logSoftmax");
}

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        Precision::FP32
};

const std::vector<std::vector<InputShape>> inputShapes2D = {
        {
                {{{-1, -1}, {{1, 100}, {100, 1}, {10, 10}}}},
                {{{-1, {1}}, {{1, 1}, {100, 1}, {10, 1}}}}
        }
};

const std::vector<int64_t> axis2D = {
        -2, -1, 0, 1
};

const auto params2D = testing::Combine(
        testing::ValuesIn(inputShapes2D),
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(axis2D));

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax2D_dynamic, LogSoftmaxLayerCPUTest, params2D,
                         LogSoftmaxLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes4D = {
        {
                {{{-1, -1, -1, -1}, {{1, 100, 1, 1}, {1, 3, 4, 3}, {2, 3, 4, 5}}}},
                {{{{1, 2}, -1, {1, 5}, -1}, {{1, 100, 1, 1}, {1, 3, 5, 3}, {2, 3, 4, 5}}}}
        }
};

const std::vector<int64_t> axis4D = {
        -4, -3, -2, -1, 0, 1, 2, 3
};

const auto params4D = testing::Combine(
        testing::ValuesIn(inputShapes4D),
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(axis4D));

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax4D_dynamic, LogSoftmaxLayerCPUTest, params4D,
                         LogSoftmaxLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
