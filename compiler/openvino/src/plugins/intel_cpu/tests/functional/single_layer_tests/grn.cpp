// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using GRNCPUTestParams = typename std::tuple<
        ov::element::Type,             // Network precision
        InferenceEngine::Precision,    // Input precision
        InferenceEngine::Precision,    // Output precision
        InferenceEngine::Layout,       // Input layout
        InferenceEngine::Layout,       // Output layout
        InputShape,                    // Input shape
        float,                         // Bias
        std::string>;                  // Device name

class GRNLayerCPUTest : public testing::WithParamInterface<GRNCPUTestParams>,
                        virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GRNCPUTestParams> obj) {
        ov::element::Type netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        InputShape inputShape;
        float bias;
        std::string targetDevice;

        std::tie(netPrecision, inPrc, outPrc,
                 inLayout, outLayout,
                 inputShape,
                 bias,
                 targetDevice) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::partialShape2str({inputShape.first}) << "_";
        result << "TS=";
        for (const auto& item : inputShape.second) {
            result << CommonTestUtils::vec2str(item) << "_";
        }
        result << "netPRC=" << netPrecision.get_type_name() << "_";
        result << "inPRC=" << inPrc.name() << "_";
        result << "outPRC=" << outPrc.name() << "_";
        result << "inL=" << inLayout << "_";
        result << "outL=" << outLayout << "_";
        result << "bias="   << bias << "_";
        result << "trgDev=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        ov::element::Type netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        InputShape inputShape;
        float bias;

        std::tie(netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, bias, targetDevice) = GetParam();

        init_input_shapes({inputShape});

        const auto paramsIn = ngraph::builder::makeDynamicParams(netPrecision, inputDynamicShapes);

        const auto paramsOut = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
        const auto grn = std::make_shared<ngraph::opset1::GRN>(paramsOut[0], bias);
        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(grn)};
        function = std::make_shared<ngraph::Function>(results, paramsIn, "Grn");
    }
};

TEST_P(GRNLayerCPUTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::bf16,
        ov::element::f16,
        ov::element::f32
};

const std::vector<float> biases = {1e-6f, 0.33f, 1.1f, 2.25f, 100.25f};

const std::vector<InputShape> dataInputStaticShapes = {{{}, {{16, 24}}}, {{}, {{3, 16, 24}}}, {{}, {{1, 3, 30, 30}}}};

const std::vector<InputShape> dataInputDynamicShapes =
    {{{-1, -1}, {{5, 17}, {10, 3}}}, {{3, {10, 12}, -1}, {{3, 12, 25}, {3, 10, 10}}},
     {{2, -1, -1, {5, 10}}, {{2, 17, 20, 7}, {2, 10, 12, 5}}}};

INSTANTIATE_TEST_SUITE_P(smoke_GRNCPUStatic, GRNLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::ValuesIn(dataInputStaticShapes),
                            ::testing::ValuesIn(biases),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GRNLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GRNCPUDynamic, GRNLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::ValuesIn(dataInputDynamicShapes),
                            ::testing::ValuesIn(biases),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GRNLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
