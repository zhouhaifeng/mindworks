// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_layer/convolution.hpp"
#include "common_test_utils/test_constants.hpp"

// using namespace LayerTestsDefinitions;
using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

using LayerTestsDefinitions::convSpecificParams;

typedef std::tuple<
        convSpecificParams,
        ElementType,     // Net precision
        ElementType,     // Input precision
        ElementType,     // Output precision
        InputShape,      // Input shape
        LayerTestsUtils::TargetDevice   // Device name
> convLayerTestParamsSet;


class ConvolutionLayerGPUTestDynamic : public testing::WithParamInterface<convLayerTestParamsSet>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convLayerTestParamsSet>& obj) {
        convSpecificParams convParams;
        ElementType netType;
        ElementType inType, outType;
        InputShape inputShape;
        std::string targetDevice;
        std::tie(convParams, netType, inType, outType, inputShape, targetDevice) = obj.param;

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        std::ostringstream result;
        result << "IS=";
        result  << CommonTestUtils::partialShape2str({inputShape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : inputShape.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        result << ")_";
        result << "K" << CommonTestUtils::vec2str(kernel) << "_";
        result << "S" << CommonTestUtils::vec2str(stride) << "_";
        result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
        result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
        result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
        result << "O=" << convOutChannels << "_";
        result << "AP=" << padType << "_";
        result << "netPRC=" << netType << "_";
        result << "inPRC=" << inType << "_";
        result << "outPRC=" << outType << "_";
        result << "trgDev=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        convSpecificParams convParams;
        InputShape inputShape;
        auto netType = ElementType::undefined;
        std::tie(convParams, netType, inType, outType, inputShape, targetDevice) = this->GetParam();

        init_input_shapes({inputShape});

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        auto inputParams = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(inputParams));

        auto convolutionNode = ngraph::builder::makeConvolution(paramOuts.front(), netType, kernel, stride, padBegin,
                                                                padEnd, dilation, padType, convOutChannels);

        ngraph::ResultVector results;
        for (size_t i = 0; i < convolutionNode->get_output_size(); i++)
            results.push_back(std::make_shared<ngraph::opset1::Result>(convolutionNode->output(i)));

        function = std::make_shared<ngraph::Function>(results, inputParams, "Convolution");
    }
};

TEST_P(ConvolutionLayerGPUTestDynamic, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {

// ======== 1D convolutions
const std::vector<ov::test::InputShape> dynInputShapes1D = {
    {
        {1, 10, ov::Dimension::dynamic()},
        {{1, 10, 20}, {1, 10, 30}, {1, 10, 50}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic1DSymPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3}),
                        ::testing::Values(SizeVector{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1}),
                        ::testing::Values(SizeVector{1}),
                        ::testing::Values(10),
                        ::testing::ValuesIn({ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes1D),
                ::testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ======== 2D convolutions
const std::vector<ov::test::InputShape> dynInputShapes2D = {
    {
        {1, 10, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        {{1, 10, 20, 20}, {1, 10, 30, 30}, {1, 10, 40, 20}}
    },
};
// ==== Symmetric pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic2DSymPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(10),
                        ::testing::ValuesIn({ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ==== Symmetric auto pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic2DSymAutoPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{0, 0}),
                        ::testing::Values(std::vector<ptrdiff_t>{0, 0}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(10),
                        ::testing::ValuesIn({ngraph::op::PadType::SAME_LOWER, ngraph::op::PadType::SAME_UPPER})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ==== Asymmetric pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic2D_AsymPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{2, 1}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(10),
                        ::testing::ValuesIn({ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ======== 3D convolutions
const std::vector<ov::test::InputShape> dynInputShapes3D = {
    {
        {1, 3, ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        {{1, 3, 10, 10, 10}, {1, 3, 20, 20, 10}, {1, 3, 15, 15, 10}}
    },
};

// ==== Symmetric pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic3DSymPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3, 3}),
                        ::testing::Values(SizeVector{1, 1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2, 1}),
                        ::testing::Values(SizeVector{1, 1, 1}),
                        ::testing::Values(3),
                        ::testing::ValuesIn({ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes3D),
                ::testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ==== Symmetric auto pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic3DSymAutoPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3, 3}),
                        ::testing::Values(SizeVector{1, 1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{0, 0, 0}),
                        ::testing::Values(std::vector<ptrdiff_t>{0, 0, 0}),
                        ::testing::Values(SizeVector{1, 1, 1}),
                        ::testing::Values(3),
                        ::testing::ValuesIn({ngraph::op::PadType::SAME_LOWER, ngraph::op::PadType::SAME_UPPER})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes3D),
                ::testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ==== Asymmetric pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic3DAsymPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3, 3}),
                        ::testing::Values(SizeVector{1, 1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{2, 1, 1}),
                        ::testing::Values(SizeVector{1, 1, 1}),
                        ::testing::Values(3),
                        ::testing::ValuesIn({ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes3D),
                ::testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

}  // namespace
} // namespace GPULayerTestsDefinitions
