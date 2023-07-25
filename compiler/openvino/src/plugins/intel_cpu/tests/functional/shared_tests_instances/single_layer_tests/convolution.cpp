// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/convolution.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32};

/* ============= 1D Convolution ============= */
const std::vector<std::vector<size_t>> kernels1D = {{3}, {5}};
const std::vector<std::vector<size_t>> strides1D = {{1}, {3}};
const std::vector<std::vector<ptrdiff_t>> padBegins1D = {{0}, {3}};
const std::vector<std::vector<ptrdiff_t>> padEnds1D = {{0}, {3}};
const std::vector<std::vector<size_t>> dilations1D = {{1}, {3}};
const std::vector<size_t> numOutChannels1D = {1, 5};

const auto conv1DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels1D), ::testing::ValuesIn(strides1D),
        ::testing::ValuesIn(padBegins1D), ::testing::ValuesIn(padEnds1D),
        ::testing::ValuesIn(dilations1D), ::testing::ValuesIn(numOutChannels1D),
        ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv1DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels1D), ::testing::ValuesIn(strides1D),
        ::testing::Values(std::vector<ptrdiff_t>({0})),
        ::testing::Values(std::vector<ptrdiff_t>({0})),
        ::testing::ValuesIn(dilations1D), ::testing::ValuesIn(numOutChannels1D),
        ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_SUITE_P(
        smoke_Convolution1D_ExplicitPadding, ConvolutionLayerTest,
        ::testing::Combine(
                conv1DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 3, 30})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Convolution1D_AutoPadValid, ConvolutionLayerTest,
        ::testing::Combine(
                conv1DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 3, 30})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t>> kernels = {{3, 3}, {3, 5}};
const std::vector<std::vector<size_t>> strides = {{1, 1}, {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}, {0, 3}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}, {0, 3}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}, {3, 1}};
const std::vector<size_t> numOutChannels = {1, 5};

const auto conv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds),
        ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_SUITE_P(
        smoke_Convolution2D_ExplicitPadding, ConvolutionLayerTest,
        ::testing::Combine(
                conv2DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Convolution2D_AutoPadValid, ConvolutionLayerTest,
        ::testing::Combine(
                conv2DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ConvolutionLayerTest::getTestCaseName);

// weight for this convolution have Acdb16a layout
// for [96,1,7,7] shape strides for 1 and 3 dimensions equals, but not default order
namespace specificWeightLayout {
const std::vector<size_t> kernels = {7, 7};
const std::vector<size_t> strides = {2, 2};
const std::vector<ptrdiff_t> padBegins = {1, 1};
const std::vector<ptrdiff_t> padEnds = {1, 1};
const std::vector<size_t> dilations = {1, 1};
const size_t numOutChannels = {96};
const auto conv2DParams_WeightLayout = ::testing::Combine(::testing::Values(kernels),
                                                          ::testing::Values(strides),
                                                          ::testing::Values(padBegins),
                                                          ::testing::Values(padEnds),
                                                          ::testing::Values(dilations),
                                                          ::testing::Values(numOutChannels),
                                                          ::testing::Values(ngraph::op::PadType::EXPLICIT));

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_SpecificWeightLayout, ConvolutionLayerTest,
                         ::testing::Combine(conv2DParams_WeightLayout,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 1, 50, 75})),
                                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         ConvolutionLayerTest::getTestCaseName);
} // namespace specificWeightLayout

/* ============= 3D Convolution ============= */
const std::vector<std::vector<size_t>> kernels3d = {{3, 3, 3}, {3, 5, 3}};
const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0}, {0, 2, 0}};
const std::vector<std::vector<size_t>> strides3d = {{1, 1, 1}, {1, 2, 1}};
const std::vector<std::vector<size_t>> dilations3d = {{1, 1, 1}, {1, 2, 1}};
const std::vector<size_t> numOutChannels3D = {1, 5};

const auto conv3DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(paddings3d), ::testing::ValuesIn(paddings3d),
        ::testing::ValuesIn(dilations3d), ::testing::ValuesIn(numOutChannels3D),
        ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv3DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::ValuesIn(dilations3d), ::testing::ValuesIn(numOutChannels3D),
        ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_SUITE_P(
        smoke_Convolution3D_ExplicitPadding, ConvolutionLayerTest,
        ::testing::Combine(
                conv3DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Convolution3D_AutoPadValid, ConvolutionLayerTest,
        ::testing::Combine(
                conv3DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ConvolutionLayerTest::getTestCaseName);

}  // namespace
