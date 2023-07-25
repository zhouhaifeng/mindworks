// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fake_quantize_and_two_output_branches_with_convolution.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::pass::low_precision;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<FakeQuantizeAndTwoOutputBranchesWithConvolution> testValues = {
    {
        { 256ul, {}, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::PartialShape({ 1, 32, 72, 48 })),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(testValues)),
    FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation::getTestCaseName);
}  // namespace
