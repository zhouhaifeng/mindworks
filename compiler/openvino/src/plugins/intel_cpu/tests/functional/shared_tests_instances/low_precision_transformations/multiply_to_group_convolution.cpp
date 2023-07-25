// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/multiply_to_group_convolution_transformation.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<element::Type> precisions = {
    element::f32
};

namespace shape4d {
const std::vector<ngraph::PartialShape> inputShapes = {
    { 1ul, 3ul, 16ul, 16ul },
    { 4ul, 3ul, 16ul, 16ul }
};

const std::vector<MultiplyToGroupConvolutionTransformationParam> params = {
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        {{1.f, 2.f, 3.f}, element::f32, Shape{1, 3, 1, 1}},
        "output/GroupConvolution",
        "U8",
        true
    },
    // Multiply with scalar is not transformed to GroupConvolution
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        {{4.f}, element::f32, Shape{1, 1, 1, 1}},
        "output/GroupConvolution",
        "",
        true
    },
    // Multiply with scalar is not transformed to GroupConvolution
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        {{4.f}, element::f32, Shape{}},
        "output/GroupConvolution",
        "",
        true
    },
    // Zero point
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f } },
        {{1.f, 2.f, 3.f}, element::f32, Shape{1, 3, 1, 1}},
        "output/GroupConvolution",
        "U8",
        true
    },
    // Zero point
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -1.28f }, { 1.27f / 2.f }, { -1.28f }, { 1.27f / 2.f} },
        {{1.f, 2.f, 3.f}, element::f32, Shape{1, 3, 1, 1}},
        "output/GroupConvolution",
        "U8",
        true
    }
};

//Comment out the tests because of the transformation is disabled by another WR
/*
INSTANTIATE_TEST_SUITE_P(smoke_LPT, MultiplyToGroupConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(params)),
    MultiplyToGroupConvolutionTransformation::getTestCaseName);
*/
}  // namespace shape4d

namespace shape5d {
const std::vector<ngraph::PartialShape> inputShapes = {
    { 1ul, 3ul, 16ul, 16ul, 16ul },
    { 4ul, 3ul, 16ul, 16ul, 16ul }
};

const std::vector<MultiplyToGroupConvolutionTransformationParam> params = {
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        {{1.f, 2.f, 3.f}, element::f32, Shape{1, 3, 1, 1, 1}},
        "output/GroupConvolution",
        "U8"
    },
    // Multiply with scalar is not transformed to GroupConvolution
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        {{4.f}, element::f32, Shape{1, 1, 1, 1, 1}},
        "output/GroupConvolution",
        ""
    },
    // Multiply with scalar is not transformed to GroupConvolution
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        {{4.f}, element::f32, Shape{}},
        "output/GroupConvolution",
        ""
    },
    // Zero point
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f } },
        {{1.f, 2.f, 3.f}, element::f32, Shape{1, 3, 1, 1, 1}},
        "output/GroupConvolution",
        "U8"
    },
    // Zero point
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -1.28f }, { 1.27f / 2.f }, { -1.28f }, { 1.27f / 2.f} },
        {{1.f, 2.f, 3.f}, element::f32, Shape{1, 3, 1, 1, 1}},
        "output/GroupConvolution",
        "U8"
    }
};

//Comment out the tests because of the transformation is disabled by another WR
/*
INSTANTIATE_TEST_SUITE_P(smoke_LPT, MultiplyToGroupConvolutionTransformation,
     ::testing::Combine(
         ::testing::ValuesIn(precisions),
         ::testing::ValuesIn(inputShapes),
         ::testing::Values(CommonTestUtils::DEVICE_CPU),
         ::testing::ValuesIn(params)),
     MultiplyToGroupConvolutionTransformation::getTestCaseName);
*/
}  // namespace shape5d
}  // namespace
