// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/mat_mul.hpp"

#include <vector>

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {InferenceEngine::Precision::FP32};

const std::vector<ShapeRelatedParams> shapeRelatedParams = {{{{5, 1}, true}, {{5, 1}, false}},
                                                            {{{1, 5}, false}, {{1, 5}, true}},
                                                            {{{5}, false}, {{5}, false}},
                                                            {{{5}, true}, {{5}, true}}};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {ngraph::helpers::InputLayerType::CONSTANT};

std::vector<std::map<std::string, std::string>> additional_config = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}},
                                                                     {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

INSTANTIATE_TEST_SUITE_P(smoke_MatMul,
                         MatMulTest,
                         ::testing::Combine(::testing::ValuesIn(shapeRelatedParams),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(secondaryInputTypes),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(additional_config)),
                         MatMulTest::getTestCaseName);

}  // namespace
