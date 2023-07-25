// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/concat.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

std::vector<int> axes = {-3, -2, -1, 0, 1, 2, 3};
std::vector<std::vector<std::vector<size_t>>> inShapes = {
        {{10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}}
};
std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                         InferenceEngine::Precision::FP16,
                                                         InferenceEngine::Precision::I64};


INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, ConcatLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(axes),
                                ::testing::ValuesIn(inShapes),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConcatLayerTest::getTestCaseName);
}  // namespace
