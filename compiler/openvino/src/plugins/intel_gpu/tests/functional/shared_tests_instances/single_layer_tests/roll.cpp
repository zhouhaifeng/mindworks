// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/roll.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecision = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP64,
    InferenceEngine::Precision::U8,
    InferenceEngine::Precision::I8,
    InferenceEngine::Precision::I16,
    InferenceEngine::Precision::U16,
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::U32,
    InferenceEngine::Precision::I64,
    InferenceEngine::Precision::U64,
    InferenceEngine::Precision::BOOL,
};

INSTANTIATE_TEST_SUITE_P(smoke_Roll_1d,
                         RollLayerTest,
                         testing::Combine(testing::Values(std::vector<size_t>{16}),  // Input shape
                                          testing::ValuesIn(inputPrecision),         // Precision
                                          testing::Values(std::vector<int64_t>{5}),  // Shift
                                          testing::Values(std::vector<int64_t>{0}),  // Axes
                                          testing::Values(CommonTestUtils::DEVICE_GPU)),
                         RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Roll_2d,
                         RollLayerTest,
                         testing::Combine(testing::Values(std::vector<size_t>{600, 450}),   // Input shape
                                          testing::ValuesIn(inputPrecision),                // Precision
                                          testing::Values(std::vector<int64_t>{300, 250}),  // Shift
                                          testing::Values(std::vector<int64_t>{0, 1}),      // Axes
                                          testing::Values(CommonTestUtils::DEVICE_GPU)),
                         RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Roll_2d_zero_shifts,
                         RollLayerTest,
                         testing::Combine(testing::Values(std::vector<size_t>{17, 19}),  // Input shape
                                          testing::ValuesIn(inputPrecision),             // Precision
                                          testing::Values(std::vector<int64_t>{0, 0}),   // Shift
                                          testing::Values(std::vector<int64_t>{0, 1}),   // Axes
                                          testing::Values(CommonTestUtils::DEVICE_GPU)),
                         RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Roll_3d,
                         RollLayerTest,
                         testing::Combine(testing::Values(std::vector<size_t>{2, 320, 320}),  // Input shape
                                          testing::ValuesIn(inputPrecision),                  // Precision
                                          testing::Values(std::vector<int64_t>{160, 160}),    // Shift
                                          testing::Values(std::vector<int64_t>{1, 2}),        // Axes
                                          testing::Values(CommonTestUtils::DEVICE_GPU)),
                         RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Roll_4d_negative_unordered_axes,
                         RollLayerTest,
                         testing::Combine(testing::Values(std::vector<size_t>{3, 11, 6, 4}),  // Input shape
                                          testing::ValuesIn(inputPrecision),                  // Precision
                                          testing::Values(std::vector<int64_t>{7, 3}),        // Shift
                                          testing::Values(std::vector<int64_t>{-3, -2}),      // Axes
                                          testing::Values(CommonTestUtils::DEVICE_GPU)),
                         RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
    smoke_Roll_5d_repeating_axes,
    RollLayerTest,
    testing::Combine(testing::Values(std::vector<size_t>{2, 16, 32, 7, 32}),                    // Input shape
                     testing::ValuesIn(inputPrecision),                                         // Precision
                     testing::Values(std::vector<int64_t>{16, 15, 10, 2, 1, 7, 2, 8, 1, 1}),    // Shift
                     testing::Values(std::vector<int64_t>{-1, -2, -3, 1, 0, 3, 3, 2, -2, -3}),  // Axes
                     testing::Values(CommonTestUtils::DEVICE_GPU)),
    RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Roll_6d_negative_shifts,
                         RollLayerTest,
                         testing::Combine(testing::Values(std::vector<size_t>{4, 16, 3, 6, 5, 2}),  // Input shape
                                          testing::ValuesIn(inputPrecision),                        // Precision
                                          testing::Values(std::vector<int64_t>{-2, -15, -2, -1, -4, -1}),  // Shift
                                          testing::Values(std::vector<int64_t>{0, 1, 2, 3, 4, 5}),         // Axes
                                          testing::Values(CommonTestUtils::DEVICE_GPU)),
                         RollLayerTest::getTestCaseName);

}  // namespace
