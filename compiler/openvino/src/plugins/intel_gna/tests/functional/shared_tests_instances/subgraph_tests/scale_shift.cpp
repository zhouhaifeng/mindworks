// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "subgraph_tests/scaleshift.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

std::vector<std::vector<std::vector<size_t>>> inShapes =
    {{{1, 8}}, {{2, 16}}, {{3, 32}}, {{4, 64}}, {{5, 128}}, {{6, 256}}, {{7, 512}}, {{8, 1024}}};

std::vector<std::vector<float>> Scales = {{2.0f}, {3.0f}, {-1.0f}, {-2.0f}, {-3.0f}};

std::vector<std::vector<float>> Shifts = {{1.0f}, {2.0f}, {3.0f}, {-1.0f}, {-2.0f}, {-3.0f}};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_SUITE_P(smoke_scale_shift,
                         ScaleShiftLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(Scales),
                                            ::testing::ValuesIn(Shifts)),
                         ScaleShiftLayerTest::getTestCaseName);
}  // namespace
