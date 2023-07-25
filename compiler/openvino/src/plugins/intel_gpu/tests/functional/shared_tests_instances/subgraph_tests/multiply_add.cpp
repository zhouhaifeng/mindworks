// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/multiply_add.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::vector<size_t>> inputShapes = {
        {1, 3},
        {1, 3, 2},
        {1, 3, 2, 5},
        {1, 3, 2, 5, 4},
        {1, 3, 2, 2, 4, 5},
};

INSTANTIATE_TEST_SUITE_P(smoke_MultipleAdd_Nd, MultiplyAddLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        MultiplyAddLayerTest::getTestCaseName);

}  // namespace
