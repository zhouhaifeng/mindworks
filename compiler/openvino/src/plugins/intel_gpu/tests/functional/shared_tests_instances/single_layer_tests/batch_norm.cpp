// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/batch_norm.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<double> epsilon = {
    1e-6,
    1e-5,
    1e-4
};
const std::vector<std::vector<size_t>> inputShapes = {
        {1, 3},
        {2, 5},
        {1, 3, 10},
        {1, 3, 1, 1},
        {2, 5, 4, 4},
};


const auto batchNormParams = testing::Combine(
        testing::ValuesIn(epsilon),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapes),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_BatchNorm,
        BatchNormLayerTest,
        batchNormParams,
        BatchNormLayerTest::getTestCaseName
);

}  // namespace
