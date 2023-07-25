// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/split_conv_concat.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, SplitConvConcat,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t >({1, 6, 40, 40})),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        SplitConvConcat::getTestCaseName);

}  // namespace


