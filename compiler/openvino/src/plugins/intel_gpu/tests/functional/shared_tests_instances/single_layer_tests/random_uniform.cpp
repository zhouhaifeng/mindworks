// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ie_precision.hpp>
#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/random_uniform.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<RandomUniformTypeSpecificParams> random_uniform_type_specific_params = {
        {InferenceEngine::Precision::I32, -100, 100},
        {InferenceEngine::Precision::FP32, 0.0f, 1.0f},
        {InferenceEngine::Precision::FP16, -10.0, 10.0}
};


const std::vector<int64_t> global_seeds = {10, 100, 500};
const std::vector<int64_t> op_seeds = {10, 50};

const std::vector<ov::Shape> output_shapes = {
        {1, 3, 3,  3},
        {1, 1, 5,  5},
        {2, 1, 10, 10}
};

INSTANTIATE_TEST_SUITE_P(
        smoke_BasicRandomUniform, RandomUniformLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(output_shapes),
                ::testing::ValuesIn(random_uniform_type_specific_params),
                ::testing::ValuesIn(global_seeds),
                ::testing::ValuesIn(op_seeds),
                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
        RandomUniformLayerTest::getTestCaseName);

}  // namespace
