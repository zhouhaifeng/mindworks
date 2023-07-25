// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/set_blob_by_type.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine;

const std::vector<FuncTestUtils::BlobType> BlobTypes = {
    FuncTestUtils::BlobType::Compound,
    FuncTestUtils::BlobType::Batched,
    FuncTestUtils::BlobType::Memory,
};

auto gpuConfig = []() {
    return std::map<std::string, std::string>{};
};  // nothing special
auto multiConfig = []() {
    return std::map<std::string, std::string>{{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_GPU}};
};
auto heteroConfig = []() {
    return std::map<std::string, std::string>{{"TARGET_FALLBACK", CommonTestUtils::DEVICE_GPU}};
};

INSTANTIATE_TEST_SUITE_P(smoke_Behavior, InferRequestSetBlobByType,
    ::testing::Combine(::testing::ValuesIn(BlobTypes),
                       ::testing::Values(CommonTestUtils::DEVICE_GPU),
                       ::testing::Values(gpuConfig())),
    InferRequestSetBlobByType::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Multi, InferRequestSetBlobByType,
    ::testing::Combine(::testing::ValuesIn(BlobTypes),
                       ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                       ::testing::Values(multiConfig())),
    InferRequestSetBlobByType::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Auto, InferRequestSetBlobByType,
    ::testing::Combine(::testing::ValuesIn(BlobTypes),
                       ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                       ::testing::Values(multiConfig())),
    InferRequestSetBlobByType::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Hetero, InferRequestSetBlobByType,
    ::testing::Combine(::testing::ValuesIn(BlobTypes),
                       ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                       ::testing::Values(heteroConfig())),
    InferRequestSetBlobByType::getTestCaseName);
