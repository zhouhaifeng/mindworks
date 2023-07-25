// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/callback.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::map<std::string, std::string>> configs = {
    {},
};

const std::vector<std::map<std::string, std::string>> multiConfigs = {
    {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_GNA}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestCallbackTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         InferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         InferRequestCallbackTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         InferRequestCallbackTests::getTestCaseName);
}  // namespace
