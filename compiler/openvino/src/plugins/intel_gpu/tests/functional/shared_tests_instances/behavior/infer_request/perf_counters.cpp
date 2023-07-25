// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/perf_counters.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
auto configs = []() {
    return std::vector<std::map<std::string, std::string>>{{}};
};

auto Multiconfigs =
    []() {
        return std::vector<std::map<std::string, std::string>>{
            {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_GPU}}};
    };

auto AutoConfigs =
    []() {
        return std::vector<std::map<std::string, std::string>>{
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_GPU},
             {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU}}};
    };

auto AutoBatchConfigs =
    []() {
        return std::vector<std::map<std::string, std::string>>{
            // explicit batch size 4 to avoid fallback to no auto-batching (i.e. plain GPU)
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(CommonTestUtils::DEVICE_GPU) + "(4)"},
             // no timeout to avoid increasing the test time
             {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "0 "}}};
    };

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_GPU),
                                            ::testing::ValuesIn(configs())),
                         InferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         InferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(Multiconfigs())),
                         InferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         InferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(AutoConfigs())),
                         InferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         InferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                            ::testing::ValuesIn(AutoBatchConfigs())),
                         InferRequestPerfCountersTest::getTestCaseName);
}  // namespace
