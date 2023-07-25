// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/exec_network_base.hpp"

using namespace ov::test::behavior;
namespace {
auto autoBatchConfigs = []() {
    return std::vector<ov::AnyMap>{
        // explicit batch size 4 to avoid fallback to no auto-batching
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(CommonTestUtils::DEVICE_TEMPLATE) + "(4)"},
         // no timeout to avoid increasing the test time
         {ov::auto_batch_timeout.name(), "0"}}};
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatchBehaviorTests, OVExecutableNetworkBaseTest,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                 ::testing::ValuesIn(autoBatchConfigs())),
                         OVExecutableNetworkBaseTest::getTestCaseName);

} // namespace
