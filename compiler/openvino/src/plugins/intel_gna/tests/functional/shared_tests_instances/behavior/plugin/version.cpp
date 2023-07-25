// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/version.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         VersionTest,
                         ::testing::Values(CommonTestUtils::DEVICE_GNA),
                         VersionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         VersionTest,
                         ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                         VersionTest::getTestCaseName);

}  // namespace
