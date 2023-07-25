// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/version.hpp"

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<std::map<std::string, std::string>> configs = {
    {}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, VersionTest,
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                        VersionTest::getTestCaseName);

}  // namespace
