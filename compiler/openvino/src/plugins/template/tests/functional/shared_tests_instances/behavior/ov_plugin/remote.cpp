// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/remote.hpp"

#include "common_test_utils/test_constants.hpp"
#include "ie_plugin_config.hpp"

using namespace ov::test;

namespace {
auto template_config = []() {
    return std::vector<ov::AnyMap>{{}};
};

std::vector<std::pair<ov::AnyMap, ov::AnyMap>> generate_remote_params() {
    return std::vector<std::pair<ov::AnyMap, ov::AnyMap>>{{{}, {}}};
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVRemoteTest,
                         ::testing::Combine(::testing::Values(ngraph::element::f32),
                                            ::testing::Values(::CommonTestUtils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(template_config()),
                                            ::testing::ValuesIn(generate_remote_params())),
                         OVRemoteTest::getTestCaseName);
}  // namespace
