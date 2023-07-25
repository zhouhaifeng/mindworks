// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/caching_tests.hpp"

using namespace ov::test::behavior;

namespace {
static const std::vector<ov::element::Type> precisionsTemplate = {
    ov::element::f32,
};

static const std::vector<std::size_t> batchSizesTemplate = {1, 2};

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_CachingSupportCase_Template,
                         CompileModelCacheTestBase,
                         ::testing::Combine(::testing::ValuesIn(CompileModelCacheTestBase::getStandardFunctions()),
                                            ::testing::ValuesIn(precisionsTemplate),
                                            ::testing::ValuesIn(batchSizesTemplate),
                                            ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                            ::testing::Values(ov::AnyMap{})),
                         CompileModelCacheTestBase::getTestCaseName);

const std::vector<ov::AnyMap> TemplateConfigs = {
    {ov::num_streams(2)},
};
const std::vector<std::string> TestTemplateTargets = {
    CommonTestUtils::DEVICE_TEMPLATE,
};
INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_Template,
                         CompileModelLoadFromMemoryTestBase,
                         ::testing::Combine(::testing::ValuesIn(TestTemplateTargets),
                                            ::testing::ValuesIn(TemplateConfigs)),
                         CompileModelLoadFromMemoryTestBase::getTestCaseName);
}  // namespace
