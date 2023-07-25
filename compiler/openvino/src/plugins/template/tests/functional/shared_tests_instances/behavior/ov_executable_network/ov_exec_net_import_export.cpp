// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "behavior/compiled_model/import_export.hpp"

#include "ie_plugin_config.hpp"
#include <common_test_utils/test_constants.hpp>

using namespace ov::test::behavior;
namespace {
const std::vector<ov::element::Type_t> netPrecisions = {
        ov::element::i8,
        ov::element::i16,
        ov::element::i32,
        ov::element::i64,
        ov::element::u8,
        ov::element::u16,
        ov::element::u32,
        ov::element::u64,
        ov::element::f16,
        ov::element::f32,
};
const std::vector<ov::AnyMap> configs = {
        {},
};
const std::vector<ov::AnyMap> multiConfigs = {
        {ov::device::priorities(CommonTestUtils::DEVICE_TEMPLATE)}};

const std::vector<ov::AnyMap> heteroConfigs = {
        {ov::device::priorities(CommonTestUtils::DEVICE_TEMPLATE)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledGraphImportExportTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                 ::testing::ValuesIn(configs)),
                         OVCompiledGraphImportExportTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
         OVCompiledGraphImportExportTest,
        ::testing::Combine(
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                ::testing::ValuesIn(multiConfigs)),
        OVCompiledGraphImportExportTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
         OVCompiledGraphImportExportTest,
        ::testing::Combine(::testing::ValuesIn(netPrecisions),
                           ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                           ::testing::ValuesIn(heteroConfigs)),
        OVCompiledGraphImportExportTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OVClassImportExportTestP,
                         OVClassCompiledModelImportExportTestP,
                         ::testing::Values("HETERO:TEMPLATE"));   

}  // namespace
