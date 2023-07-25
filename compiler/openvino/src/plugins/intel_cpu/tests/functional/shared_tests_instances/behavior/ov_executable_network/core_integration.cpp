// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"

#include "openvino/runtime/core.hpp"

using namespace ov::test::behavior;

using namespace InferenceEngine::PluginConfigParams;

namespace {
//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassCompiledModelGetPropertyTest, OVClassCompiledModelGetPropertyTest,
        ::testing::Values("CPU", "MULTI:CPU", "HETERO:CPU", "AUTO:CPU"));

const std::vector<std::tuple<std::string, std::pair<ov::AnyMap, std::string>>> GetMetricTest_ExecutionDevice_CPU = {
        {"CPU", std::make_pair(ov::AnyMap{}, "CPU")}};

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassCompiledModelGetPropertyTest, OVClassCompiledModelGetPropertyTest_EXEC_DEVICES,
        ::testing::ValuesIn(GetMetricTest_ExecutionDevice_CPU));

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassCompiledModelGetIncorrectPropertyTest, OVClassCompiledModelGetIncorrectPropertyTest,
        ::testing::Values("CPU", "MULTI:CPU", "HETERO:CPU", "AUTO:CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassCompiledModelGetConfigTest, OVClassCompiledModelGetConfigTest,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassCompiledModelSetIncorrectConfigTest, OVClassCompiledModelSetIncorrectConfigTest,
        ::testing::Values("CPU"));

//////////////////////////////////////////////////////////////////////////////////////////

} // namespace

