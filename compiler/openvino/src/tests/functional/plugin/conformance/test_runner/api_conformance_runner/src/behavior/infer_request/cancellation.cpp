// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/cancellation.hpp"
#include "api_conformance_helpers.hpp"

namespace {
using namespace BehaviorTestsDefinitions;
using namespace ov::test::conformance;

INSTANTIATE_TEST_SUITE_P(ie_infer_request, InferRequestCancellationTests,
                         ::testing::Combine(
                                 ::testing::ValuesIn(return_all_possible_device_combination()),
                                 ::testing::Values(ie_config)),
                         InferRequestCancellationTests::getTestCaseName);
}  // namespace
