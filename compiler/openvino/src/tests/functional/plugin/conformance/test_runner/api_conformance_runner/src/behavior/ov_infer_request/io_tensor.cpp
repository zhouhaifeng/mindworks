// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/io_tensor.hpp"

#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {
INSTANTIATE_TEST_SUITE_P(ov_infer_request_mandatory, OVInferRequestIOTensorTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(return_all_possible_device_combination()),
                                ::testing::Values(pluginConfig)),
                        OVInferRequestIOTensorTest::getTestCaseName);

std::vector<ov::element::Type> ovIOTensorElemTypes = {
    ov::element::boolean,
    ov::element::bf16,
    ov::element::f16,
    ov::element::f32,
    ov::element::f64,
    ov::element::i8,
    ov::element::i16,
    ov::element::i32,
    ov::element::i64,
    ov::element::u8,
    ov::element::u16,
    ov::element::u32,
    ov::element::u64,
};

INSTANTIATE_TEST_SUITE_P(ov_infer_request_mandatory, OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(ovIOTensorElemTypes),
                                 ::testing::ValuesIn(return_all_possible_device_combination()),
                                 ::testing::Values(pluginConfig)),
                         OVInferRequestIOTensorSetPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_infer_request_mandatory, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(
                                 ::testing::ValuesIn(ovIOTensorElemTypes),
                                 ::testing::ValuesIn(return_all_possible_device_combination()),
                                 ::testing::Values(pluginConfig)),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);
}  // namespace
