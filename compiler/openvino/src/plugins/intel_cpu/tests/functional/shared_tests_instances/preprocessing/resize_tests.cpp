// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preprocessing/resize_tests.hpp"

#include <gtest/gtest.h>

using namespace ov::preprocess;

INSTANTIATE_TEST_SUITE_P(smoke_Preprocessing,
                         PreprocessingResizeTests,
                         testing::Values(CommonTestUtils::DEVICE_CPU),
                         PreprocessingResizeTests::getTestCaseName);
