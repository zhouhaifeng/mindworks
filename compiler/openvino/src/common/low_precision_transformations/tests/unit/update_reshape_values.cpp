// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_blob.h>
#include <gtest/gtest.h>
#include "low_precision/network_helper.hpp"

using LPT_ReshapeTransformation = ::testing::Test;

TEST(LPT_UpdateReshapeValuesTransformation, updateReshapeValues_3_3_32_1_to_1_1_32_1) {
    ASSERT_EQ(
        ngraph::Shape({1, 1, 32, 1}),
        ngraph::pass::low_precision::NetworkHelper::updateReshapeValues({ 1, 32 }, { 9, 32 }, { 3, 3, 32, 1 }));
}
