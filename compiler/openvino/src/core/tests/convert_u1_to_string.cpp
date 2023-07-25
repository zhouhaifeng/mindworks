// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_tools.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace ngraph;
using namespace std;

TEST(convert_u1_to_string, convert_u1_to_string) {
    vector<uint8_t> values{171, 16};
    auto constant = make_shared<op::Constant>(element::u1, Shape{12}, &values[0]);

    vector<string> ref{"1", "0", "1", "0", "1", "0", "1", "1", "0", "0", "0", "1"};
    for (size_t i = 0; i < 12; ++i) {
        ASSERT_EQ(constant->convert_value_to_string(i), ref[i]);
    }
}
