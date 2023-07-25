// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, prelu) {
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto slope = make_shared<op::Parameter>(element::f32, Shape{2});
    Shape prelu_shape{2, 4};
    auto prelu = make_shared<op::PRelu>(param, slope);
    ASSERT_EQ(prelu->get_element_type(), element::f32);
    ASSERT_EQ(prelu->get_shape(), prelu_shape);
}
