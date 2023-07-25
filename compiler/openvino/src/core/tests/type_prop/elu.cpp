// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, elu) {
    Shape data_shape{2, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto elu = make_shared<op::Elu>(data, 1);
    ASSERT_EQ(elu->get_element_type(), element::f32);
    ASSERT_EQ(elu->get_shape(), data_shape);
}
