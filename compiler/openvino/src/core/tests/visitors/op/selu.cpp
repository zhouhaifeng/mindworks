// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/visitor.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, selu_op) {
    NodeBuilder::get_ops().register_factory<opset1::Selu>();
    const auto data_input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto alpha = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto lambda = make_shared<op::Parameter>(element::f32, Shape{1});

    const auto op = make_shared<opset1::Selu>(data_input, alpha, lambda);

    NodeBuilder builder(op, {data_input, alpha, lambda});
    const auto expected_attr_count = 0;
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<opset1::Selu>(builder.create()));

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
