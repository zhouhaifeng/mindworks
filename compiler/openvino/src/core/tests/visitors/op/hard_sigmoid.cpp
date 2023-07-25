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

TEST(attributes, hardsigmoid_op) {
    NodeBuilder::get_ops().register_factory<opset1::HardSigmoid>();
    const auto data = make_shared<op::Parameter>(element::f32, Shape{2, 5});
    const auto alpha = make_shared<op::Parameter>(element::f32, Shape{});
    const auto beta = make_shared<op::Parameter>(element::f32, Shape{});

    const auto hardsigmoid = make_shared<opset1::HardSigmoid>(data, alpha, beta);
    NodeBuilder builder(hardsigmoid, {data, alpha, beta});
    EXPECT_NO_THROW(auto g_hardsigmoid = ov::as_type_ptr<opset1::HardSigmoid>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
