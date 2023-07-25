// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/visitor.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"

using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, select_fp32) {
    NodeBuilder::get_ops().register_factory<opset1::Select>();
    auto in_cond = std::make_shared<op::Parameter>(element::boolean, Shape{3, 2});
    auto in_then = std::make_shared<op::Parameter>(element::f32, Shape{3, 2});
    auto in_else = std::make_shared<op::Parameter>(element::f32, Shape{3, 2});

    auto auto_broadcast = op::AutoBroadcastType::NUMPY;

    auto select = std::make_shared<opset1::Select>(in_cond, in_then, in_else, auto_broadcast);
    NodeBuilder builder(select, {in_cond, in_then, in_else});

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    auto g_select = ov::as_type_ptr<opset1::Select>(builder.create());
    EXPECT_EQ(g_select->get_autob(), select->get_autob());
}

TEST(attributes, select_i32) {
    NodeBuilder::get_ops().register_factory<opset1::Select>();
    auto in_cond = std::make_shared<op::Parameter>(element::boolean, Shape{3, 2});
    auto in_then = std::make_shared<op::Parameter>(element::i32, Shape{3, 2});
    auto in_else = std::make_shared<op::Parameter>(element::i32, Shape{3, 2});

    auto auto_broadcast = op::AutoBroadcastType::NUMPY;

    auto select = std::make_shared<opset1::Select>(in_cond, in_then, in_else, auto_broadcast);
    NodeBuilder builder(select, {in_cond, in_then, in_else});

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    auto g_select = ov::as_type_ptr<opset1::Select>(builder.create());
    EXPECT_EQ(g_select->get_autob(), select->get_autob());
}
