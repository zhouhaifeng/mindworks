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
using ngraph::test::ValueMap;

TEST(attributes, split_op) {
    NodeBuilder::get_ops().register_factory<opset1::Split>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto axis = make_shared<op::Parameter>(element::i32, Shape{});
    auto num_splits = 2;
    auto split = make_shared<opset1::Split>(data, axis, num_splits);
    NodeBuilder builder(split, {data, axis});
    auto g_split = ov::as_type_ptr<opset1::Split>(builder.create());

    EXPECT_EQ(g_split->get_num_splits(), split->get_num_splits());
}

TEST(attributes, split_op2) {
    NodeBuilder::get_ops().register_factory<opset1::Split>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{300});
    auto axis = make_shared<op::Parameter>(element::i32, Shape{});
    auto num_splits = 3;
    auto split = make_shared<opset1::Split>(data, axis, num_splits);
    NodeBuilder builder(split, {data, axis});
    auto g_split = ov::as_type_ptr<opset1::Split>(builder.create());

    EXPECT_EQ(g_split->get_num_splits(), split->get_num_splits());
}
