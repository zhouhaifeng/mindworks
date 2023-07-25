// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/visitor.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset7.hpp"
#include "ngraph/opsets/opset8.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, gather_v1_op) {
    NodeBuilder::get_ops().register_factory<opset1::Gather>();
    auto data = make_shared<opset1::Parameter>(element::i32, Shape{2, 3, 4});
    auto indices = make_shared<opset1::Parameter>(element::i32, Shape{2});
    auto axis = make_shared<opset1::Constant>(element::i32, Shape{}, 2);

    auto gather = make_shared<opset1::Gather>(data, indices, axis);
    NodeBuilder builder(gather, {data, indices, axis});
    auto g_gather = ov::as_type_ptr<opset1::Gather>(builder.create());

    EXPECT_EQ(g_gather->get_batch_dims(), gather->get_batch_dims());
}

TEST(attributes, gather_v7_op) {
    NodeBuilder::get_ops().register_factory<opset7::Gather>();
    auto data = make_shared<opset1::Parameter>(element::i32, Shape{2, 3, 4});
    auto indices = make_shared<opset1::Parameter>(element::i32, Shape{2});
    auto axis = make_shared<opset1::Constant>(element::i32, Shape{}, 2);
    int64_t batch_dims = 1;

    auto gather = make_shared<opset7::Gather>(data, indices, axis, batch_dims);
    NodeBuilder builder(gather, {data, indices, axis});
    auto g_gather = ov::as_type_ptr<opset7::Gather>(builder.create());

    EXPECT_EQ(g_gather->get_batch_dims(), gather->get_batch_dims());
}

TEST(attributes, gather_v8_op) {
    NodeBuilder::get_ops().register_factory<opset8::Gather>();
    auto data = make_shared<opset1::Parameter>(element::i32, Shape{2, 3, 4});
    auto indices = make_shared<opset1::Parameter>(element::i32, Shape{2});
    auto axis = make_shared<opset1::Constant>(element::i32, Shape{}, 2);
    int64_t batch_dims = 1;

    auto gather = make_shared<opset8::Gather>(data, indices, axis, batch_dims);
    NodeBuilder builder(gather, {data, indices, axis});
    auto g_gather = ov::as_type_ptr<opset8::Gather>(builder.create());

    EXPECT_EQ(g_gather->get_batch_dims(), gather->get_batch_dims());
}
