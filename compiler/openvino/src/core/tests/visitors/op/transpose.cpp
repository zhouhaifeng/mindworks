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

TEST(attributes, transpose_op) {
    using namespace opset1;

    NodeBuilder::get_ops().register_factory<Transpose>();
    const auto data_input = make_shared<Parameter>(element::f32, Shape{1, 2, 3});
    const auto axes_order_input = make_shared<Parameter>(element::i32, Shape{3});

    const auto op = make_shared<Transpose>(data_input, axes_order_input);

    NodeBuilder builder(op, {data_input, axes_order_input});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<Transpose>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
