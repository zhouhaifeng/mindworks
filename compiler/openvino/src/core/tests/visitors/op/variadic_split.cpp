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

TEST(attributes, variadic_split_op) {
    using namespace opset1;

    NodeBuilder::get_ops().register_factory<op::v1::VariadicSplit>();
    auto data = make_shared<Parameter>(element::i32, Shape{200});
    auto axis = make_shared<Parameter>(element::i32, Shape{1});
    auto split_lengths = make_shared<Parameter>(element::i32, Shape{1});

    auto split = make_shared<VariadicSplit>(data, axis, split_lengths);
    NodeBuilder builder(split, {data, axis, split_lengths});
    EXPECT_NO_THROW(auto g_split = ov::as_type_ptr<VariadicSplit>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
