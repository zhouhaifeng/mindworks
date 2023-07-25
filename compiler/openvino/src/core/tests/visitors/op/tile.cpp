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

TEST(attributes, tile_op) {
    NodeBuilder::get_ops().register_factory<opset1::Tile>();
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto repeats = make_shared<op::Constant>(element::i64, Shape{4});

    const auto tile = make_shared<opset1::Tile>(data, repeats);
    NodeBuilder builder(tile, {data, repeats});
    EXPECT_NO_THROW(auto g_tile = ov::as_type_ptr<opset1::Tile>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
