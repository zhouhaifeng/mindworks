// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/visitor.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset8.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, max_pool_op) {
    NodeBuilder::get_ops().register_factory<opset1::MaxPool>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});

    auto strides = Strides{2};
    auto pads_begin = Shape{1};
    auto pads_end = Shape{1};
    auto kernel = Shape{1};
    auto rounding_mode = op::RoundingType::FLOOR;
    auto auto_pad = op::PadType::EXPLICIT;

    auto max_pool = make_shared<opset1::MaxPool>(data, strides, pads_begin, pads_end, kernel, rounding_mode, auto_pad);
    NodeBuilder builder(max_pool, {data});
    auto g_max_pool = ov::as_type_ptr<opset1::MaxPool>(builder.create());

    EXPECT_EQ(g_max_pool->get_strides(), max_pool->get_strides());
    EXPECT_EQ(g_max_pool->get_pads_begin(), max_pool->get_pads_begin());
    EXPECT_EQ(g_max_pool->get_pads_end(), max_pool->get_pads_end());
    EXPECT_EQ(g_max_pool->get_kernel(), max_pool->get_kernel());
    EXPECT_EQ(g_max_pool->get_rounding_type(), max_pool->get_rounding_type());
    EXPECT_EQ(g_max_pool->get_auto_pad(), max_pool->get_auto_pad());
}

TEST(attributes, max_pool_v8_op) {
    NodeBuilder::get_ops().register_factory<opset8::MaxPool>();
    const auto data = make_shared<op::Parameter>(element::i32, Shape{1, 3, 37, 37});

    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};
    const auto pads_begin = Shape{1, 1};
    const auto pads_end = Shape{1, 1};
    const auto kernel = Shape{2, 2};
    const auto rounding_mode = op::RoundingType::CEIL;
    const auto auto_pad = op::PadType::EXPLICIT;
    const element::Type& index_element_type = element::i32;

    const auto max_pool = make_shared<opset8::MaxPool>(data,
                                                       strides,
                                                       dilations,
                                                       pads_begin,
                                                       pads_end,
                                                       kernel,
                                                       rounding_mode,
                                                       auto_pad,
                                                       index_element_type);
    NodeBuilder builder(max_pool, {data});
    auto g_max_pool = ov::as_type_ptr<opset8::MaxPool>(builder.create());

    EXPECT_EQ(g_max_pool->get_strides(), max_pool->get_strides());
    EXPECT_EQ(g_max_pool->get_dilations(), max_pool->get_dilations());
    EXPECT_EQ(g_max_pool->get_pads_begin(), max_pool->get_pads_begin());
    EXPECT_EQ(g_max_pool->get_pads_end(), max_pool->get_pads_end());
    EXPECT_EQ(g_max_pool->get_kernel(), max_pool->get_kernel());
    EXPECT_EQ(g_max_pool->get_rounding_type(), max_pool->get_rounding_type());
    EXPECT_EQ(g_max_pool->get_auto_pad(), max_pool->get_auto_pad());
    EXPECT_EQ(g_max_pool->get_index_element_type(), max_pool->get_index_element_type());
}
