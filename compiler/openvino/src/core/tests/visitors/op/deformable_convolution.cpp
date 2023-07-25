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

TEST(attributes, deformable_convolution_default_attributes) {
    NodeBuilder::get_ops().register_factory<opset1::DeformableConvolution>();
    const Shape inputs_shape{1, 1, 5, 5};
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 1, 5, 5});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{1, 1, 3, 3});
    auto offsets = make_shared<op::Parameter>(element::f32, Shape{1, 18, 3, 3});
    auto strides = Strides{1, 1};
    auto pads_begin = CoordinateDiff{0, 0};
    auto pads_end = CoordinateDiff{0, 0};
    auto dilations = Strides{1, 1};
    auto convolution =
        make_shared<opset1::DeformableConvolution>(data, offsets, filters, strides, pads_begin, pads_end, dilations);
    NodeBuilder builder(convolution, {data, offsets, filters});
    auto g_convolution = ov::as_type_ptr<opset1::DeformableConvolution>(builder.create());

    // attribute count
    const auto expected_attr_count = 7;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_convolution->get_strides(), convolution->get_strides());
    EXPECT_EQ(g_convolution->get_pads_begin(), convolution->get_pads_begin());
    EXPECT_EQ(g_convolution->get_pads_end(), convolution->get_pads_end());
    EXPECT_EQ(g_convolution->get_dilations(), convolution->get_dilations());
    EXPECT_EQ(g_convolution->get_auto_pad(), convolution->get_auto_pad());
    EXPECT_EQ(g_convolution->get_group(), convolution->get_group());
    EXPECT_EQ(g_convolution->get_deformable_group(), convolution->get_deformable_group());
}

TEST(attributes, deformable_convolution_attributes) {
    NodeBuilder::get_ops().register_factory<opset1::DeformableConvolution>();
    const Shape inputs_shape{1, 1, 5, 5};
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 5, 5});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 1, 3, 3});
    auto offsets = make_shared<op::Parameter>(element::f32, Shape{1, 36, 5, 5});
    auto strides = Strides{1, 1};
    auto pads_begin = CoordinateDiff{0, 0};
    auto pads_end = CoordinateDiff{0, 0};
    auto dilations = Strides{1, 1};
    auto convolution = make_shared<opset1::DeformableConvolution>(data,
                                                                  offsets,
                                                                  filters,
                                                                  strides,
                                                                  pads_begin,
                                                                  pads_end,
                                                                  dilations,
                                                                  op::PadType::SAME_LOWER,
                                                                  2,
                                                                  2);
    NodeBuilder builder(convolution, {data, offsets, filters});
    auto g_convolution = ov::as_type_ptr<opset1::DeformableConvolution>(builder.create());

    // attribute count
    const auto expected_attr_count = 7;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_convolution->get_strides(), convolution->get_strides());
    EXPECT_EQ(g_convolution->get_pads_begin(), convolution->get_pads_begin());
    EXPECT_EQ(g_convolution->get_pads_end(), convolution->get_pads_end());
    EXPECT_EQ(g_convolution->get_dilations(), convolution->get_dilations());
    EXPECT_EQ(g_convolution->get_auto_pad(), convolution->get_auto_pad());
    EXPECT_EQ(g_convolution->get_group(), convolution->get_group());
    EXPECT_EQ(g_convolution->get_deformable_group(), convolution->get_deformable_group());
}

TEST(attributes, deformable_convolution_v8_default_attributes) {
    NodeBuilder::get_ops().register_factory<opset8::DeformableConvolution>();
    const Shape inputs_shape{1, 1, 5, 5};
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 1, 5, 5});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{1, 1, 3, 3});
    auto offsets = make_shared<op::Parameter>(element::f32, Shape{1, 18, 3, 3});
    auto strides = Strides{1, 1};
    auto pads_begin = CoordinateDiff{0, 0};
    auto pads_end = CoordinateDiff{0, 0};
    auto dilations = Strides{1, 1};
    auto convolution =
        make_shared<opset8::DeformableConvolution>(data, offsets, filters, strides, pads_begin, pads_end, dilations);
    NodeBuilder builder(convolution, {data, offsets, filters});
    auto g_convolution = ov::as_type_ptr<opset8::DeformableConvolution>(builder.create());

    // attribute count
    const auto expected_attr_count = 8;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_convolution->get_strides(), convolution->get_strides());
    EXPECT_EQ(g_convolution->get_pads_begin(), convolution->get_pads_begin());
    EXPECT_EQ(g_convolution->get_pads_end(), convolution->get_pads_end());
    EXPECT_EQ(g_convolution->get_dilations(), convolution->get_dilations());
    EXPECT_EQ(g_convolution->get_auto_pad(), convolution->get_auto_pad());
    EXPECT_EQ(g_convolution->get_group(), convolution->get_group());
    EXPECT_EQ(g_convolution->get_deformable_group(), convolution->get_deformable_group());
    EXPECT_EQ(g_convolution->get_bilinear_interpolation_pad(), convolution->get_bilinear_interpolation_pad());
}

TEST(attributes, deformable_convolution_v8_attributes) {
    NodeBuilder::get_ops().register_factory<opset8::DeformableConvolution>();
    const Shape inputs_shape{1, 1, 5, 5};
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 5, 5});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 1, 3, 3});
    auto offsets = make_shared<op::Parameter>(element::f32, Shape{1, 36, 5, 5});
    auto mask = make_shared<op::Parameter>(element::f32, Shape{1, 18, 5, 5});
    auto strides = Strides{1, 1};
    auto pads_begin = CoordinateDiff{0, 0};
    auto pads_end = CoordinateDiff{0, 0};
    auto dilations = Strides{1, 1};
    auto convolution = make_shared<opset8::DeformableConvolution>(data,
                                                                  offsets,
                                                                  filters,
                                                                  mask,
                                                                  strides,
                                                                  pads_begin,
                                                                  pads_end,
                                                                  dilations,
                                                                  op::PadType::SAME_LOWER,
                                                                  2,
                                                                  2,
                                                                  true);
    NodeBuilder builder(convolution, {data, offsets, filters, mask});
    auto g_convolution = ov::as_type_ptr<opset8::DeformableConvolution>(builder.create());

    // attribute count
    const auto expected_attr_count = 8;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_convolution->get_strides(), convolution->get_strides());
    EXPECT_EQ(g_convolution->get_pads_begin(), convolution->get_pads_begin());
    EXPECT_EQ(g_convolution->get_pads_end(), convolution->get_pads_end());
    EXPECT_EQ(g_convolution->get_dilations(), convolution->get_dilations());
    EXPECT_EQ(g_convolution->get_auto_pad(), convolution->get_auto_pad());
    EXPECT_EQ(g_convolution->get_group(), convolution->get_group());
    EXPECT_EQ(g_convolution->get_deformable_group(), convolution->get_deformable_group());
    EXPECT_EQ(g_convolution->get_bilinear_interpolation_pad(), convolution->get_bilinear_interpolation_pad());
}
