// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/visitor.hpp"
#include "gtest/gtest.h"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/op/i420_to_bgr.hpp"
#include "openvino/op/i420_to_rgb.hpp"

using namespace std;
using namespace ov;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, convert_color_i420_rgb) {
    NodeBuilder::get_ops().register_factory<op::v8::I420toRGB>();
    auto data = make_shared<op::v0::Parameter>(element::u8, Shape{3, 720, 640, 1});
    auto convert_color = make_shared<op::v8::I420toRGB>(data);
    NodeBuilder builder(convert_color, {data});
    EXPECT_NO_THROW(auto g_concat = ov::as_type_ptr<op::v8::I420toRGB>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, convert_color_i420_bgr) {
    NodeBuilder::get_ops().register_factory<op::v8::I420toBGR>();
    auto data = make_shared<op::v0::Parameter>(element::u8, Shape{3, 720, 640, 1});
    auto convert_color = make_shared<op::v8::I420toBGR>(data);
    NodeBuilder builder(convert_color, {data});
    EXPECT_NO_THROW(auto g_concat = ov::as_type_ptr<op::v8::I420toRGB>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, convert_color_i420_rgb_3planes) {
    NodeBuilder::get_ops().register_factory<op::v8::I420toRGB>();
    auto data1 = make_shared<op::v0::Parameter>(element::u8, Shape{3, 480, 640, 1});
    auto data2 = make_shared<op::v0::Parameter>(element::u8, Shape{3, 240, 320, 1});
    auto data3 = make_shared<op::v0::Parameter>(element::u8, Shape{3, 240, 320, 1});
    auto convert_color = make_shared<op::v8::I420toRGB>(data1, data2, data3);
    NodeBuilder builder(convert_color, {data1, data2, data3});
    EXPECT_NO_THROW(auto g_concat = ov::as_type_ptr<op::v8::I420toRGB>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, convert_color_i420_bgr_3planes) {
    NodeBuilder::get_ops().register_factory<op::v8::I420toBGR>();
    auto data1 = make_shared<op::v0::Parameter>(element::u8, Shape{3, 480, 640, 1});
    auto data2 = make_shared<op::v0::Parameter>(element::u8, Shape{3, 240, 320, 1});
    auto data3 = make_shared<op::v0::Parameter>(element::u8, Shape{3, 240, 320, 1});
    auto convert_color = make_shared<op::v8::I420toBGR>(data1, data2, data3);
    NodeBuilder builder(convert_color, {data1, data2, data3});
    EXPECT_NO_THROW(auto g_concat = ov::as_type_ptr<op::v8::I420toRGB>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
