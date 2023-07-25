// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/visitor.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, fake_quantize_op) {
    NodeBuilder::get_ops().register_factory<opset1::FakeQuantize>();
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto input_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto input_high = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_high = make_shared<op::Parameter>(element::f32, Shape{});

    auto levels = 5;
    auto auto_broadcast = op::AutoBroadcastType::NUMPY;

    const auto fake_quantize =
        make_shared<op::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels, auto_broadcast);
    NodeBuilder builder(fake_quantize, {data, input_low, input_high, output_low, output_high});
    auto g_fake_quantize = ov::as_type_ptr<opset1::FakeQuantize>(builder.create());

    // attribute count
    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_fake_quantize->get_levels(), fake_quantize->get_levels());
    EXPECT_EQ(g_fake_quantize->get_auto_broadcast(), fake_quantize->get_auto_broadcast());
}
