// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;
using namespace testing;

using LogicalNotTestParam = std::tuple<element::Type, PartialShape>;

namespace {
using namespace ngraph::element;
constexpr size_t exp_num_of_outputs = 1;

const auto types = Values(boolean, i16, i32, i64, u16, u32, u64, f32, f64);
const auto static_shapes = Values(PartialShape{0}, PartialShape{1}, PartialShape{2, 3, 7, 8});
const auto dynamic_shapes = Values(PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()},
                                   PartialShape{2, {-1, 5}, {4, -1}, -1, {3, 8}},
                                   PartialShape::dynamic());
}  // namespace

class LogicalNotTest : public TypePropOpTest<op::v1::LogicalNot>, public WithParamInterface<LogicalNotTestParam> {
protected:
    void SetUp() override {
        std::tie(exp_type, exp_shape) = GetParam();
    }

    element::Type exp_type;
    PartialShape exp_shape;
};

INSTANTIATE_TEST_SUITE_P(type_prop_static_shape,
                         LogicalNotTest,
                         Combine(types, static_shapes),
                         PrintToStringParamName());
INSTANTIATE_TEST_SUITE_P(type_prop_dynamic_static_rank,
                         LogicalNotTest,
                         Combine(types, dynamic_shapes),
                         PrintToStringParamName());

TEST_P(LogicalNotTest, propagate_dimensions) {
    const auto input = std::make_shared<op::Parameter>(exp_type, exp_shape);
    const auto op = make_op(input);

    EXPECT_EQ(op->get_element_type(), exp_type);
    EXPECT_EQ(op->get_output_size(), exp_num_of_outputs);
    EXPECT_EQ(op->get_output_partial_shape(0), exp_shape);
}

TEST_P(LogicalNotTest, propagate_labels) {
    if (exp_shape.rank().is_static()) {
        set_shape_labels(exp_shape, 10);
    }
    const auto exp_labels = get_shape_labels(exp_shape);

    const auto input = std::make_shared<op::Parameter>(exp_type, exp_shape);
    const auto op = make_op(input);

    EXPECT_EQ(get_shape_labels(op->get_output_partial_shape(0)), exp_labels);
}

TEST_P(LogicalNotTest, default_ctor) {
    const auto op = std::make_shared<op::v1::LogicalNot>();
    const auto input = std::make_shared<op::Parameter>(exp_type, exp_shape);

    op->set_argument(0, input);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_element_type(), exp_type);
    EXPECT_EQ(op->get_output_size(), exp_num_of_outputs);
    EXPECT_EQ(op->get_output_partial_shape(0), exp_shape);
}
