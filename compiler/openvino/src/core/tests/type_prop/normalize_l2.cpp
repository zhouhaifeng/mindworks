// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, normalize_l2) {
    PartialShape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int64_t>{1, 2});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;
    auto normalize = make_shared<op::v0::NormalizeL2>(data, axes, eps, eps_mode);

    EXPECT_EQ(normalize->get_element_type(), element::f32);
    EXPECT_EQ(normalize->get_output_partial_shape(0), data_shape);
}

TEST(type_prop, normalize_l2_dynamic) {
    PartialShape data_shape{2, Dimension::dynamic(), 3, Dimension(4, 6)};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int64_t>{1, 2});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;
    auto normalize = make_shared<op::v0::NormalizeL2>(data, axes, eps, eps_mode);

    EXPECT_EQ(normalize->get_element_type(), element::f32);
    EXPECT_EQ(normalize->get_output_partial_shape(0), data_shape);
}

TEST(type_prop, normalize_l2_axes_input_not_constant) {
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto axes = make_shared<op::Parameter>(element::u64, Shape{1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;
    ASSERT_NO_THROW(auto op = make_shared<op::v0::NormalizeL2>(data, axes, eps, eps_mode));
}

TEST(type_prop, normalize_l2_invalid_axes_rank) {
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{1, 2}, vector<int64_t>{1, 2});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    try {
        auto normalize = make_shared<op::v0::NormalizeL2>(data, axes, eps, eps_mode);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input axes must be scalar or have rank equal to 1"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, normalize_l2_axes_out_of_bounds) {
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{3, 4});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    try {
        auto normalize = make_shared<op::v0::NormalizeL2>(data, axes, eps, eps_mode);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    } catch (const ov::AssertFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("(axis_range_min <= axis) && (axis <= axis_range_max)"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, normalize_l2_negative_axes) {
    PartialShape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i32, Shape{1}, vector<int64_t>{-1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;
    auto normalize = make_shared<op::v0::NormalizeL2>(data, axes, eps, eps_mode);

    EXPECT_EQ(normalize->get_element_type(), element::f32);
    EXPECT_EQ(normalize->get_reduction_axes(), ov::AxisSet{3});
    EXPECT_EQ(normalize->get_output_partial_shape(0), data_shape);
}
