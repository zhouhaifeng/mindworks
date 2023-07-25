// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/type_prop.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "openvino/core/dimension_tracker.hpp"

using namespace std;
using namespace ngraph;
using namespace testing;

namespace {
using type = ngraph::element::Type;
void type_check(const type& refType) {
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<op::Parameter>(refType, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(refType, updates_shape);
    auto A = op::Constant::create(element::i32, Shape{1}, {1});
    auto scatter_update = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
    EXPECT_EQ(scatter_update->get_output_element_type(0), refType);
    EXPECT_EQ(scatter_update->get_output_shape(0), ref_shape);
}

void incorrect_type_check(const type& refType,
                          const type& indicesType,
                          const type& updatesType,
                          const type& axisType,
                          const std::string& errorStr) {
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<op::Parameter>(refType, ref_shape);
    auto I = make_shared<op::Parameter>(indicesType, indices_shape);
    auto U = make_shared<op::Parameter>(updatesType, updates_shape);
    auto A = op::Constant::create(axisType, Shape{1}, {1});
    try {
        auto G = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect element type of the input";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), errorStr);
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

void incorrect_shape_check(const Shape& refShape,
                           const Shape& indicesShape,
                           const Shape& updatesShape,
                           const Shape& axisShape,
                           const float axisVal,
                           const std::string& errorStr) {
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<op::Parameter>(element::f32, refShape);
    auto I = make_shared<op::Parameter>(element::i32, indicesShape);
    auto U = make_shared<op::Parameter>(element::f32, updatesShape);
    auto A = op::Constant::create(element::i32, axisShape, {axisVal});
    try {
        auto G = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect shape of the input";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), errorStr);
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
}  // namespace

TEST(type_prop, scatter_update_output_type_check_f16) {
    type_check(element::f16);
}

TEST(type_prop, scatter_update_output_type_check_f32) {
    type_check(element::f32);
}

TEST(type_prop, scatter_update_output_type_check_bf16) {
    type_check(element::bf16);
}

TEST(type_prop, scatter_update_output_type_check_i8) {
    type_check(element::i8);
}

TEST(type_prop, scatter_update_output_type_check_i16) {
    type_check(element::i16);
}

TEST(type_prop, scatter_update_output_type_check_i32) {
    type_check(element::i32);
}

TEST(type_prop, scatter_update_output_type_check_i64) {
    type_check(element::i64);
}

TEST(type_prop, scatter_update_output_type_check_u8) {
    type_check(element::u8);
}

TEST(type_prop, scatter_update_output_type_check_u16) {
    type_check(element::u16);
}

TEST(type_prop, scatter_update_output_type_check_u32) {
    type_check(element::u32);
}

TEST(type_prop, scatter_update_output_type_check_u64) {
    type_check(element::u64);
}

TEST(type_prop, scatter_update_v3_fail_updates_data_et_not_equal) {
    incorrect_type_check(element::f32,
                         element::i32,
                         element::u32,
                         element::i32,
                         "Element types for input data and updates do not match");
}

TEST(type_prop, scatter_update_v3_fail_indices_element_type) {
    incorrect_type_check(element::f32,
                         element::f16,
                         element::f32,
                         element::i64,
                         "Indices element type must be of an integral number type");
}

TEST(type_prop, scatter_update_v3_fail_axis_element_type) {
    incorrect_type_check(element::i16,
                         element::u64,
                         element::i16,
                         element::f32,
                         "Axis element type must be of an integral number type");
}

TEST(type_prop, scatter_update_v3_fail_updates_rank) {
    incorrect_shape_check({2, 3, 4},
                          {2, 1},
                          {2, 1, 4},
                          {},
                          0,
                          "Updates rank is expected to be rank(indices) + rank(data) - 1");
}

TEST(type_prop, scatter_update_v3_fail_updates_shape_axis) {
    incorrect_shape_check({2, 3, 4},
                          {2, 1},
                          {2, 2, 1, 4},
                          {},
                          0,
                          "Updates shape must have appropriate dimensions equal to indices and data dimensions");
}

TEST(type_prop, scatter_update_v3_fail_updates_shape_indices) {
    incorrect_shape_check({2, 3, 4},
                          {2, 1},
                          {2, 3, 1, 4},
                          {},
                          1,
                          "Updates shape must have appropriate dimensions equal to indices and data dimensions");
}

TEST(type_prop, scatter_update_v3_fail_updates_shape_data_before_axis) {
    incorrect_shape_check({2, 3, 4},
                          {2, 1},
                          {3, 2, 1, 4},
                          {},
                          1,
                          "Updates shape must have appropriate dimensions equal to indices and data dimensions");
}

TEST(type_prop, scatter_update_v3_fail_updates_shape_data_after_axis) {
    incorrect_shape_check({2, 3, 4},
                          {2, 1},
                          {2, 2, 1, 5},
                          {},
                          1,
                          "Updates shape must have appropriate dimensions equal to indices and data dimensions");
}

TEST(type_prop, scatter_update_v3_fail_axis_shape) {
    incorrect_shape_check({2, 3, 4},
                          {2, 1},
                          {2, 2, 1, 4},
                          {2},
                          1,
                          "Axis input shape is required to be scalar or 1D tensor");
}

TEST(type_prop, scatter_update_v3_dynamic_data_shape) {
    PartialShape ref_shape = PartialShape::dynamic();
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<op::Parameter>(element::i8, ref_shape);
    auto I = make_shared<op::Parameter>(element::i16, indices_shape);
    auto U = make_shared<op::Parameter>(element::i8, updates_shape);
    auto A = op::Constant::create(element::i16, Shape{}, {1});

    auto scatter_update = make_shared<op::v3::ScatterUpdate>(R, I, U, A);
    EXPECT_EQ(scatter_update->get_output_element_type(0), element::i8);
    EXPECT_TRUE(scatter_update->get_output_partial_shape(0).is_dynamic());
}

TEST(type_prop, scatter_update_v3_interval_label_data_shape) {
    auto labeled_dim = Dimension(1, 9);
    ov::label_t label = 222;
    ov::DimensionTracker::set_label(labeled_dim, label);
    PartialShape data_shape = PartialShape{-1, {2, 8}, labeled_dim, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{3, 2, 1, 2, 4};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto idx = make_shared<op::Parameter>(element::i32, indices_shape);
    auto updates = make_shared<op::Parameter>(element::f32, updates_shape);
    auto axis = op::Constant::create(element::i32, Shape{}, {1});

    auto scatter_update = make_shared<op::v3::ScatterUpdate>(data, idx, updates, axis);

    const auto& output_shape = scatter_update->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, data_shape);
    EXPECT_THAT(get_shape_labels(output_shape), ElementsAre(ov::no_label, ov::no_label, label, ov::no_label));
    EXPECT_EQ(scatter_update->get_output_element_type(0), element::f32);
}

TEST(type_prop, scatter_update_v3_value_label_propagation) {
    auto labeled_dim = Dimension(5, 7);
    ov::label_t label = 2345664;
    ov::DimensionTracker::set_label(labeled_dim, label);
    PartialShape data_shape = PartialShape{labeled_dim};

    auto data = make_shared<op::Parameter>(element::i8, data_shape);
    auto shape_of = make_shared<op::v3::ShapeOf>(data);
    auto scatter_update = make_shared<op::v3::ScatterUpdate>(op::Constant::create(element::i64, Shape{2}, {1, 0}),
                                                             op::Constant::create(element::i64, Shape{1}, {1}),
                                                             shape_of,
                                                             op::Constant::create(element::i64, Shape{1}, {0}));
    auto broadcast =
        make_shared<op::v3::Broadcast>(op::Constant::create(element::i64, Shape{1, 1}, {4}), scatter_update);

    const auto& output_shape = broadcast->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, PartialShape({1, {5, 7}}));
    EXPECT_EQ(ov::DimensionTracker::get_label(output_shape[0]), ov::no_label);
    EXPECT_EQ(ov::DimensionTracker::get_label(output_shape[1]), label);
}

TEST(type_prop, scatter_update_v3_partial_value_propagation) {
    // strided slice should take from 5 to 7 elements from the 10 elements in the input data
    auto input = make_shared<op::Parameter>(element::i8, PartialShape{ov::Dimension(5, 7)});
    auto shape = make_shared<op::v3::ShapeOf>(input);
    auto scatter_update = make_shared<op::v3::ScatterUpdate>(op::Constant::create(element::i64, Shape{2}, {1, 0}),
                                                             op::Constant::create(element::i64, Shape{1}, {1}),
                                                             shape,
                                                             op::Constant::create(element::i64, Shape{1}, {0}));
    const auto& masks = std::vector<int64_t>(0, 2);
    const auto& strided_slice = make_shared<op::v1::StridedSlice>(
        op::Constant::create(element::i64, Shape{1, 10}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0}),
        op::Constant::create(element::i64, Shape{2}, {0, 0}),
        scatter_update,
        op::Constant::create(element::i64, Shape{2}, {1, 1}),
        masks,
        masks);

    const auto& reshape = make_shared<op::v1::Reshape>(strided_slice, scatter_update, false);
    EXPECT_EQ(reshape->get_output_partial_shape(0), PartialShape({1, {5, 7}}));
}
