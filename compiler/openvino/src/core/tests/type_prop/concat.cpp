// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/type_prop.hpp"
#include "gmock/gmock.h"
#include "ngraph/ngraph.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/pass/graph_rewrite.hpp"

using namespace std;
using namespace ngraph;
using namespace testing;

TEST(type_prop, concat_deduce) {
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
    EXPECT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 12, 4}));
}

TEST(type_prop, concat_deduce_wrong_rank) {
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32,
                                             Shape{
                                                 2,
                                                 2,
                                             });
    try {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                                         "have equal dimension everywhere except on the concatenation axis"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_wrong_shape) {
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 5});
    try {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                                         "have equal dimension everywhere except on the concatenation axis"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_axis_oob) {
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 5});
    try {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Concatenation axis (3) is out of bounds"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_axis_barely_in_bounds) {
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 8});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 12});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 2);
    EXPECT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 3, 24}));
}

TEST(type_prop, concat_deduce_elem_type_mismatch) {
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::i32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4});
    try {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument element types are inconsistent"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_et_consistent) {
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::dynamic, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);

    EXPECT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 12, 4}));
}

TEST(type_prop, concat_partial_et_inconsistent) {
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::dynamic, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::i32, Shape{2, 2, 4});
    try {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent element types not detected (some dynamic)";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument element types are inconsistent"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_some_rank_dynamic_others_rank_static_dynamic_rank_inconsistent) {
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, Dimension::dynamic(), 4});
    try {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent ranks not detected (some args rank-dynamic, some args rank-static "
                  "dynamic)";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                                         "have equal dimension everywhere except on the concatenation axis"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_some_rank_dynamic_others_rank_static_dynamic_dims_inconsistent) {
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape{3, 3, Dimension::dynamic()});
    try {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent dimensions not detected (some args rank-dynamic, some args "
                  "rank-static dynamic)";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                                         "have equal dimension everywhere except on the concatenation axis"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_some_rank_dynamic_others_rank_static_dynamic_dims_intransitively_inconsistent) {
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 3, Dimension::dynamic()});
    auto param3 = make_shared<op::Parameter>(element::f32, PartialShape{3, 3, Dimension::dynamic()});
    try {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2, param3}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent dimensions not detected (some args rank-dynamic, some args "
                  "rank-static dynamic)";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                                         "have equal dimension everywhere except on the concatenation axis"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_some_rank_dynamic_others_rank_static_with_concat_axis_static_dims_inconsistent) {
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{2, 2, 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape{3, 3, Dimension::dynamic()});

    try {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent dimensions not detected (some args rank-dynamic, some args "
                  "rank-static dynamic)";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                                         "have equal dimension everywhere except on the concatenation axis"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_all_static_with_concat_axis_static_compatible_result_static) {
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{2, 2, 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4, 3});
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, Dimension::dynamic()});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);

    ASSERT_EQ(c->get_shape(), (Shape{2, 9, 3}));
}

TEST(type_prop, concat_partial_all_static_with_concat_axis_static_dims_incompatible) {
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{2, 2, 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4, 3});
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape{3, 3, Dimension::dynamic()});
    try {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent dimensions not detected (some args rank-dynamic, some args "
                  "rank-static dynamic)";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                                         "have equal dimension everywhere except on the concatenation axis"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_negative_axis_correct) {
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{3, 2, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{7, 2, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4});

    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, -3);

    EXPECT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{12, 2, 4}));
}

TEST(type_prop, concat_partial_negative_axis_incorrect) {
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4});

    try {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, -4);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect negative axis value not detected (out of bounds)";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Concatenation axis (-1) is out of bounds"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

/** \brief Test uses evaluate lower/upper and label of concat op. */
TEST(type_prop, concat_dynamic_value_and_label_propagation) {
    Dimension marked_0 = Dimension(3);
    ov::DimensionTracker::set_label(marked_0, 10);
    PartialShape target_0 = PartialShape{marked_0, 4};

    Dimension marked_1 = Dimension(5);
    ov::DimensionTracker::set_label(marked_1, 15);
    PartialShape target_1 = PartialShape{4, marked_1, 9};

    auto param = make_shared<op::Parameter>(element::f32, Shape{1});
    auto param_0 = make_shared<op::Parameter>(element::f32, target_0);
    auto shape_0 = make_shared<op::ShapeOf>(param_0);

    auto param_1 = make_shared<op::Parameter>(element::f32, target_1);
    auto shape_1 = make_shared<op::ShapeOf>(param_1);

    auto five = op::Constant::create(element::i64, {1}, {5});
    auto target_shape = std::make_shared<op::Concat>(OutputVector{shape_0, five, shape_1}, 0);

    auto bc = make_shared<op::v1::Broadcast>(param, target_shape);
    EXPECT_EQ(bc->get_shape(), (Shape{3, 4, 5, 4, 5, 9}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    const auto labels = get_shape_labels(output_shape);
    ASSERT_THAT(labels, ElementsAre(10, 0, 0, 0, 15, 0));
}

/** \brief Test uses evaluate lower/upper and label of concat op. */
TEST(type_prop, concat_dynamic_value_and_label_propagation_1) {
    Dimension marked_0 = Dimension(3);
    ov::DimensionTracker::set_label(marked_0, 1000);
    PartialShape target_0 = PartialShape{marked_0, 4};

    Dimension marked_1 = Dimension(5);
    ov::DimensionTracker::set_label(marked_1, 1500);
    PartialShape target_1 = PartialShape{4, marked_1, 9};

    auto param = make_shared<op::Parameter>(element::f32, Shape{1});
    auto param_0 = make_shared<op::Parameter>(element::f32, target_0);
    auto shape_0 = make_shared<op::ShapeOf>(param_0);
    auto convert_0 = make_shared<op::Convert>(shape_0, element::i8);

    auto param_1 = make_shared<op::Parameter>(element::f32, target_1);
    auto shape_1 = make_shared<op::ShapeOf>(param_1);
    auto convert_1 = make_shared<op::Convert>(shape_1, element::i8);

    auto five = op::Constant::create(element::i8, {1}, {5});
    auto target_shape = std::make_shared<op::Concat>(OutputVector{convert_0, five, convert_1}, 0);

    auto convert = make_shared<op::Convert>(target_shape, element::i64);

    auto bc = make_shared<op::v1::Broadcast>(param, target_shape);
    EXPECT_EQ(bc->get_shape(), (Shape{3, 4, 5, 4, 5, 9}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    const auto labels = get_shape_labels(output_shape);
    ASSERT_THAT(labels, ElementsAre(1000, 0, 0, 0, 1500, 0));
}

TEST(type_prop, concat_interval_dimensions) {
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{3, 2, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{7, 2, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4});

    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, -3);

    EXPECT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{12, 2, 4}));
}

using PartialShapeVector = std::vector<PartialShape>;
using ConcatTestParams = std::tuple<PartialShapeVector,      // input shapes
                                    std::tuple<int64_t,      // concatenation axis
                                               PartialShape  // expected shape
                                               >>;

class ConcatTest : public TestWithParam<ConcatTestParams> {
protected:
    void SetUp() override {
        int64_t axis;
        PartialShapeVector input_shapes;
        ov::pass::NodeRegistry params;

        std::forward_as_tuple(input_shapes, std::tie(axis, exp_shape)) = GetParam();

        for (const auto& shape : input_shapes) {
            params.make<op::Parameter>(element::f32, shape);
        }

        c = make_shared<op::Concat>(params.get(), axis);
    }

    PartialShape exp_shape;
    std::shared_ptr<op::Concat> c;
};

const auto shapes_with_interval_dim = Values(PartialShapeVector{(PartialShape::dynamic()),
                                                                {2, Dimension(2, 5), 3, 1},
                                                                {2, 4, 3, Dimension(1, 4)},
                                                                {2, 4, 3, 1}});

INSTANTIATE_TEST_SUITE_P(type_prop_interval_dim_mixed_ranks,
                         ConcatTest,
                         Combine(shapes_with_interval_dim,
                                 Values(std::make_tuple(1, PartialShape({2, Dimension(10, -1), 3, 1})),  // axis 1
                                        std::make_tuple(-1, PartialShape({2, 4, 3, Dimension(3, -1)})),  // axis 2
                                        std::make_tuple(2, PartialShape({2, 4, Dimension(9, -1), 1}))    // axis 3
                                        )),
                         PrintToStringParamName());

const auto shapes_all_dynamic_ranks = Values(PartialShapeVector{(PartialShape::dynamic()),
                                                                (PartialShape::dynamic()),
                                                                (PartialShape::dynamic()),
                                                                (PartialShape::dynamic())});

INSTANTIATE_TEST_SUITE_P(type_prop_dynamic_ranks_against_axis_range,
                         ConcatTest,
                         Combine(shapes_all_dynamic_ranks,
                                 Combine(Range<int64_t>(-4, 4), Values(PartialShape::dynamic()))),
                         PrintToStringParamName());

const auto shapes_static_dynamic_ranks =
    Values(PartialShapeVector{PartialShape({4, 2, Dimension::dynamic(), 3}),
                              PartialShape::dynamic(),
                              PartialShape({4, 2, Dimension::dynamic(), Dimension::dynamic()})});

INSTANTIATE_TEST_SUITE_P(type_prop_mixed_ranks_and_dims,
                         ConcatTest,
                         Combine(shapes_static_dynamic_ranks,
                                 Values(
                                     // concat all dynamic dims
                                     std::make_tuple(2, PartialShape({4, 2, Dimension::dynamic(), 3})),
                                     // concat dynamic and interval dim
                                     std::make_tuple(1, PartialShape({4, Dimension(4, -1), Dimension::dynamic(), 3})))),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(type_prop_1d_shapes,
                         ConcatTest,
                         Values(
                             // concat all dynamic dims
                             std::make_tuple(PartialShapeVector{{-1}, {-1}, {-1}},
                                             std::make_tuple(0, PartialShape({-1}))),
                             // concat dynamic and not matching static dims
                             std::make_tuple(PartialShapeVector{{3}, PartialShape::dynamic(), {2}},
                                             std::make_tuple(0, PartialShape({Dimension(5, -1)}))),
                             // concat all static dim
                             std::make_tuple(PartialShapeVector{{3}, {3}, {3}}, std::make_tuple(0, PartialShape({9}))),
                             // concat dynamic and interval dim
                             std::make_tuple(PartialShapeVector{{3}, {Dimension::dynamic()}, {Dimension(3, 4)}},
                                             std::make_tuple(0, PartialShape({Dimension(6, -1)})))),
                         PrintToStringParamName());

/** \brief Shape propagation no exception. */
TEST_P(ConcatTest, partial_shape_propagation) {
    ASSERT_EQ(c->get_default_output().get_partial_shape(), exp_shape);
}
