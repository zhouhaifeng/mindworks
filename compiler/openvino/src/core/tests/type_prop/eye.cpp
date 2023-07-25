// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "eye_shape_inference.hpp"
#include "gtest/gtest.h"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov;
using namespace testing;
using namespace ov::opset10;

TEST(type_prop, eye_constant) {
    auto num_rows = op::v0::Constant::create(element::i64, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i64, Shape{}, {3});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {0});

    auto eye = std::make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::bf16);

    EXPECT_EQ(eye->get_output_element_type(0), element::bf16);
    EXPECT_EQ(eye->get_output_size(), 1);
    EXPECT_EQ(eye->get_output_partial_shape(0), ov::PartialShape({6, 3}));
}

TEST(type_prop, eye_batch_shape_constant) {
    auto num_rows = op::v0::Constant::create(element::i64, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i64, Shape{}, {3});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {0});
    auto batch_shape = op::v0::Constant::create(element::i64, Shape{1}, {2});

    auto eye = std::make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, batch_shape, element::bf16);

    EXPECT_EQ(eye->get_output_element_type(0), element::bf16);
    EXPECT_EQ(eye->get_output_partial_shape(0), PartialShape({2, 6, 3}));
}

TEST(type_prop, eye_rows_param) {
    constexpr label_t row_label = 2;
    auto rows_dim = Dimension{0, 1};
    DimensionTracker::set_label(rows_dim, row_label);

    auto num_rows = make_shared<op::v0::Parameter>(element::i64, PartialShape{rows_dim});
    auto num_columns = op::v0::Constant::create(element::i64, Shape{}, {10});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::f32);

    EXPECT_EQ(eye->get_output_element_type(0), element::f32);
    EXPECT_EQ(eye->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 10}));
    EXPECT_THAT(get_shape_labels(eye->get_output_partial_shape(0)), Each(no_label));
}

TEST(type_prop, eye_rows_const) {
    constexpr label_t columns_label = 2;
    auto columns_dim = Dimension{0, 1};
    DimensionTracker::set_label(columns_dim, columns_label);

    auto num_rows = op::v0::Constant::create(element::i64, Shape{}, {10});
    auto num_columns = make_shared<op::v0::Parameter>(element::i64, PartialShape{columns_dim});
    auto diagonal_index = make_shared<op::v0::Parameter>(element::i64, PartialShape{1});

    auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::f32);

    EXPECT_EQ(eye->get_output_element_type(0), element::f32);
    EXPECT_EQ(eye->get_output_partial_shape(0), PartialShape({10, Dimension::dynamic()}));
    EXPECT_THAT(get_shape_labels(eye->get_output_partial_shape(0)), Each(no_label));
}

TEST(type_prop, eye_batch_shape_const) {
    constexpr label_t batch_label = 2;
    auto batch_dim = Dimension{2};
    DimensionTracker::set_label(batch_dim, batch_label);

    auto num_rows = make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    auto num_columns = num_rows;
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {0});
    auto batch_shape = op::v0::Constant::create(element::i64, Shape{2}, {2, 3});

    auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, batch_shape, element::f32);

    EXPECT_EQ(eye->get_output_element_type(0), element::f32);
    EXPECT_EQ(eye->get_output_partial_shape(0), PartialShape({2, 3, Dimension::dynamic(), Dimension::dynamic()}));
    EXPECT_THAT(get_shape_labels(eye->get_output_partial_shape(0)), Each(no_label));
}

TEST(type_prop, eye_batch_shape_params) {
    auto num_rows = make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    auto num_columns = make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    auto diagonal_index = make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    auto batch_shape = make_shared<op::v0::Parameter>(element::i64, PartialShape{2});

    auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, batch_shape, element::f64);

    EXPECT_EQ(eye->get_output_element_type(0), element::f64);
    EXPECT_EQ(eye->get_output_partial_shape(0), PartialShape().dynamic(4));
}

TEST(type_prop, eye_batch_shape_shape_of) {
    auto batch_shape = PartialShape{{1, 10}, {10, 25}};
    set_shape_labels(batch_shape, 10);

    auto num_rows = Constant::create(element::i64, Shape{}, {10});
    auto num_columns = num_rows;
    auto diagonal_index = make_shared<Parameter>(element::i64, PartialShape{1});
    auto batch = make_shared<Parameter>(element::i64, batch_shape);
    auto shape_of = make_shared<ShapeOf>(batch);

    auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, shape_of, element::f64);

    EXPECT_EQ(eye->get_output_element_type(0), element::f64);
    EXPECT_EQ(eye->get_output_partial_shape(0), PartialShape({{1, 10}, {10, 25}, 10, 10}));
    EXPECT_THAT(get_shape_labels(eye->get_output_partial_shape(0)), ElementsAre(10, 11, no_label, no_label));
}

TEST(type_prop, eye_invalid_num_rows_value) {
    auto num_rows = op::v0::Constant::create(element::i64, Shape{1}, {-4});
    auto num_columns = op::v0::Constant::create(element::i64, Shape{}, {2});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    OV_EXPECT_THROW(auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32),
                    AssertFailure,
                    HasSubstr("Value -4 not in range [0:"));
}

TEST(type_prop, eye_invalid_num_columns_value) {
    auto num_rows = op::v0::Constant::create(element::i64, Shape{1}, {6});
    auto num_columns = op::v0::Constant::create(element::i64, Shape{}, {-6});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    OV_EXPECT_THROW(auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32),
                    AssertFailure,
                    HasSubstr("Value -6 not in range [0:"));
}

TEST(type_prop, eye_invalid_num_rows_type) {
    auto num_rows = op::v0::Constant::create(element::bf16, Shape{}, {1.2});
    auto num_columns = op::v0::Constant::create(element::i64, Shape{}, {2});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    try {
        auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid num rows value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Type of the 'num_rows' should be int32 or int64. Got: bf16"));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_num_columns_type) {
    auto num_rows = op::v0::Constant::create(element::i64, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::bf16, Shape{}, {6.5});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    try {
        auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid num columns value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Type of the 'num_columns' should be int32 or int64. Got: bf16"));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_diagonal_index_type) {
    auto num_rows = op::v0::Constant::create(element::i64, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i64, Shape{}, {6});
    auto diagonal_index = op::v0::Constant::create(element::bf16, Shape{}, {6.5});

    try {
        auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid diagonal index value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Type of the 'diagonal_index' should be int32 or int64. Got: bf16"));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_num_rows_shape) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{3}, {1, 1, 1});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    try {
        auto Eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid num rows value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'num_columns' value input should have 1 element."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_num_columns_shape) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{3}, {1, 1, 1});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    try {
        auto Eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid num columns value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'num_columns' value input should have 1 element."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_diagonal_index_shape) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto diagonal_index = op::v0::Constant::create(element::i32, Shape{3}, {1, 1, 1});

    try {
        auto Eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid diagonal index value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'diagonal_index' value input should have 1 element."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_num_rows_rank) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{1, 1}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{1}, {1});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    try {
        auto Eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid num rows value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'num_rows' value must be a scalar or 1D tensor."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_num_columns_rank) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{2, 1}, {1, 2});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    try {
        auto Eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid num columns value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'num_columns' value must be a scalar or 1D tensor."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_diagonal_index_rank) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{}, {2});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{1, 2, 1}, {2, 8});

    try {
        auto Eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid diagonal index value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'diagonal_index' value must be a scalar or 1D tensor."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_dynamic_batch_shape_dyn_rank) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{}, {4});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});
    auto batch_shape = make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());

    auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, batch_shape, element::i32);

    EXPECT_EQ(eye->get_output_element_type(0), element::i32);
    EXPECT_EQ(eye->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, eye_dynamic_batch_shape_1D) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{}, {4});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});
    auto batch_shape = make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));

    auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, batch_shape, element::i32);

    EXPECT_EQ(eye->get_output_element_type(0), element::i32);
    EXPECT_EQ(eye->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, eye_dynamic_batch_shape_invalid_rank) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{}, {4});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});
    auto batch_shape = make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(2));

    try {
        auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, batch_shape, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid 'batch_shape' value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'batch_shape' input must be a 1D tensor."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

class TypePropEyeV9Test : public TypePropOpTest<op::v9::Eye> {};

TEST_F(TypePropEyeV9Test, eye_batch_shape_param_other_ins_const) {
    auto num_rows = Constant::create(element::i64, Shape{1}, {5});
    auto num_columns = Constant::create(element::i64, Shape{1}, {6});
    auto diagonal_index = Constant::create(element::i64, Shape{1}, {0});
    auto batch_shape = std::make_shared<Parameter>(element::i64, PartialShape{3});

    auto op = make_op(num_rows, num_columns, diagonal_index, batch_shape, element::f32);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, -1, -1, 5, 6}));
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), Each(no_label));
}

TEST_F(TypePropEyeV9Test, default_ctor) {
    auto num_rows = Constant::create(element::i64, Shape{1}, {2});
    auto num_columns = Constant::create(element::i64, Shape{1}, {16});
    auto diagonal_index = Constant::create(element::i64, Shape{1}, {0});
    auto batch_shape = Constant::create(element::i64, Shape{3}, {3, 1, 2});

    auto op = make_op();

    op->set_arguments(OutputVector{num_rows, num_columns, diagonal_index, batch_shape});
    op->set_out_type(element::i32);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({3, 1, 2, 2, 16}));
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), Each(no_label));
}

TEST_F(TypePropEyeV9Test, default_ctor_no_arguments) {
    auto op = make_op();
    op->set_out_type(element::i32);

    int64_t rows = 8, cols = 5;
    auto batch = std::array<int32_t, 3>{2, 4, 1};
    const auto constant_map = std::map<size_t, HostTensorPtr>{
        {0, std::make_shared<HostTensor>(element::i64, Shape{}, &rows)},
        {1, std::make_shared<HostTensor>(element::i64, Shape{}, &cols)},
        {3, std::make_shared<HostTensor>(element::i32, Shape{batch.size()}, batch.data())}};

    const auto output_shapes =
        op::v9::shape_infer(op.get(), PartialShapes{{}, {}, {}, {3}}, make_tensor_accessor(constant_map));

    EXPECT_EQ(op->get_out_type(), element::i32);
    EXPECT_EQ(output_shapes.front(), PartialShape({2, 4, 1, 8, 5}));
}

TEST_F(TypePropEyeV9Test, preserve_partial_values_and_labels) {
    auto rows_shape = PartialShape{{2, 5}};
    set_shape_labels(rows_shape, 30);
    auto rows = std::make_shared<Parameter>(element::i64, rows_shape);
    auto num_rows = make_shared<ShapeOf>(rows);

    auto columns_shape = PartialShape{{1, 3}};
    set_shape_labels(columns_shape, 40);
    auto columns = std::make_shared<Parameter>(element::i64, columns_shape);
    auto shape_of_columns = make_shared<ShapeOf>(columns);
    auto num_columns = std::make_shared<Squeeze>(shape_of_columns, Constant::create(element::i64, Shape{}, {0}));

    auto batch_shape = PartialShape{{1, 10}, {10, 25}};
    set_shape_labels(batch_shape, 10);

    auto diagonal_index = make_shared<Parameter>(element::i64, PartialShape{1});
    auto batch = make_shared<ShapeOf>(make_shared<Parameter>(element::i64, batch_shape));

    auto op = make_op(num_rows, num_columns, diagonal_index, batch, element::i32);

    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{1, 10}, {10, 25}, {2, 5}, {1, 3}}));
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), ElementsAre(10, 11, 30, 40));
}
