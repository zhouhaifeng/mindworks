// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "convolution_shape_inference.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;
using namespace testing;

TEST(type_prop, convolution_v1_partial_rank) {
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::v1::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 padding_below,
                                                 padding_above,
                                                 window_dilation_strides);

    EXPECT_EQ(conv->get_output_partial_shape(0), PartialShape({-1, -1, {1, -1}, {1, -1}}));
}

TEST(type_prop, convolution_v1_partial_auto_padding_same) {
    PartialShape data_batch_shape{1, 1, 5, 5};
    PartialShape filters_shape{1, 1, 3, 3};
    set_shape_labels(data_batch_shape, 10);
    set_shape_labels(filters_shape, 20);
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv =
        make_shared<op::v1::Convolution>(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    EXPECT_EQ(conv->get_output_partial_shape(0), (PartialShape{1, 1, 5, 5}));
    EXPECT_THAT(get_shape_labels(conv->get_output_partial_shape(0)), ElementsAre(10, 20, ov::no_label, ov::no_label));
    EXPECT_EQ(conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    EXPECT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, convolution_v1_partial_auto_padding_same_nc_dims_dynamic_same_lower) {
    PartialShape data_batch_shape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    PartialShape filters_shape{1, 1, 3, 3};
    set_shape_labels(data_batch_shape, 10);
    set_shape_labels(filters_shape, 20);
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv =
        make_shared<op::v1::Convolution>(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    EXPECT_EQ(conv->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 1, 5, 5}));
    EXPECT_THAT(get_shape_labels(conv->get_output_partial_shape(0)), ElementsAre(10, 20, ov::no_label, ov::no_label));
    EXPECT_EQ(conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    EXPECT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, convolution_v1_partial_auto_padding_same_nc_dims_dynamic_same_upper) {
    const PartialShape data_batch_shape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape filters_shape{1, 1, 2, 2};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_UPPER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv =
        make_shared<op::v1::Convolution>(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    ASSERT_EQ(conv->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 1, 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, convolution_v1_partial_auto_padding_same_spatial_dims_dynamic) {
    PartialShape data_batch_shape{1, 1, Dimension::dynamic(), {3, 5}};
    PartialShape filters_shape{1, 1, 3, 3};
    set_shape_labels(data_batch_shape, 10);
    set_shape_labels(filters_shape, 20);
    Strides strides{2, 2};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv =
        make_shared<op::v1::Convolution>(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    EXPECT_EQ(conv->get_output_partial_shape(0), PartialShape({1, 1, Dimension::dynamic(), {2, 3}}));
    EXPECT_THAT(get_shape_labels(conv->get_output_partial_shape(0)), ElementsAre(10, 20, ov::no_label, ov::no_label));
    EXPECT_EQ(conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_pads_end(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, convolution_v1_partial_data_shape_dynamic) {
    const PartialShape data_batch_shape{PartialShape::dynamic()};
    const PartialShape filters_shape{1, 1, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv =
        make_shared<op::v1::Convolution>(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    EXPECT_EQ(conv->get_output_partial_shape(0),
              PartialShape({Dimension::dynamic(), 1, Dimension::dynamic(), Dimension::dynamic()}));
    EXPECT_EQ(conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_pads_end(), (CoordinateDiff{0, 0}));
}

class TypePropConvolutionV1Test : public TypePropOpTest<op::v1::Convolution> {
protected:
    CoordinateDiff empty_pad{};
};

TEST_F(TypePropConvolutionV1Test, default_ctor) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 5, 5});
    const auto filters = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 4, 4});

    const auto op = make_op();
    op->set_arguments(OutputVector{data, filters});
    op->set_strides({1, 3});
    op->set_dilations({1, 2});
    op->set_pads_begin({2, 2});
    op->set_pads_end({2, 2});
    op->set_auto_pad(op::PadType::EXPLICIT);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_strides(), Strides({1, 3}));
    EXPECT_EQ(op->get_dilations(), Strides({1, 2}));
    EXPECT_EQ(op->get_pads_begin(), CoordinateDiff({2, 2}));
    EXPECT_EQ(op->get_pads_end(), CoordinateDiff({2, 2}));
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({1, 2, 6, 1}));
}

TEST_F(TypePropConvolutionV1Test, data_dynamic_rank_filters_2d) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto filters = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 4, 4});
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};

    auto op = make_op(data, filters, strides, empty_pad, empty_pad, dilations, op::PadType::SAME_UPPER);

    EXPECT_THAT(op->get_pads_begin(), ElementsAre(0, 0));
    EXPECT_THAT(op->get_pads_end(), ElementsAre(0, 0));
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, 2, -1, -1}));
}

TEST_F(TypePropConvolutionV1Test, data_rank_to_low) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{2, 3});
    const auto filters = make_shared<op::Parameter>(element::f32, PartialShape{2, 3});
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};

    OV_EXPECT_THROW(auto op = make_op(data, filters, strides, empty_pad, empty_pad, dilations, op::PadType::SAME_LOWER),
                    NodeValidationFailure,
                    HasSubstr("Expected a 3D, 4D or 5D tensor for the input"));
}

TEST_F(TypePropConvolutionV1Test, data_rank_to_high) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 5, 5, 5, 5});
    const auto filters = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 4, 4, 4, 4});
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};

    OV_EXPECT_THROW(auto op = make_op(data, filters, strides, empty_pad, empty_pad, dilations, op::PadType::SAME_LOWER),
                    NodeValidationFailure,
                    HasSubstr("Expected a 3D, 4D or 5D tensor for the input"));
}

TEST_F(TypePropConvolutionV1Test, data_and_filters_rank_not_compatible) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 5, 5});
    const auto filters = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 4});
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};

    OV_EXPECT_THROW(auto op = make_op(data, filters, strides, empty_pad, empty_pad, dilations, op::PadType::SAME_LOWER),
                    NodeValidationFailure,
                    HasSubstr("Data batch and filters rank do not match"));
}

TEST_F(TypePropConvolutionV1Test, data_and_filters_channel_number_not_compatible) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{2, 2, 5, 5});
    const auto filters = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 4, 4});
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};

    OV_EXPECT_THROW(auto op = make_op(data, filters, strides, empty_pad, empty_pad, dilations, op::PadType::SAME_LOWER),
                    NodeValidationFailure,
                    HasSubstr("Data batch channel count (2) does not match filter input channel count (3)"));
}

TEST_F(TypePropConvolutionV1Test, strides_not_defined_only_for_spatial_dims) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 5, 5});
    const auto filters = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 4, 4});
    const auto strides = Strides{1, 1, 1};
    const auto dilations = Strides{1, 1};

    OV_EXPECT_THROW(auto op = make_op(data, filters, strides, empty_pad, empty_pad, dilations, op::PadType::SAME_LOWER),
                    NodeValidationFailure,
                    HasSubstr("Strides should be defined for all and only spatial dimensions."));
}

TEST_F(TypePropConvolutionV1Test, dilations_not_defined_only_for_spatial_dims) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 5, 5});
    const auto filters = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 4, 4});
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1};

    OV_EXPECT_THROW(auto op = make_op(data, filters, strides, empty_pad, empty_pad, dilations, op::PadType::SAME_LOWER),
                    NodeValidationFailure,
                    HasSubstr("Dilations should be defined for all and only spatial dimensions."));
}

TEST_F(TypePropConvolutionV1Test, strides_has_zeros) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 5, 5});
    const auto filters = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 4, 4});
    const auto strides = Strides{1, 0};
    const auto dilations = Strides{1, 1};

    OV_EXPECT_THROW(auto op = make_op(data, filters, strides, empty_pad, empty_pad, dilations, op::PadType::SAME_LOWER),
                    NodeValidationFailure,
                    HasSubstr("Strides has zero dimension"));
}

TEST_F(TypePropConvolutionV1Test, dilations_has_zeros) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 5, 5});
    const auto filters = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 4, 4});
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{0, 1};

    OV_EXPECT_THROW(auto op = make_op(data, filters, strides, empty_pad, empty_pad, dilations, op::PadType::SAME_LOWER),
                    NodeValidationFailure,
                    HasSubstr("Filter dilations has zero dimension"));
}

TEST_F(TypePropConvolutionV1Test, pads_not_defined_for_spatial_only) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 5, 5});
    const auto filters = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 4, 4});
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};
    const auto pads_begin = CoordinateDiff{2, 2};
    const auto pads_end = CoordinateDiff{2, 2, 2};

    OV_EXPECT_THROW(auto op = make_op(data, filters, strides, pads_begin, pads_end, dilations),
                    NodeValidationFailure,
                    HasSubstr("Pads begin and end should be defined for all and only spatial dimensions."));
}
