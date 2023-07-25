// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;
using namespace testing;

TEST(type_prop, bin_convolution_auto_padding_same) {
    PartialShape data_batch_shape{1, 1, 5, 5};
    PartialShape filters_shape{1, 1, 3, 3};
    set_shape_labels(data_batch_shape, 10);
    set_shape_labels(filters_shape, 20);
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::u1, filters_shape);

    auto conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                       filters,
                                                       strides,
                                                       pads_begin,
                                                       pads_end,
                                                       dilations,
                                                       mode,
                                                       pad_value,
                                                       auto_pad);

    EXPECT_THAT(get_shape_labels(conv->get_output_partial_shape(0)), ElementsAre(10, 20, ov::no_label, ov::no_label));
    EXPECT_EQ(conv->get_output_partial_shape(0), (PartialShape{1, 1, 5, 5}));
    EXPECT_EQ(conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    EXPECT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, bin_convolution_auto_padding_same_lower_spatial_dims_static) {
    PartialShape data_batch_shape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    PartialShape filters_shape{Dimension::dynamic(), Dimension::dynamic(), 3, 3};
    set_shape_labels(data_batch_shape, 10);
    set_shape_labels(filters_shape, 20);
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::u1, filters_shape);

    auto conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                       filters,
                                                       strides,
                                                       pads_begin,
                                                       pads_end,
                                                       dilations,
                                                       mode,
                                                       pad_value,
                                                       auto_pad);

    EXPECT_THAT(get_shape_labels(conv->get_output_partial_shape(0)), ElementsAre(10, 20, ov::no_label, ov::no_label));
    EXPECT_EQ(conv->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), Dimension::dynamic(), 5, 5}));
    EXPECT_EQ(conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    EXPECT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, bin_convolution_auto_padding_same_upper_spatial_dims_static) {
    const PartialShape data_batch_shape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape filters_shape{Dimension::dynamic(), Dimension::dynamic(), 2, 2};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::SAME_UPPER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::u1, filters_shape);

    auto conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                       filters,
                                                       strides,
                                                       pads_begin,
                                                       pads_end,
                                                       dilations,
                                                       mode,
                                                       pad_value,
                                                       auto_pad);

    EXPECT_EQ(conv->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), Dimension::dynamic(), 5, 5}));
    EXPECT_EQ(conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, bin_convolution_auto_padding_same_data_batch_spatial_dims_dynamic) {
    PartialShape data_batch_shape{1, 1, Dimension::dynamic(), 5};
    PartialShape filters_shape{Dimension::dynamic(), 1, 3, 3};
    set_shape_labels(data_batch_shape, 10);
    set_shape_labels(filters_shape, 20);
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::u1, filters_shape);

    auto conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                       filters,
                                                       strides,
                                                       pads_begin,
                                                       pads_end,
                                                       dilations,
                                                       mode,
                                                       pad_value,
                                                       auto_pad);

    EXPECT_THAT(get_shape_labels(conv->get_output_partial_shape(0)), ElementsAre(10, 20, ov::no_label, ov::no_label));
    EXPECT_EQ(conv->get_output_partial_shape(0), (PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), 5}));
    EXPECT_EQ(conv->get_pads_begin(), (CoordinateDiff{0, 1}));
    EXPECT_EQ(conv->get_pads_end(), (CoordinateDiff{0, 1}));
}

TEST(type_prop, bin_convolution_dyn_data_batch) {
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::EXPLICIT;

    const auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto filters = make_shared<op::Parameter>(element::u1, PartialShape{1, 1, 3, 3});
    const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                 filters,
                                                                 Strides{},
                                                                 CoordinateDiff{},
                                                                 CoordinateDiff{},
                                                                 Strides{},
                                                                 mode,
                                                                 pad_value,
                                                                 auto_pad);

    EXPECT_EQ(bin_conv->get_output_partial_shape(0), (PartialShape{-1, 1, {1, -1}, {1, -1}}));
}

TEST(type_prop, bin_convolution_dyn_filters) {
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::EXPLICIT;

    const auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape{1, 1, 5, 5});
    const auto filters = make_shared<op::Parameter>(element::u1, PartialShape::dynamic());
    const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                 filters,
                                                                 Strides{},
                                                                 CoordinateDiff{},
                                                                 CoordinateDiff{},
                                                                 Strides{},
                                                                 mode,
                                                                 pad_value,
                                                                 auto_pad);

    EXPECT_EQ(bin_conv->get_output_partial_shape(0), (PartialShape{1, -1, {1, 5}, {1, 5}}));
}

TEST(type_prop, bin_convolution_dyn_data_batch_and_filters) {
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::EXPLICIT;

    const auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto filters = make_shared<op::Parameter>(element::u1, PartialShape::dynamic());
    const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                 filters,
                                                                 Strides{},
                                                                 CoordinateDiff{},
                                                                 CoordinateDiff{},
                                                                 Strides{},
                                                                 mode,
                                                                 pad_value,
                                                                 auto_pad);

    EXPECT_EQ(bin_conv->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, bin_convolution_invalid_inputs_et) {
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::EXPLICIT;
    try {
        const auto data_batch = make_shared<op::Parameter>(element::boolean, PartialShape{1, 1, 5, 5});
        const auto filters = make_shared<op::Parameter>(element::u1, PartialShape{1, 1, 3, 3});
        const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                     filters,
                                                                     Strides{},
                                                                     CoordinateDiff{},
                                                                     CoordinateDiff{},
                                                                     Strides{},
                                                                     mode,
                                                                     pad_value,
                                                                     auto_pad);
        // data batch element type must be float point
        FAIL() << "Incompatible element type of data batch input not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Data batch element type must be numeric");
    } catch (...) {
        FAIL() << "Data batch element type validation check failed for unexpected reason";
    }
    // TODO: Add test with check filters element type once u1 is supported in nGraph Python API
    // (#49517)
}

TEST(type_prop, bin_convolution_incompatible_input_channels) {
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::EXPLICIT;

    auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape{1, 1, 5, 5});
    auto filters = make_shared<op::Parameter>(element::u1, PartialShape{1, 2, 3, 3});

    try {
        auto conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                           filters,
                                                           Strides{},
                                                           CoordinateDiff{},
                                                           CoordinateDiff{},
                                                           Strides{},
                                                           mode,
                                                           pad_value,
                                                           auto_pad);
        FAIL() << "Incompatible input channel dimension in data batch and filters not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch channel count"));
    } catch (...) {
        FAIL() << "Data batch and filters input channel count validation check failed for "
                  "unexpected reason";
    }
}

TEST(type_prop, bin_convolution_invalid_input_ranks) {
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::EXPLICIT;

    // data partial shape provided is rank 4 (Conv2D)
    // filter partial shape provided is rank 5 (Conv3D)
    try {
        const auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape{1, 1, 5, 5});
        const auto filters = make_shared<op::Parameter>(element::u1, PartialShape{1, 1, 3, 3, 3});
        const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                     filters,
                                                                     Strides{},
                                                                     CoordinateDiff{},
                                                                     CoordinateDiff{},
                                                                     Strides{},
                                                                     mode,
                                                                     pad_value,
                                                                     auto_pad);
        // data batch and filters have incompatible ranks
        FAIL() << "Incompatible input ranks not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Data batch and filters rank do not match");
    } catch (...) {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }

    // data partial shape provided is rank 5 (Conv3D)
    // filter partial shape provided is rank 4 (Conv2D)
    try {
        const auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape{1, 1, 5, 5, 5});
        const auto filters = make_shared<op::Parameter>(element::u1, PartialShape{1, 1, 3, 3});
        const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                     filters,
                                                                     Strides{},
                                                                     CoordinateDiff{},
                                                                     CoordinateDiff{},
                                                                     Strides{},
                                                                     mode,
                                                                     pad_value,
                                                                     auto_pad);
        // data batch and filters have incompatible ranks
        FAIL() << "Incompatible input ranks not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Data batch and filters rank do not match");
    } catch (...) {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }
}

TEST(type_prop, bin_convolution_invalid_spatial_dims_parameters) {
    Strides strides_1d{1};
    Strides strides_3d{1, 1, 1};

    Strides dilations_2d{1, 1};
    Strides dilations_3d{1, 1, 1};

    CoordinateDiff pads_end_2d{0, 0};
    CoordinateDiff pads_begin_3d{0, 0, 0};

    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::EXPLICIT;

    try {
        const auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape{1, 1, 5, 5});
        const auto filters = make_shared<op::Parameter>(element::u1, PartialShape{1, 1, 3, 3});
        const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                     filters,
                                                                     strides_3d,
                                                                     CoordinateDiff{},
                                                                     CoordinateDiff{},
                                                                     dilations_2d,
                                                                     mode,
                                                                     pad_value,
                                                                     auto_pad);
        // Strides have incompatible number of spatial dimensions
        FAIL() << "Incompatible stride number of spatial dimensions not detected.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Strides should be defined for all and only spatial dimensions."));
    } catch (...) {
        FAIL() << "Strides validation check failed for unexpected reason.";
    }

    try {
        const auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape{1, 1, 5, 5});
        const auto filters = make_shared<op::Parameter>(element::u1, PartialShape{1, 1, 3, 3});
        const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                     filters,
                                                                     Strides{1, 1},
                                                                     CoordinateDiff{},
                                                                     CoordinateDiff{},
                                                                     dilations_3d,
                                                                     mode,
                                                                     pad_value,
                                                                     auto_pad);
        // Dilations have incompatible number of spatial dimensions
        FAIL() << "Incompatible dilations number of spatial dimensions not detected.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Dilations should be defined for all and only spatial dimensions."));
    } catch (...) {
        FAIL() << "Dilations validation check failed for unexpected reason.";
    }

    try {
        const auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape{1, 1, 5, 5});
        const auto filters = make_shared<op::Parameter>(element::u1, PartialShape{1, 1, 3, 3});
        const auto bin_conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                                     filters,
                                                                     Strides{1, 1},
                                                                     pads_begin_3d,
                                                                     pads_end_2d,
                                                                     dilations_2d,
                                                                     mode,
                                                                     pad_value,
                                                                     auto_pad);
        // Pads have incompatible number of spatial dimensions
        FAIL() << "Incompatible pads number of spatial dimensions not detected.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Pads begin and end should be defined for all and only spatial dimensions."));
    } catch (...) {
        FAIL() << "Pads validation check failed for unexpected reason.";
    }
}

class TypePropBinaryConvolutionV1Test : public TypePropOpTest<op::v1::BinaryConvolution> {
protected:
    CoordinateDiff empty_pad{};
};

TEST_F(TypePropBinaryConvolutionV1Test, default_ctor) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 5, 5});
    const auto filters = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 4, 4});

    const auto op = make_op();
    op->set_arguments(OutputVector{data, filters});
    op->set_strides({1, 3});
    op->set_dilations({1, 2});
    op->set_pads_begin({2, 2});
    op->set_pads_end({2, 2});
    op->set_auto_pad(op::PadType::EXPLICIT);
    op->set_mode(op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT);
    op->set_pad_value(1.0f);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_strides(), Strides({1, 3}));
    EXPECT_EQ(op->get_dilations(), Strides({1, 2}));
    EXPECT_EQ(op->get_pads_begin(), CoordinateDiff({2, 2}));
    EXPECT_EQ(op->get_pads_end(), CoordinateDiff({2, 2}));
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({1, 2, 6, 1}));
}

TEST_F(TypePropBinaryConvolutionV1Test, interval_shapes) {
    PartialShape data_batch_pshape{{1, 3}, 1, {1, 5}, {3, 10}};
    PartialShape filters_pshape{2, {1, 3}, 3, 3};
    set_shape_labels(data_batch_pshape, 10);
    set_shape_labels(filters_pshape, 20);

    constexpr auto et = element::f32;
    constexpr auto auto_pad = op::PadType::EXPLICIT;
    constexpr auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    constexpr auto pad_value = 1.0f;

    const auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    const auto filters = make_shared<op::Parameter>(et, filters_pshape);
    const auto op = make_op(data_batch,
                            filters,
                            Strides{},
                            CoordinateDiff{},
                            CoordinateDiff{},
                            Strides{},
                            mode,
                            pad_value,
                            auto_pad);

    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), ElementsAre(10, 20, ov::no_label, ov::no_label));
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{1, 3}, 2, {1, 3}, {1, 8}}));
    EXPECT_EQ(op->get_pads_begin(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(op->get_pads_end(), (CoordinateDiff{0, 0}));
}
