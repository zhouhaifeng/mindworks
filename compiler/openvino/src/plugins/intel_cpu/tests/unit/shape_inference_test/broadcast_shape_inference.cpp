// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, BroadcastBidirectionalTest) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto broadcast_v3 = std::make_shared<op::v3::Broadcast>(input, target_shape, op::BroadcastType::BIDIRECTIONAL);

    int32_t target_shape_val[] = {1, 16, 50, 1};
    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] =
        std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, ov::Shape{4}, target_shape_val);

    std::vector<StaticShape> static_input_shapes = {StaticShape{16, 1, 8}, StaticShape{4}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(broadcast_v3.get(), static_input_shapes, static_output_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 16, 50, 8}));

    static_input_shapes = {StaticShape{16, 1, 1}, StaticShape{4}};
    static_output_shapes = {StaticShape{}};
    EXPECT_THROW(shape_inference(broadcast_v3.get(), static_input_shapes, static_output_shapes, {}), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, BroadcastBidirectionalConstantTest) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto target_shape = std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{3}, std::vector<int32_t>{16, 1, 40});
    auto broadcast_v3 = std::make_shared<op::v3::Broadcast>(input, target_shape, op::BroadcastType::BIDIRECTIONAL);

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 16, 50, 1}, StaticShape{3}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(broadcast_v3.get(), static_input_shapes, static_output_shapes, {});
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 16, 50, 40}));
}

TEST(StaticShapeInferenceTest, BroadcastPDPDTest) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto broadcast_v3 =
        std::make_shared<op::v3::Broadcast>(input, target_shape, op::BroadcastModeSpec(op::BroadcastType::PDPD, 1));

    int32_t target_shape_val[] = {2, 3, 6};
    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] =
        std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, ov::Shape{3}, target_shape_val);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 1}, StaticShape{3}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(broadcast_v3.get(), static_input_shapes, static_output_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 3, 6}));

    static_input_shapes = {StaticShape{3, 1}, StaticShape{3}};
    static_output_shapes = {StaticShape{}};
    EXPECT_THROW(shape_inference(broadcast_v3.get(), static_input_shapes, static_output_shapes, {}), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, BroadcastPDPDConstantTest) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto target_shape = std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{3}, std::vector<int32_t>{2, 3, 6});
    auto broadcast_v3 =
        std::make_shared<op::v3::Broadcast>(input, target_shape, op::BroadcastModeSpec(op::BroadcastType::PDPD, 1));

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 1}, StaticShape{3}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(broadcast_v3.get(), static_input_shapes, static_output_shapes, {});
    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 3, 6}));
}

TEST(StaticShapeInferenceTest, BroadcastNumpyTest) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto broadcast_v3 = std::make_shared<op::v3::Broadcast>(input, target_shape, op::BroadcastType::NUMPY);

    int32_t target_shape_val[] = {1, 16, 50, 50};
    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] =
        std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, ov::Shape{4}, target_shape_val);

    std::vector<StaticShape> static_input_shapes = {StaticShape{16, 1, 1}, StaticShape{4}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(broadcast_v3.get(), static_input_shapes, static_output_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 16, 50, 50}));

    static_input_shapes = {StaticShape{16, 1, 1}, StaticShape{4}};
    static_output_shapes = {StaticShape{}};
    EXPECT_THROW(shape_inference(broadcast_v3.get(), static_input_shapes, static_output_shapes, {}), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, BroadcastNumpyConstantTest) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto target_shape =
        std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{4}, std::vector<int32_t>{1, 16, 50, 50});
    auto broadcast_v3 = std::make_shared<op::v3::Broadcast>(input, target_shape, op::BroadcastType::NUMPY);

    std::vector<StaticShape> static_input_shapes = {StaticShape{16, 1, 1}, StaticShape{4}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(broadcast_v3.get(), static_input_shapes, static_output_shapes, {});
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 16, 50, 50}));
}

TEST(StaticShapeInferenceTest, BroadcastExplicitTest) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1});
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto axes_mapping = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto broadcast_v3 =
        std::make_shared<op::v3::Broadcast>(input, target_shape, axes_mapping, op::BroadcastType::EXPLICIT);

    int32_t target_shape_val[] = {1, 16, 50, 50};
    int32_t axes_mapping_val[] = {1};
    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] =
        std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, ov::Shape{4}, target_shape_val);
    constant_data[2] =
        std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, ov::Shape{1}, axes_mapping_val);

    std::vector<StaticShape> static_input_shapes = {StaticShape{16}, StaticShape{4}, StaticShape{1}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(broadcast_v3.get(), static_input_shapes, static_output_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 16, 50, 50}));

    constant_data.erase(1);
    EXPECT_THROW(shape_inference(broadcast_v3.get(), static_input_shapes, static_output_shapes, constant_data),
                 NodeValidationFailure);
    EXPECT_THROW(shape_inference(broadcast_v3.get(), static_input_shapes, static_output_shapes, {}), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, BroadcastExplicitConstantTest) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1});
    auto target_shape =
        std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{4}, std::vector<int32_t>{1, 16, 50, 50});
    auto axes_mapping = std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{1}, std::vector<int32_t>{1});
    auto broadcast_v3 =
        std::make_shared<op::v3::Broadcast>(input, target_shape, axes_mapping, op::BroadcastType::EXPLICIT);

    std::vector<StaticShape> static_input_shapes = {StaticShape{16}, StaticShape{4}, StaticShape{1}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(broadcast_v3.get(), static_input_shapes, static_output_shapes, {});
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 16, 50, 50}));
}

// BroadcastV1 test

TEST(StaticShapeInferenceTest, BroadcastV1PDPDTest) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto broadcast_v1 =
        std::make_shared<op::v1::Broadcast>(input, target_shape, op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1));

    int32_t target_shape_val[] = {2, 3, 6};
    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] =
        std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, ov::Shape{3}, target_shape_val);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 1}, StaticShape{3}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(broadcast_v1.get(), static_input_shapes, static_output_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 3, 6}));

    static_input_shapes = {StaticShape{3, 1}, StaticShape{3}};
    static_output_shapes = {StaticShape{}};
    EXPECT_THROW(shape_inference(broadcast_v1.get(), static_input_shapes, static_output_shapes, {}), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, BroadcastV1NumpyTest) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto broadcast_v1 = std::make_shared<op::v1::Broadcast>(input, target_shape);

    int32_t target_shape_val[] = {2, 3, 6};
    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] =
        std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, ov::Shape{3}, target_shape_val);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 1}, StaticShape{3}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(broadcast_v1.get(), static_input_shapes, static_output_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 3, 6}));

    static_input_shapes = {StaticShape{3, 1}, StaticShape{3}};
    static_output_shapes = {StaticShape{}};
    EXPECT_THROW(shape_inference(broadcast_v1.get(), static_input_shapes, static_output_shapes, {}), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, BroadcastV1ExplicitTest) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto axes_mapping = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto broadcast_v1 = std::make_shared<op::v1::Broadcast>(input, target_shape, axes_mapping);

    int32_t target_shape_val[] = {2, 3, 1};
    int32_t axes_mapping_val[] = {1, 2};
    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] =
        std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, ov::Shape{3}, target_shape_val);
    constant_data[2] =
        std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, ov::Shape{2}, axes_mapping_val);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 1}, StaticShape{3}, StaticShape{2}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(broadcast_v1.get(), static_input_shapes, static_output_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 3, 1}));

    static_input_shapes = {StaticShape{3, 1}, StaticShape{3}, StaticShape{2}};
    static_output_shapes = {StaticShape{}};
    EXPECT_THROW(shape_inference(broadcast_v1.get(), static_input_shapes, static_output_shapes, {}), NodeValidationFailure);
}
