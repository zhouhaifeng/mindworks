// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/op/depth_to_space.hpp>
#include <ngraph/op/space_to_depth.hpp>
#include <ngraph/pass/manager.hpp>
#include <sstream>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/convert_depth_to_space.hpp>
#include <transformations/op_conversions/convert_space_to_depth.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/test_common.hpp"

using namespace testing;

TEST(TransformationTests, TestDepthToSpaceTransformBlockFirst) {
    auto input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1, 12, 1080, 1616});
    std::shared_ptr<ngraph::Function> f(nullptr);

    {
        auto depth_to_space =
            std::make_shared<ngraph::op::DepthToSpace>(input,
                                                       ngraph::op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
                                                       2);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{depth_to_space}, ngraph::ParameterVector{input});
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertDepthToSpace>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    auto consumers = input->output(0).get_target_inputs();
    ASSERT_EQ(consumers.size(), 1);

    auto reshape_begin = consumers.begin()->get_node();
    auto shape_begin = std::dynamic_pointer_cast<ov::op::v0::Constant>(
        reshape_begin->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_begin_value = shape_begin->get_vector<int64_t>();
    std::vector<int64_t> shape_begin_value_ref{1, 2, 2, 3, 1080, 1616};
    ASSERT_EQ(shape_begin_value, shape_begin_value_ref);

    consumers = reshape_begin->output(0).get_target_inputs();
    ASSERT_EQ(consumers.size(), 1);

    auto transpose = consumers.begin()->get_node();
    auto order =
        std::dynamic_pointer_cast<ov::op::v0::Constant>(transpose->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> order_value = order->get_vector<int64_t>();
    std::vector<int64_t> order_value_ref{0, 3, 4, 1, 5, 2};
    ASSERT_EQ(order_value, order_value_ref);

    consumers = transpose->output(0).get_target_inputs();
    auto reshape_end = consumers.begin()->get_node();
    auto shape_end = std::dynamic_pointer_cast<ov::op::v0::Constant>(
        reshape_end->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_end_value = shape_end->get_vector<int64_t>();
    std::vector<int64_t> shape_end_value_ref{1, 3, 2 * 1080, 2 * 1616};
    ASSERT_EQ(shape_end_value, shape_end_value_ref);
}

TEST(TransformationTests, TestDepthToSpaceTransformDepthFirst) {
    auto input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1, 12, 1080, 1616});
    std::shared_ptr<ngraph::Function> f(nullptr);

    {
        auto depth_to_space =
            std::make_shared<ngraph::op::DepthToSpace>(input,
                                                       ngraph::op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST,
                                                       2);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{depth_to_space}, ngraph::ParameterVector{input});
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertDepthToSpace>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    auto consumers = input->output(0).get_target_inputs();
    ASSERT_EQ(consumers.size(), 1);

    auto reshape_begin = consumers.begin()->get_node();
    auto shape_begin = std::dynamic_pointer_cast<ov::op::v0::Constant>(
        reshape_begin->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_begin_value = shape_begin->get_vector<int64_t>();
    std::vector<int64_t> shape_begin_value_ref{1, 3, 2, 2, 1080, 1616};
    ASSERT_EQ(shape_begin_value, shape_begin_value_ref);

    consumers = reshape_begin->output(0).get_target_inputs();
    ASSERT_EQ(consumers.size(), 1);

    auto transpose = consumers.begin()->get_node();
    auto order =
        std::dynamic_pointer_cast<ov::op::v0::Constant>(transpose->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> order_value = order->get_vector<int64_t>();
    std::vector<int64_t> order_value_ref{0, 1, 4, 2, 5, 3};
    ASSERT_EQ(order_value, order_value_ref);

    consumers = transpose->output(0).get_target_inputs();
    auto reshape_end = consumers.begin()->get_node();
    auto shape_end = std::dynamic_pointer_cast<ov::op::v0::Constant>(
        reshape_end->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_end_value = shape_end->get_vector<int64_t>();
    std::vector<int64_t> shape_end_value_ref{1, 3, 2 * 1080, 2 * 1616};
    ASSERT_EQ(shape_end_value, shape_end_value_ref);
}

TEST(TransformationTests, TestSpaceToDepthTransformBlockFirst) {
    auto input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1, 12, 1080, 1616});
    std::shared_ptr<ngraph::Function> f(nullptr);

    {
        auto space_to_depth =
            std::make_shared<ngraph::op::SpaceToDepth>(input,
                                                       ngraph::op::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
                                                       2);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{space_to_depth}, ngraph::ParameterVector{input});
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertSpaceToDepth>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    auto consumers = input->output(0).get_target_inputs();
    ASSERT_EQ(consumers.size(), 1);

    auto reshape_begin = consumers.begin()->get_node();
    auto shape_begin = std::dynamic_pointer_cast<ov::op::v0::Constant>(
        reshape_begin->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_begin_value = shape_begin->get_vector<int64_t>();
    std::vector<int64_t> shape_begin_value_ref{1, 12, 1080 / 2, 2, 1616 / 2, 2};
    ASSERT_EQ(shape_begin_value, shape_begin_value_ref);

    consumers = reshape_begin->output(0).get_target_inputs();
    ASSERT_EQ(consumers.size(), 1);

    auto transpose = consumers.begin()->get_node();
    auto order =
        std::dynamic_pointer_cast<ov::op::v0::Constant>(transpose->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> order_value = order->get_vector<int64_t>();
    std::vector<int64_t> order_value_ref{0, 3, 5, 1, 2, 4};
    ASSERT_EQ(order_value, order_value_ref);

    consumers = transpose->output(0).get_target_inputs();
    auto reshape_end = consumers.begin()->get_node();
    auto shape_end = std::dynamic_pointer_cast<ov::op::v0::Constant>(
        reshape_end->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_end_value = shape_end->get_vector<int64_t>();
    std::vector<int64_t> shape_end_value_ref{1, 12 * 4, 1080 / 2, 1616 / 2};
    ASSERT_EQ(shape_end_value, shape_end_value_ref);
}

TEST(TransformationTests, TestSpaceToDepthTransformDepthFirst) {
    auto input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1, 12, 1080, 1616});
    std::shared_ptr<ngraph::Function> f(nullptr);

    {
        auto space_to_depth =
            std::make_shared<ngraph::op::SpaceToDepth>(input,
                                                       ngraph::op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST,
                                                       2);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{space_to_depth}, ngraph::ParameterVector{input});
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertSpaceToDepth>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    auto consumers = input->output(0).get_target_inputs();
    ASSERT_EQ(consumers.size(), 1);

    auto reshape_begin = consumers.begin()->get_node();
    auto shape_begin = std::dynamic_pointer_cast<ov::op::v0::Constant>(
        reshape_begin->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_begin_value = shape_begin->get_vector<int64_t>();
    std::vector<int64_t> shape_begin_value_ref{1, 12, 1080 / 2, 2, 1616 / 2, 2};
    ASSERT_EQ(shape_begin_value, shape_begin_value_ref);

    consumers = reshape_begin->output(0).get_target_inputs();
    ASSERT_EQ(consumers.size(), 1);

    auto transpose = consumers.begin()->get_node();
    auto order =
        std::dynamic_pointer_cast<ov::op::v0::Constant>(transpose->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> order_value = order->get_vector<int64_t>();
    std::vector<int64_t> order_value_ref{0, 1, 3, 5, 2, 4};
    ASSERT_EQ(order_value, order_value_ref);

    consumers = transpose->output(0).get_target_inputs();
    auto reshape_end = consumers.begin()->get_node();
    auto shape_end = std::dynamic_pointer_cast<ov::op::v0::Constant>(
        reshape_end->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_end_value = shape_end->get_vector<int64_t>();
    std::vector<int64_t> shape_end_value_ref{1, 12 * 4, 1080 / 2, 1616 / 2};
    ASSERT_EQ(shape_end_value, shape_end_value_ref);
}

TEST(TransformationTests, TestSpaceToDepthDynamic) {
    auto input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    std::shared_ptr<ngraph::Function> f(nullptr);

    {
        auto space_to_depth =
            std::make_shared<ngraph::op::SpaceToDepth>(input,
                                                       ngraph::op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST,
                                                       2);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{space_to_depth}, ngraph::ParameterVector{input});
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::ConvertSpaceToDepth>();
        ASSERT_NO_THROW(m.run_passes(f));
    }
}

TEST(TransformationTests, TestDepthToSpaceDynamic) {
    auto input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    std::shared_ptr<ngraph::Function> f(nullptr);

    {
        auto depth_to_space =
            std::make_shared<ngraph::op::DepthToSpace>(input,
                                                       ngraph::op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
                                                       2);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{depth_to_space}, ngraph::ParameterVector{input});
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::ConvertDepthToSpace>();
        ASSERT_NO_THROW(m.run_passes(f));
    }
}
