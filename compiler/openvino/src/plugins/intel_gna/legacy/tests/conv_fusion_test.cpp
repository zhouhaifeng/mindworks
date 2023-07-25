// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <legacy/transformations/convert_opset1_to_legacy/conv_bias_fusion.hpp>
#include <map>
#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <queue>
#include <sstream>
#include <string>
#include <transformations/common_optimizations/conv_mul_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/test_common.hpp"

using namespace ngraph;
using namespace testing;

using InputShape = ngraph::PartialShape;
using WeightsShape = ngraph::Shape;
using EltwiseType = ngraph::NodeTypeInfo;
using EltwiseShape = ngraph::Shape;
using IsNegative = bool;

class ConvFusionTests
    : public ov::test::TestsCommon,
      public testing::WithParamInterface<std::tuple<InputShape, WeightsShape, EltwiseType, EltwiseShape, IsNegative>> {
public:
    std::shared_ptr<ngraph::Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& weights_shape = std::get<1>(GetParam());
        const auto& eltwise_type = std::get<2>(GetParam());
        const auto& eltwise_shape = std::get<3>(GetParam());
        const auto& is_negative = std::get<4>(GetParam());

        f = get_initial_function(input_shape, weights_shape, eltwise_type, eltwise_shape);

        if (is_negative) {
            f_ref = get_initial_function(input_shape, weights_shape, eltwise_type, eltwise_shape);
        } else {
            f_ref = get_reference_function(input_shape, weights_shape, eltwise_type, eltwise_shape);
            ngraph::pass::Manager manager;
            manager.register_pass<ngraph::pass::ConstantFolding>();
            manager.run_passes(f_ref);
        }
    }

private:
    std::shared_ptr<ngraph::Function> get_initial_function(const InputShape& input_shape,
                                                           const WeightsShape& weights_shape,
                                                           const EltwiseType& eltwise_type,
                                                           const EltwiseShape& eltwise_shape) {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, input_shape);
        auto weights = ngraph::opset5::Constant::create(ngraph::element::f32, weights_shape, {1});
        auto conv = std::make_shared<ngraph::op::ConvolutionIE>(input,
                                                                weights,
                                                                ngraph::Strides(spatial_dims, 1),
                                                                ngraph::Strides(spatial_dims, 1),
                                                                ngraph::CoordinateDiff(spatial_dims, 0),
                                                                ngraph::CoordinateDiff(spatial_dims, 0),
                                                                ngraph::element::f32);

        auto const_node = ngraph::opset5::Constant::create(ngraph::element::f32, eltwise_shape, {1.1});
        ngraph::Output<ngraph::Node> eltwise;
        if (eltwise_type == ngraph::opset5::Add::get_type_info_static()) {
            eltwise = std::make_shared<ngraph::opset5::Add>(conv, const_node);
        } else if (eltwise_type == ngraph::opset5::Multiply::get_type_info_static()) {
            eltwise = std::make_shared<ngraph::opset5::Multiply>(conv, const_node);
        } else {
            OPENVINO_THROW("Unsupported element type");
        }

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{eltwise.get_node_shared_ptr()},
                                                  ngraph::ParameterVector{input});
    }

    std::shared_ptr<ngraph::Function> get_reference_function(const InputShape& input_shape,
                                                             const WeightsShape& weights_shape,
                                                             const EltwiseType& eltwise_type,
                                                             const EltwiseShape& eltwise_shape) {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, input_shape);
        ngraph::Output<ngraph::Node> weights =
            ngraph::opset5::Constant::create(ngraph::element::f32, weights_shape, {1});
        ngraph::Output<ngraph::Node> conv =
            std::make_shared<ngraph::op::ConvolutionIE>(input,
                                                        weights,
                                                        ngraph::Strides(spatial_dims, 1),
                                                        ngraph::Strides(spatial_dims, 1),
                                                        ngraph::CoordinateDiff(spatial_dims, 0),
                                                        ngraph::CoordinateDiff(spatial_dims, 0),
                                                        ngraph::element::f32);

        ngraph::Output<ngraph::Node> const_node;
        const_node = ngraph::opset5::Constant::create(ngraph::element::f32, eltwise_shape, {1.1});
        if (eltwise_type == ngraph::opset5::Add::get_type_info_static()) {
            if (eltwise_shape.size() != 1) {
                const_node = ov::op::util::reshapeTo(const_node, ngraph::Shape{ngraph::shape_size(eltwise_shape)});
            }
            conv = conv.get_node_shared_ptr()->copy_with_new_inputs({input, weights, const_node});
        } else if (eltwise_type == ngraph::opset5::Multiply::get_type_info_static()) {
            if (eltwise_shape.size() > 1) {
                const_node = ov::op::util::reshapeTo(const_node, ngraph::Shape{ngraph::shape_size(eltwise_shape)});
            }
            ngraph::Shape const_shape(weights_shape.size(), 1);
            const_shape[0] = weights_shape[0];
            weights =
                std::make_shared<ngraph::opset5::Multiply>(weights, ov::op::util::reshapeTo(const_node, const_shape));
            conv = conv.get_node_shared_ptr()->copy_with_new_inputs({input, weights});
        } else {
            OPENVINO_THROW("Unsupported element type");
        }

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{conv.get_node_shared_ptr()},
                                                  ngraph::ParameterVector{input});
    }
};

TEST_P(ConvFusionTests, CompareFunctions) {
    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    ngraph::pass::Manager manager;
    manager.register_pass<ov::pass::InitUniqueNames>(unh);
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::ConvFusion>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ov::pass::CheckUniqueNames>(unh);
    manager.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    auto fc =
        FunctionsComparator::no_default().enable(FunctionsComparator::NODES).enable(FunctionsComparator::PRECISIONS);
    auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

using add = ngraph::opset5::Add;
using mul = ngraph::opset5::Multiply;

INSTANTIATE_TEST_SUITE_P(ConvAddFusion,
                         ConvFusionTests,
                         testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN},
                                                         WeightsShape{8, 3, 1, 2, 3},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{8, 1, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64, 64},
                                                         WeightsShape{8, 3, 1, 2, 3},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{8, 1, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{2, DYN, 64, 64, 64},
                                                         WeightsShape{9, 3, 2, 3, 1},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{9, 1, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{3, 3, DYN, 64, 64},
                                                         WeightsShape{6, 3, 3, 4, 2},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{6, 1, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{3, 3, 64, DYN, 64},
                                                         WeightsShape{5, 3, 3, 4, 3},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{5, 1, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{3, 3, 64, 64, DYN},
                                                         WeightsShape{5, 3, 3, 4, 3},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{5, 1, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{1, 3, 64, 64},
                                                         WeightsShape{6, 3, 1, 1},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{6, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{DYN, DYN, DYN, DYN},
                                                         WeightsShape{7, 3, 1, 1},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{7, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64},
                                                         WeightsShape{8, 3, 1, 2},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{8, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{2, DYN, 64, 64},
                                                         WeightsShape{9, 3, 2, 3},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{9, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{3, 3, DYN, 64},
                                                         WeightsShape{6, 3, 3, 4},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{6, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{3, 3, 64, DYN},
                                                         WeightsShape{5, 3, 3, 4},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{5, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{DYN, DYN, DYN},
                                                         WeightsShape{5, 3, 1},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{5, 1},
                                                         false),
                                         std::make_tuple(InputShape{DYN, 3, 10},
                                                         WeightsShape{3, 3, 1},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{3, 1},
                                                         false),
                                         std::make_tuple(InputShape{2, DYN, 9},
                                                         WeightsShape{2, 3, 2},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{2, 1},
                                                         false),
                                         std::make_tuple(InputShape{3, 3, DYN},
                                                         WeightsShape{1, 3, 3},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{1, 1},
                                                         false)));

INSTANTIATE_TEST_SUITE_P(ConvAddFusionNegative,
                         ConvFusionTests,
                         testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN},
                                                         WeightsShape{8, 3, 1, 2, 3},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{2, 1},
                                                         true),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64, 64},
                                                         WeightsShape{8, 3, 1, 2, 3},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{8, 1, 1, 1, 1},
                                                         true),
                                         std::make_tuple(InputShape{2, DYN, 64, 64, 64},
                                                         WeightsShape{9, 3, 2, 3, 1},
                                                         add::get_type_info_static(),
                                                         EltwiseShape{2, 1, 1, 1, 1},
                                                         true)));

INSTANTIATE_TEST_SUITE_P(ConvMulFusion,
                         ConvFusionTests,
                         testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN},
                                                         WeightsShape{8, 3, 1, 2, 3},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{8, 1, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64, 64},
                                                         WeightsShape{8, 3, 1, 2, 3},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{8, 1, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{2, DYN, 64, 64, 64},
                                                         WeightsShape{9, 3, 2, 3, 1},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{9, 1, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{3, 3, DYN, 64, 64},
                                                         WeightsShape{6, 3, 3, 4, 2},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{6, 1, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{3, 3, 64, DYN, 64},
                                                         WeightsShape{5, 3, 3, 4, 3},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{5, 1, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{3, 3, 64, 64, DYN},
                                                         WeightsShape{5, 3, 3, 4, 3},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{5, 1, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{1, 3, 64, 64},
                                                         WeightsShape{6, 3, 1, 1},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{6, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{DYN, DYN, DYN, DYN},
                                                         WeightsShape{7, 3, 1, 1},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{7, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64},
                                                         WeightsShape{8, 3, 1, 2},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{8, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{2, DYN, 64, 64},
                                                         WeightsShape{9, 3, 2, 3},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{9, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{3, 3, DYN, 64},
                                                         WeightsShape{6, 3, 3, 4},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{6, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{3, 3, 64, DYN},
                                                         WeightsShape{5, 3, 3, 4},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{5, 1, 1},
                                                         false),
                                         std::make_tuple(InputShape{DYN, DYN, DYN},
                                                         WeightsShape{5, 3, 1},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{5, 1},
                                                         false),
                                         std::make_tuple(InputShape{DYN, 3, 10},
                                                         WeightsShape{3, 3, 1},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{3, 1},
                                                         false),
                                         std::make_tuple(InputShape{2, DYN, 9},
                                                         WeightsShape{2, 3, 2},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{2, 1},
                                                         false),
                                         std::make_tuple(InputShape{3, 3, DYN},
                                                         WeightsShape{1, 3, 3},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{1, 1},
                                                         false)));

INSTANTIATE_TEST_SUITE_P(ConvMulFusionNegative,
                         ConvFusionTests,
                         testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN},
                                                         WeightsShape{8, 3, 1, 2, 3},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{2, 1},
                                                         true),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64},
                                                         WeightsShape{3, 3, 2, 3},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{8, 1, 1, 1},
                                                         true),
                                         std::make_tuple(InputShape{2, DYN, 64, 64},
                                                         WeightsShape{3, 2, 3, 1},
                                                         mul::get_type_info_static(),
                                                         EltwiseShape{9, 1, 1, 1, 1},
                                                         true)));

TEST_F(TransformationTestsF, WeightsWithReshape) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 4, 7, 7});
        auto weights = opset5::Constant::create(element::f32, Shape{4, 1, 3, 3}, std::vector<float>(36, 1));
        auto reshape =
            std::make_shared<opset5::Reshape>(weights,
                                              opset5::Constant::create(element::i64, Shape{5}, Shape{4, 1, 1, 3, 3}),
                                              false);
        auto conv = std::make_shared<opset5::GroupConvolution>(data,
                                                               reshape,
                                                               Strides{1, 1},
                                                               CoordinateDiff{0, 0},
                                                               CoordinateDiff{0, 0},
                                                               Strides{1, 1});
        auto mul = std::make_shared<opset5::Multiply>(
            conv,
            opset5::Constant::create(element::f32, Shape{4, 1, 1}, std::vector<float>(4, 2)));
        function = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::GroupConvolutionMultiplyFusion>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 4, 7, 7});
        auto weights = opset5::Constant::create(element::f32, Shape{4, 1, 1, 3, 3}, std::vector<float>(36, 2));
        auto conv = std::make_shared<opset5::GroupConvolution>(data,
                                                               weights,
                                                               Strides{1, 1},
                                                               CoordinateDiff{0, 0},
                                                               CoordinateDiff{0, 0},
                                                               Strides{1, 1});
        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeWeightsWithReshape) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 4, 7, 7});
        auto weights = opset5::Constant::create(element::f32, Shape{36}, std::vector<float>(36, 1));
        auto reshape =
            std::make_shared<opset5::Reshape>(weights,
                                              opset5::Constant::create(element::i64, Shape{5}, Shape{4, 1, 1, 3, 3}),
                                              false);
        auto conv = std::make_shared<opset5::GroupConvolution>(data,
                                                               reshape,
                                                               Strides{1, 1},
                                                               CoordinateDiff{0, 0},
                                                               CoordinateDiff{0, 0},
                                                               Strides{1, 1});
        auto mul = std::make_shared<opset5::Multiply>(
            conv,
            opset5::Constant::create(element::f32, Shape{4, 1, 1}, std::vector<float>(4, 2)));
        function = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::GroupConvolutionMultiplyFusion>();
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 4, 7, 7});
        auto weights = opset5::Constant::create(element::f32, Shape{36}, std::vector<float>(36, 1));
        auto reshape =
            std::make_shared<opset5::Reshape>(weights,
                                              opset5::Constant::create(element::i64, Shape{5}, Shape{4, 1, 1, 3, 3}),
                                              false);
        auto conv = std::make_shared<opset5::GroupConvolution>(data,
                                                               reshape,
                                                               Strides{1, 1},
                                                               CoordinateDiff{0, 0},
                                                               CoordinateDiff{0, 0},
                                                               Strides{1, 1});
        auto mul = std::make_shared<opset5::Multiply>(
            conv,
            opset5::Constant::create(element::f32, Shape{4, 1, 1}, std::vector<float>(4, 2)));
        function_ref = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, WeightsWithReshapeScalarMultiplier) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 4, 7, 7});
        auto weights = opset5::Constant::create(element::f32, Shape{36}, std::vector<float>(36, 1));
        auto reshape =
            std::make_shared<opset5::Reshape>(weights,
                                              opset5::Constant::create(element::i64, Shape{5}, Shape{4, 1, 1, 3, 3}),
                                              false);
        auto conv = std::make_shared<opset5::GroupConvolution>(data,
                                                               reshape,
                                                               Strides{1, 1},
                                                               CoordinateDiff{0, 0},
                                                               CoordinateDiff{0, 0},
                                                               Strides{1, 1});
        auto mul = std::make_shared<opset5::Multiply>(conv, opset5::Constant::create(element::f32, Shape{1}, {2.0f}));
        function = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::GroupConvolutionMultiplyFusion>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 4, 7, 7});
        auto weights = opset5::Constant::create(element::f32, Shape{4, 1, 1, 3, 3}, std::vector<float>(36, 2));
        auto conv = std::make_shared<opset5::GroupConvolution>(data,
                                                               weights,
                                                               Strides{1, 1},
                                                               CoordinateDiff{0, 0},
                                                               CoordinateDiff{0, 0},
                                                               Strides{1, 1});
        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, WeightsWithoutReshape) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 4, 7, 7});
        auto weights = opset5::Constant::create(element::f32, Shape{4, 1, 1, 3, 3}, std::vector<float>(36, 1));
        auto conv = std::make_shared<opset5::GroupConvolution>(data,
                                                               weights,
                                                               Strides{1, 1},
                                                               CoordinateDiff{0, 0},
                                                               CoordinateDiff{0, 0},
                                                               Strides{1, 1});
        auto mul = std::make_shared<opset5::Multiply>(
            conv,
            opset5::Constant::create(element::f32, Shape{4, 1, 1}, std::vector<float>(4, 2)));
        function = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::GroupConvolutionMultiplyFusion>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 4, 7, 7});
        auto weights = opset5::Constant::create(element::f32, Shape{4, 1, 1, 3, 3}, std::vector<float>(36, 2));
        auto conv = std::make_shared<opset5::GroupConvolution>(data,
                                                               weights,
                                                               Strides{1, 1},
                                                               CoordinateDiff{0, 0},
                                                               CoordinateDiff{0, 0},
                                                               Strides{1, 1});
        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, WeightsWithFakeQuantizeAndReshape) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 4, 7, 7});
        auto weights = opset5::Constant::create(element::f32, Shape{6, 2, 3, 3}, std::vector<float>(108, 1));
        auto fq = std::make_shared<opset5::FakeQuantize>(weights,
                                                         opset5::Constant::create(element::f32, Shape{1}, {0}),
                                                         opset5::Constant::create(element::f32, Shape{1}, {1}),
                                                         opset5::Constant::create(element::f32, Shape{1}, {0}),
                                                         opset5::Constant::create(element::f32, Shape{1}, {10}),
                                                         2);
        auto reshape =
            std::make_shared<opset5::Reshape>(fq,
                                              opset5::Constant::create(element::i64, Shape{5}, Shape{2, 3, 2, 3, 3}),
                                              false);
        auto conv = std::make_shared<opset5::GroupConvolution>(data,
                                                               reshape,
                                                               Strides{1, 1},
                                                               CoordinateDiff{0, 0},
                                                               CoordinateDiff{0, 0},
                                                               Strides{1, 1});
        auto mul = std::make_shared<opset5::Multiply>(
            conv,
            opset5::Constant::create(element::f32, Shape{6, 1, 1}, std::vector<float>(6, 2)));
        function = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::GroupConvolutionMultiplyFusion>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 4, 7, 7});
        auto weights = opset5::Constant::create(element::f32, Shape{6, 2, 3, 3}, std::vector<float>(108, 1));
        auto fq = std::make_shared<opset5::FakeQuantize>(weights,
                                                         opset5::Constant::create(element::f32, Shape{1}, {0}),
                                                         opset5::Constant::create(element::f32, Shape{1}, {1}),
                                                         opset5::Constant::create(element::f32, Shape{1}, {0}),
                                                         opset5::Constant::create(element::f32, Shape{1}, {10}),
                                                         2);
        auto mul = std::make_shared<opset5::Multiply>(
            fq,
            opset5::Constant::create(element::f32, Shape{6, 1, 1, 1}, std::vector<float>(6, 2)));
        auto reshape =
            std::make_shared<opset5::Reshape>(mul,
                                              opset5::Constant::create(element::i64, Shape{5}, Shape{2, 3, 2, 3, 3}),
                                              false);
        auto conv = std::make_shared<opset5::GroupConvolution>(data,
                                                               reshape,
                                                               Strides{1, 1},
                                                               CoordinateDiff{0, 0},
                                                               CoordinateDiff{0, 0},
                                                               Strides{1, 1});
        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});
    }
}
