// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <queue>
#include <string>
#include <transformations/common_optimizations/dilated_convolution_converter.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, DilatedConvolutionConverter) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto filters = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 3, 3});
        auto space_to_batch =
            std::make_shared<opset6::SpaceToBatch>(data,
                                                   op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::Constant::create(element::i64, Shape{4}, {0, 0, 2, 2}),
                                                   op::Constant::create(element::i64, Shape{4}, {0, 0, 2, 2}));
        auto conv = std::make_shared<opset6::Convolution>(space_to_batch,
                                                          filters,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        auto batch_to_space =
            std::make_shared<opset6::BatchToSpace>(conv,
                                                   op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}),
                                                   op::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}));
        function = std::make_shared<Function>(NodeVector{batch_to_space}, ParameterVector{data, filters});

        manager.register_pass<ov::pass::DilatedConvolutionConverter>();
    }
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto filters = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 3, 3});
        auto conv = std::make_shared<opset6::Convolution>(data,
                                                          filters,
                                                          Strides{1, 1},
                                                          CoordinateDiff{1, 1},
                                                          CoordinateDiff{1, 1},
                                                          Strides{2, 2},
                                                          op::PadType::EXPLICIT);
        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, NegativeDilatedConvolutionConverterPadsLessThanCrops) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto filters = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 3, 3});
        auto space_to_batch =
            std::make_shared<opset6::SpaceToBatch>(data,
                                                   op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}),
                                                   op::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}));
        auto conv = std::make_shared<opset6::Convolution>(space_to_batch,
                                                          filters,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        auto batch_to_space =
            std::make_shared<opset6::BatchToSpace>(conv,
                                                   op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0}),
                                                   op::Constant::create(element::i64, Shape{4}, {0, 0, 2, 3}));
        function = std::make_shared<Function>(NodeVector{batch_to_space}, ParameterVector{data, filters});

        manager.register_pass<ov::pass::DilatedConvolutionConverter>();
    }
}

TEST_F(TransformationTestsF, NegativeDilatedConvolutionConverterNonZeroPadsForNC) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto filters = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 5, 3, 3});
        auto space_to_batch =
            std::make_shared<opset6::SpaceToBatch>(data,
                                                   op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                                                   op::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}));
        auto conv = std::make_shared<opset6::Convolution>(space_to_batch,
                                                          filters,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        auto batch_to_space =
            std::make_shared<opset6::BatchToSpace>(conv,
                                                   op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0}),
                                                   op::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}));
        function = std::make_shared<Function>(NodeVector{batch_to_space}, ParameterVector{data, filters});

        manager.register_pass<ov::pass::DilatedConvolutionConverter>();
    }
}

TEST_F(TransformationTestsF, DilatedGroupConvolutionConverter) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto filters = std::make_shared<opset6::Parameter>(element::f32, Shape{4, 1, 1, 3, 3});
        auto space_to_batch =
            std::make_shared<opset6::SpaceToBatch>(data,
                                                   op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::Constant::create(element::i64, Shape{4}, {0, 0, 2, 2}),
                                                   op::Constant::create(element::i64, Shape{4}, {0, 0, 2, 2}));
        auto conv = std::make_shared<opset6::GroupConvolution>(space_to_batch,
                                                               filters,
                                                               Strides{1, 1},
                                                               CoordinateDiff{0, 0},
                                                               CoordinateDiff{0, 0},
                                                               Strides{1, 1});
        auto batch_to_space =
            std::make_shared<opset6::BatchToSpace>(conv,
                                                   op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}),
                                                   op::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}));
        function = std::make_shared<Function>(NodeVector{batch_to_space}, ParameterVector{data, filters});

        manager.register_pass<ov::pass::DilatedConvolutionConverter>();
    }
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto filters = std::make_shared<opset6::Parameter>(element::f32, Shape{4, 1, 1, 3, 3});
        auto conv = std::make_shared<opset6::GroupConvolution>(data,
                                                               filters,
                                                               Strides{1, 1},
                                                               CoordinateDiff{1, 1},
                                                               CoordinateDiff{1, 1},
                                                               Strides{2, 2},
                                                               op::PadType::EXPLICIT);
        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}
