// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <queue>
#include <string>
#include <transformations/common_optimizations/simplify_shape_of_sub_graph.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

auto gatherv7 =
    [](const std::shared_ptr<Node> input, std::vector<int64_t> indices, bool scalar = false) -> Output<Node> {
    std::shared_ptr<Node> indices_node;
    if (scalar)
        indices_node = opset7::Constant::create(element::i64, {}, indices);
    else
        indices_node = opset7::Constant::create(element::i64, {indices.size()}, indices);
    return std::make_shared<ngraph::opset7::Gather>(input,
                                                    indices_node,
                                                    opset7::Constant::create(element::i64, {}, {0}));
};

auto gatherv8 =
    [](const std::shared_ptr<Node> input, std::vector<int64_t> indices, bool scalar = false) -> Output<Node> {
    std::shared_ptr<Node> indices_node;
    if (scalar)
        indices_node = opset7::Constant::create(element::i64, {}, indices);
    else
        indices_node = opset7::Constant::create(element::i64, {indices.size()}, indices);
    return std::make_shared<ngraph::opset8::Gather>(input,
                                                    indices_node,
                                                    opset7::Constant::create(element::i64, {}, {0}));
};

TEST_F(TransformationTestsF, ShapeSubGraphTestGatherv7) {
    Shape data_shape{1, 2, 3, 4};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_1 = gatherv7(shape_op_1, {1}, true);
        auto unsqueeze_1 =
            std::make_shared<opset7::Unsqueeze>(gather_1, opset7::Constant::create(element::i64, {1}, {0}));

        auto shape_op_2 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_2 = gatherv7(shape_op_2, {2}, true);
        auto unsqueeze_2 =
            std::make_shared<opset7::Unsqueeze>(gather_2, opset7::Constant::create(element::i64, {1}, {0}));

        auto const_1 = opset7::Constant::create(element::i64, Shape{1}, {2});
        auto const_2 = opset7::Constant::create(element::i64, Shape{1}, {2});

        auto concat = std::make_shared<opset7::Concat>(OutputVector{unsqueeze_1, unsqueeze_2, const_1, const_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, false);
        function = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SimplifyShapeOfSubGraph>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_1 = gatherv7(shape_op_1, {1, 2});

        auto const_1 = opset7::Constant::create(element::i64, Shape{1}, {2});
        auto const_2 = opset7::Constant::create(element::i64, Shape{1}, {2});

        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_1, const_1, const_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, false);
        function_ref = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ShapeSubGraphTestGatherv8) {
    Shape data_shape{1, 2, 3, 4};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_1 = gatherv8(shape_op_1, {1}, true);
        auto unsqueeze_1 =
            std::make_shared<opset7::Unsqueeze>(gather_1, opset7::Constant::create(element::i64, {1}, {0}));

        auto shape_op_2 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_2 = gatherv8(shape_op_2, {2}, true);
        auto unsqueeze_2 =
            std::make_shared<opset7::Unsqueeze>(gather_2, opset7::Constant::create(element::i64, {1}, {0}));

        auto const_1 = opset7::Constant::create(element::i64, Shape{1}, {2});
        auto const_2 = opset7::Constant::create(element::i64, Shape{1}, {2});

        auto concat = std::make_shared<opset7::Concat>(OutputVector{unsqueeze_1, unsqueeze_2, const_1, const_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, false);
        function = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SimplifyShapeOfSubGraph>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_1 = gatherv8(shape_op_1, {1, 2});

        auto const_1 = opset7::Constant::create(element::i64, Shape{1}, {2});
        auto const_2 = opset7::Constant::create(element::i64, Shape{1}, {2});

        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_1, const_1, const_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, false);
        function_ref = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ShapeNopSubGraphTestGatherv7) {
    PartialShape data_shape{-1, -1};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_1 = gatherv7(shape_op_1, {0}, true);
        auto unsqueeze_1 =
            std::make_shared<opset7::Unsqueeze>(gather_1, opset7::Constant::create(element::i64, {1}, {0}));

        auto shape_op_2 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_2 = gatherv7(shape_op_2, {1}, true);
        auto unsqueeze_2 =
            std::make_shared<opset7::Unsqueeze>(gather_2, opset7::Constant::create(element::i64, {1}, {0}));

        auto concat = std::make_shared<opset7::Concat>(OutputVector{unsqueeze_1, unsqueeze_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, false);
        function = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SimplifyShapeOfSubGraph>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto shape_op_1 = std::make_shared<opset7::ShapeOf>(data);
        auto reshape = std::make_shared<opset7::Reshape>(data, shape_op_1, false);
        function_ref = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ShapeNopSubGraphTestGatherv8) {
    PartialShape data_shape{-1, -1};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_1 = gatherv8(shape_op_1, {0}, true);
        auto unsqueeze_1 =
            std::make_shared<opset7::Unsqueeze>(gather_1, opset7::Constant::create(element::i64, {1}, {0}));

        auto shape_op_2 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_2 = gatherv8(shape_op_2, {1}, true);
        auto unsqueeze_2 =
            std::make_shared<opset7::Unsqueeze>(gather_2, opset7::Constant::create(element::i64, {1}, {0}));

        auto concat = std::make_shared<opset7::Concat>(OutputVector{unsqueeze_1, unsqueeze_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, false);
        function = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SimplifyShapeOfSubGraph>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto shape_op_1 = std::make_shared<opset7::ShapeOf>(data);
        auto reshape = std::make_shared<opset7::Reshape>(data, shape_op_1, false);
        function_ref = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupedGatherEliminationNegative) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_op = std::make_shared<opset7::ShapeOf>(data);
        auto gather = gatherv8(shape_op, {1}, true);
        auto unsqueeze = std::make_shared<opset7::Unsqueeze>(gather, opset7::Constant::create(element::i64, {1}, {0}));

        auto constant_1 = ngraph::opset7::Constant::create(element::i64, {1}, {0});
        auto constant_2 = ngraph::opset7::Constant::create(element::i64, {1}, {1});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{constant_1, constant_2, unsqueeze}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        function = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::GroupedGatherElimination>();
    }
}
