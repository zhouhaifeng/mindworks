// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <legacy/ngraph_ops/fully_connected.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_matmul_to_fc_or_gemm.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/reshape_fully_connected.hpp>
#include <map>
#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <queue>
#include <sstream>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/test_common.hpp"

using namespace testing;

TEST_F(TransformationTestsF, ConvertMatMulTest1) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        function =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});

        manager.register_pass<ngraph::pass::ConvertMatMulToFC>();
        manager.register_pass<ngraph::pass::ConvertMatMulToGemm>();
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 1});

        auto reshape = ov::op::util::reshapeTo(input2, {1, 2, 1});

        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, reshape, false, false);

        function_ref =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulTest2) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        function =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});

        manager.register_pass<ngraph::pass::ConvertMatMulToFC>();
        manager.register_pass<ngraph::pass::ConvertMatMulToGemm>();
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2});

        auto usnqueeze_input2 = std::make_shared<ngraph::opset1::Unsqueeze>(
            input2,
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1}));
        auto reshape = ov::op::util::reshapeTo(usnqueeze_input2, {1, 2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, reshape, false, false);
        auto reshape_output = ov::op::util::reshapeTo(matmul, {3, 1});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_output},
                                                          ngraph::ParameterVector{input1, input2});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulTest3) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        function =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});
        manager.register_pass<ngraph::pass::ConvertMatMulToFC>();
        manager.register_pass<ngraph::pass::ConvertMatMulToGemm>();
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 1});

        auto usnqueeze_input1 = std::make_shared<ngraph::opset1::Unsqueeze>(
            input1,
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
        auto reshape = ov::op::util::reshapeTo(usnqueeze_input1, {1, 1, 2});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(reshape, input2, false, false);
        auto reshape_output = ov::op::util::reshapeTo(matmul, {3, 1});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_output},
                                                          ngraph::ParameterVector{input1, input2});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulTest4) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        function =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});
        manager.register_pass<ngraph::pass::ConvertMatMulToFC>();
        manager.register_pass<ngraph::pass::ConvertMatMulToGemm>();
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        function_ref =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulTest5) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        manager.register_pass<ngraph::pass::ConvertMatMulToFC>();
        manager.register_pass<ngraph::pass::ConvertMatMulToGemm>();
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto input3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2}, {1});
        auto matmul = std::make_shared<ngraph::op::FullyConnected>(input1, input2, input3, ngraph::Shape{3, 2, 2});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulTest6) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        manager.register_pass<ngraph::pass::ConvertMatMulToFC>();
        manager.register_pass<ngraph::pass::ConvertMatMulToGemm>();
        manager.register_pass<ngraph::pass::ReshapeFullyConnected>();
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto input3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2}, {1});
        auto reshape_begin = ov::op::util::reshapeTo(input1, ngraph::Shape{6, 2});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(reshape_begin, input2, input3, ngraph::Shape{6, 2});
        auto reshape_end = ov::op::util::reshapeTo(fc, ngraph::Shape{3, 2, 2});

        function_ref =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_end}, ngraph::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulTest7) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});

        auto pass_config = manager.get_pass_config();
        manager.register_pass<ngraph::pass::ConvertMatMulToFC>();
        manager.register_pass<ngraph::pass::ConvertMatMulToGemm>();
        manager.register_pass<ngraph::pass::ReshapeFullyConnected>();

        auto callback = [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
            if (auto fc_op = std::dynamic_pointer_cast<const ngraph::op::FullyConnected>(node)) {
                if (fc_op->input_value(0).get_shape().size() == 3) {
                    return true;
                }
            }
            return false;
        };

        pass_config->set_callback<ngraph::pass::ReshapeFullyConnected>(callback);
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto input3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2}, {1});
        auto matmul = std::make_shared<ngraph::op::FullyConnected>(input1, input2, input3, ngraph::Shape{3, 2, 2});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }
}

TEST(TransformationTests, ConvertMatMulDynamic) {
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

    auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ngraph::pass::ConvertMatMulToFC>();
    m.register_pass<ngraph::pass::ConvertMatMulToGemm>();
    m.register_pass<ngraph::pass::ReshapeFullyConnected>();
    ASSERT_NO_THROW(m.run_passes(f));
}
