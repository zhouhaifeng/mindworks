// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pass/manager.hpp>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/log_softmax_decomposition.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, LogSoftmaxDecomposition) {
    {
        auto data = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2});
        auto log_softmax = std::make_shared<ngraph::opset5::LogSoftmax>(data, 1);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{log_softmax}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::LogSoftmaxDecomposition>();
    }

    {
        auto input0 = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2});
        auto axis1_const = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto max = std::make_shared<ngraph::opset5::ReduceMax>(input0, axis1_const, true);
        auto sub = std::make_shared<ngraph::opset5::Subtract>(input0, max);
        auto exp = std::make_shared<ngraph::opset5::Exp>(sub);
        auto axis2_const = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto sum = std::make_shared<ngraph::opset5::ReduceSum>(exp, axis2_const, true);
        auto log = std::make_shared<ngraph::opset5::Log>(sum);
        auto sub_end = std::make_shared<ngraph::opset5::Subtract>(sub, log);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{sub_end}, ngraph::ParameterVector{input0});
    }
}
