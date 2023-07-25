// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pass/manager.hpp>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/hswish_decomposition.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, HSwishDecompositionTest) {
    {
        auto input =
            std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hswish = std::make_shared<ngraph::opset4::HSwish>(input);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{hswish}, ngraph::ParameterVector{input});

        manager.register_pass<ov::pass::HSwishDecomposition>();
    }

    {
        auto input =
            std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset4::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset4::Relu>(add);
        auto min_constant = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto min = std::make_shared<ngraph::opset4::Minimum>(relu, min_constant);
        auto mul_first = std::make_shared<ngraph::opset4::Multiply>(input, min);
        auto mul_constant = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.1666666716});
        auto mul_second = std::make_shared<ngraph::opset4::Multiply>(mul_first, mul_constant);

        function_ref =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_second}, ngraph::ParameterVector{input});
    }
}
