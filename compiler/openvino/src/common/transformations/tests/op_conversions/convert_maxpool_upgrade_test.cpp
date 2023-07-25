// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/convert_maxpool_upgrade.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, ConvertMaxPool1ToMaxPool8) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        ngraph::Strides strides{1};
        ngraph::Shape pads_begin{0}, pads_end{0}, kernel{1};
        auto maxpool_1 = std::make_shared<ngraph::opset1::MaxPool>(data, strides, pads_begin, pads_end, kernel);

        function = std::make_shared<ngraph::Function>(ngraph::OutputVector{maxpool_1->output(0)},
                                                      ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::ConvertMaxPool1ToMaxPool8>();
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        ngraph::Strides strides{1}, dilations{1};
        ngraph::Shape pads_begin{0}, pads_end{0}, kernel{1};
        auto maxpool_8 =
            std::make_shared<ngraph::opset8::MaxPool>(data, strides, dilations, pads_begin, pads_end, kernel);

        function_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{maxpool_8->output(0)},
                                                          ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, negative_ConvertMaxPool1ToMaxPool8_dynamic_rank) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        ngraph::Strides strides{1};
        ngraph::Shape pads_begin{0}, pads_end{0}, kernel{1};
        auto maxpool_1 = std::make_shared<ngraph::opset1::MaxPool>(data, strides, pads_begin, pads_end, kernel);

        function = std::make_shared<ngraph::Function>(ngraph::OutputVector{maxpool_1->output(0)},
                                                      ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::ConvertMaxPool1ToMaxPool8>();
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        ngraph::Strides strides{1};
        ngraph::Shape pads_begin{0}, pads_end{0}, kernel{1};
        auto maxpool_1 = std::make_shared<ngraph::opset1::MaxPool>(data, strides, pads_begin, pads_end, kernel);

        function_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{maxpool_1->output(0)},
                                                          ngraph::ParameterVector{data});
    }
}
