// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <ov_ops/nms_static_shape_ie.hpp>
#include <queue>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/convert_matrix_nms_to_matrix_nms_ie.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace ngraph;

namespace testing {
class ConvertMatrixNmsToMatrixNmsIEFixture : public ::testing::WithParamInterface<element::Type>,
                                             public TransformationTestsF {
public:
    static std::string getTestCaseName(testing::TestParamInfo<element::Type> obj) {
        std::ostringstream result;
        result << "ConvertMatrixNmsToMatrixNmsIE_" << obj.param.get_type_name();
        return result.str();
    }
    void Execute() {
        element::Type element_type = this->GetParam();
        {
            auto boxes = std::make_shared<opset1::Parameter>(element_type, Shape{1, 1000, 4});
            auto scores = std::make_shared<opset1::Parameter>(element_type, Shape{1, 1, 1000});

            auto nms = std::make_shared<opset8::MatrixNms>(boxes, scores, opset8::MatrixNms::Attributes());

            function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

            manager.register_pass<ov::pass::ConvertMatrixNmsToMatrixNmsIE>();
            manager.register_pass<ngraph::pass::ConstantFolding>();
        }

        {
            auto boxes = std::make_shared<opset1::Parameter>(element_type, Shape{1, 1000, 4});
            auto scores = std::make_shared<opset1::Parameter>(element_type, Shape{1, 1, 1000});
            auto nms = std::make_shared<ov::op::internal::NmsStaticShapeIE<ngraph::opset8::MatrixNms>>(
                boxes,
                scores,
                opset8::MatrixNms::Attributes());

            function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
        }
        ASSERT_EQ(function->get_output_element_type(0), function_ref->get_output_element_type(0))
            << "Output element type mismatch " << function->get_output_element_type(0).get_type_name() << " vs "
            << function_ref->get_output_element_type(0).get_type_name();
    }
};

TEST_P(ConvertMatrixNmsToMatrixNmsIEFixture, CompareFunctions) {
    Execute();
}

INSTANTIATE_TEST_SUITE_P(ConvertMatrixNmsToMatrixNmsIE,
                         ConvertMatrixNmsToMatrixNmsIEFixture,
                         ::testing::ValuesIn(std::vector<element::Type>{element::f32, element::f16}),
                         ConvertMatrixNmsToMatrixNmsIEFixture::getTestCaseName);
}  // namespace testing
