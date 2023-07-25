// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/opsets/opset3.hpp>
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeComparison(const ov::Output<Node> &in0,
                                         const ov::Output<Node> &in1,
                                         ngraph::helpers::ComparisonTypes comparisonType) {
    switch (comparisonType) {
        case ngraph::helpers::ComparisonTypes::EQUAL:
            return std::make_shared<ngraph::opset3::Equal>(in0, in1);
        case ngraph::helpers::ComparisonTypes::NOT_EQUAL:
            return std::make_shared<ngraph::opset3::NotEqual>(in0, in1);
        case ngraph::helpers::ComparisonTypes::GREATER:
            return std::make_shared<ngraph::opset3::Greater>(in0, in1);
        case ngraph::helpers::ComparisonTypes::GREATER_EQUAL:
            return std::make_shared<ngraph::opset3::GreaterEqual>(in0, in1);
        case ngraph::helpers::ComparisonTypes::IS_FINITE:
            return std::make_shared<ov::op::v10::IsFinite>(in0);
        case ngraph::helpers::ComparisonTypes::IS_INF:
            return std::make_shared<ov::op::v10::IsInf>(in0);
        case ngraph::helpers::ComparisonTypes::IS_NAN:
            return std::make_shared<ov::op::v10::IsNaN>(in0);
        case ngraph::helpers::ComparisonTypes::LESS:
            return std::make_shared<ngraph::opset3::Less>(in0, in1);
        case ngraph::helpers::ComparisonTypes::LESS_EQUAL:
            return std::make_shared<ngraph::opset3::LessEqual>(in0, in1);
        default: {
            throw std::runtime_error("Incorrect type of Comparison operation");
        }
    }
}

}  // namespace builder
}  // namespace ngraph
