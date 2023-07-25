// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/subtract_multiply_to_multiply_add_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "lpt_ngraph_functions/subtract_multiply_to_multiply_add_function.hpp"

namespace LayerTestsDefinitions {

std::string SubtractMultiplyToMultiplyAddTransformation::getTestCaseName(const testing::TestParamInfo<SubtractMultiplyToMultiplyAddTransformationParams>& obj) {
    std::string targetDevice;
    SubtractMultiplyToMultiplyAddTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result <<
        targetDevice << "_" <<
        testValues.inputShape << "_" <<
        testValues.precision << "_" <<
        testValues.fqOnData;
    return result.str();
}

void SubtractMultiplyToMultiplyAddTransformation::SetUp() {
    SubtractMultiplyToMultiplyAddTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::SubtractMultiplyToMultiplyAddFunction::getOriginal(
        testValues.inputShape,
        testValues.precision,
        testValues.fqOnData);
}

TEST_P(SubtractMultiplyToMultiplyAddTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
