// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        InferenceEngine::Precision,         // Network Precision
        std::string,                        // Target Device
        std::map<std::string, std::string>  //Configuration
> SplitConcatConcatParams;

class SplitConcatConcatTest:
        public testing::WithParamInterface<SplitConcatConcatParams>,
        virtual public LayerTestsUtils::LayerTestsCommon{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SplitConcatConcatParams> &obj);
protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
