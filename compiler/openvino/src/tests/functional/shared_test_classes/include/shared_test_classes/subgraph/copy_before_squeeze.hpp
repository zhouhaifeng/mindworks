// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
    InferenceEngine::Precision,        //Network precision
    std::string,                       //Device name
    std::vector<size_t>,               //Input shape
    std::map<std::string, std::string> //Configuration
> CopyBeforeSqueezeTuple;

class CopyBeforeSqueezeTest
    : public testing::WithParamInterface<CopyBeforeSqueezeTuple>,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CopyBeforeSqueezeTuple>& obj);
protected:
    void SetUp() override;
};
} // namespace SubgraphTestsDefinitions
