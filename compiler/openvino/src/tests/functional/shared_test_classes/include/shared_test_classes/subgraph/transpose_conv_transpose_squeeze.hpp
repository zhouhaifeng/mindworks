// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        std::vector<size_t>,                 // Kernel Shape
        std::vector<size_t>,                 // Strides
        size_t,                              // Input channels
        size_t                               // Output channels
> ConvParams;

typedef std::tuple<
        ConvParams,
        InferenceEngine::Precision,        // Net precision
        InferenceEngine::SizeVector,       // Input shapes
        LayerTestsUtils::TargetDevice,     // Device name
        std::map<std::string, std::string> // Additional backend configuration and alis name to it
> TransposeConvTestParams;

class TransposeConvTest : public testing::WithParamInterface<TransposeConvTestParams>,
                   virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TransposeConvTestParams>& obj);

protected:
    void SetUp() override;
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    float inputDataMin        = 0.0;
    float inputDataMax        = 0.2;
    float inputDataResolution = 1;
    int32_t  seed = 1;
};

}  // namespace SubgraphTestsDefinitions
