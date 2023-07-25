// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <ie_precision.hpp>
#include <ie_common.h>
#include "../base/layer_test_utils.hpp"

namespace SubgraphTestsDefinitions {
typedef std::tuple<
        std::vector<size_t>,              // levels
        std::vector<std::vector<size_t>>, // const inputs shape
        std::vector<float>,               // clamp min max
        std::vector<float>               // input generator data: low, high, resolution
> fqSpecificParams;

typedef std::tuple<
        fqSpecificParams,
        InferenceEngine::Precision,        // Net precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        InferenceEngine::Layout,           // Input layout
        InferenceEngine::Layout,           // Output layout
        InferenceEngine::SizeVector,       // Input shapes
        LayerTestsUtils::TargetDevice,     // Device name
        std::pair<std::string, std::map<std::string, std::string>> // Additional backend configuration and alis name to it
> fqSubgraphTestParamsSet;

class ClampFakeQuantizeSubgraphTest : public testing::WithParamInterface<fqSubgraphTestParamsSet>,
                                 virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<fqSubgraphTestParamsSet>& obj);

protected:
    void SetUp() override;
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    float inputDataMin        = 0.0;
    float inputDataMax        = 10.0;
    float inputDataResolution = 1.0;
    int32_t  seed = 1;
};
} // namespace SubgraphTestsDefinitions
