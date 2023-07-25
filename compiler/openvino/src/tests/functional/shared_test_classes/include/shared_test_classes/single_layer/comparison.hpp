// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <map>
#include <shared_test_classes/base/layer_test_utils.hpp>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ie_core.hpp"

namespace LayerTestsDefinitions {
namespace ComparisonParams {
using InputShapesTuple = std::pair<std::vector<size_t>, std::vector<size_t>>;
} // ComparisonParams

typedef std::tuple<
    ComparisonParams::InputShapesTuple, // Input shapes tuple
    InferenceEngine::Precision,         // NG Inputs precision
    ngraph::helpers::ComparisonTypes,   // Comparison op type
    ngraph::helpers::InputLayerType,    // Second input type
    InferenceEngine::Precision,         // IE in precision
    InferenceEngine::Precision,         // IE out precision
    std::string,                        // Device name
    std::map<std::string, std::string>  // Additional network configuration
> ComparisonTestParams;

class ComparisonLayerTest : public testing::WithParamInterface<ComparisonTestParams>,
    virtual public LayerTestsUtils::LayerTestsCommon {
    ngraph::helpers::ComparisonTypes comparisonOpType;
protected:
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ComparisonTestParams> &obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &inputInfo) const override;
};
} // namespace LayerTestsDefinitions
