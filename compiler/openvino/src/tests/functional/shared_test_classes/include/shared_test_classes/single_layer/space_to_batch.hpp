// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using spaceToBatchParamsTuple = typename std::tuple<
        std::vector<int64_t>,               // block_shape
        std::vector<int64_t>,               // pads_begin
        std::vector<int64_t>,               // pads_end
        std::vector<size_t>,               // Input shapes
        InferenceEngine::Precision,        // Network precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        InferenceEngine::Layout,           // Input layout
        InferenceEngine::Layout,           // Output layout
        std::string>;                      // Device name>;

class SpaceToBatchLayerTest : public testing::WithParamInterface<spaceToBatchParamsTuple>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<spaceToBatchParamsTuple> &obj);

protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
