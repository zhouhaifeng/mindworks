// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_common.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include <ie_core.hpp>

namespace SubgraphTestsDefinitions {
typedef std::tuple<
    std::string,                        // Target device name
    InferenceEngine::Precision,         // Network precision
    size_t,                             // Mutiples of concat size to be used as input size
    size_t,                             // Concat size
    std::map<std::string, std::string>  // Configuration
> memoryEltwiseReshapeConcatParams;

class MemoryEltwiseReshapeConcatTest : virtual public LayerTestsUtils::LayerTestsCommon,
    public testing::WithParamInterface<memoryEltwiseReshapeConcatParams> {
private:
    void initTestModel();
    // you have to replace memory layers since ngraph does not support them
    void initNgraphFriendlyModel();

    // since we switching models we need to generate and save these values in SetUp
    size_t inputSize;
    size_t concatSize;
    ngraph::element::Type ngPrc;
    std::vector<float> memory_init;
    std::vector<float> concat_vals;
protected:
    void SetUp() override;
    void Run() override;
    void LoadNetwork() override;
    void Infer() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<memoryEltwiseReshapeConcatParams> &obj);
};
} // namespace SubgraphTestsDefinitions
