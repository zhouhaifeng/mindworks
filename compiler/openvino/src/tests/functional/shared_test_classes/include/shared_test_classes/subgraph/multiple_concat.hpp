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
    size_t,                             // Input size
    size_t,                             // Const size
    std::map<std::string, std::string>  // Configuration
> multipleConcatParams;

class MultipleConcatTest : virtual public LayerTestsUtils::LayerTestsCommon,
    public testing::WithParamInterface<multipleConcatParams> {
protected:
    void SetUp() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<multipleConcatParams> &obj);
};
} // namespace SubgraphTestsDefinitions
