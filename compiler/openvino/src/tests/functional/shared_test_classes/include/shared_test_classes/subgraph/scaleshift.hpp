// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {

using ScaleShiftParamsTuple = typename std::tuple<
        std::vector<std::vector<size_t>>, //input shapes
        InferenceEngine::Precision,       //Network precision
        std::string,                      //Device name
        std::vector<float>,               //scale
        std::vector<float>>;              //shift

class ScaleShiftLayerTest:
        public testing::WithParamInterface<ScaleShiftParamsTuple>,
        virtual public LayerTestsUtils::LayerTestsCommon{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ScaleShiftParamsTuple> &obj);
protected:
    void SetUp() override;
};
}  // namespace SubgraphTestsDefinitions
