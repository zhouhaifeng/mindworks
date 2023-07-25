// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

typedef std::tuple<
        std::vector<InputShape>,               // input shape
        int64_t ,                              // Max rois
        ElementType,                           // Network precision
        std::string                            // Device name
> ExperimentalDetectronTopKROIsTestParams;

class ExperimentalDetectronTopKROIsLayerTest : public testing::WithParamInterface<ExperimentalDetectronTopKROIsTestParams>,
                                               virtual public SubgraphBaseTest {
protected:
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalDetectronTopKROIsTestParams>& obj);
};
} // namespace subgraph
} // namespace test
} // namespace ov
