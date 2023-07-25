// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/scaleshift_conv_scaleshift.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ScaleShiftAfterConvTest, CompareWithRefImpl) {
    LoadNetwork();
    GenerateInputs();
    Infer();
    // Create another copy of function for validation since some data will be changed by GNA plugin
    SetUp();
    Validate();
};

TEST_P(ScaleShiftBeforeConvTest, CompareWithRefImpl) {
    LoadNetwork();
    GenerateInputs();
    Infer();
    // Create another copy of function for validation since some data will be changed by GNA plugin
    SetUp();
    Validate();
};

}  // namespace SubgraphTestsDefinitions
