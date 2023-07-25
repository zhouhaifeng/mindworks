// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/concat_conv.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ConcatConvTest, CompareWithRefImpl) {
    Run();
};

}  // namespace SubgraphTestsDefinitions
