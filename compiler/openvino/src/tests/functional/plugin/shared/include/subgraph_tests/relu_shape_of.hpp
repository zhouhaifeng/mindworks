// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/relu_shape_of.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ReluShapeOfSubgraphTest, CompareWithRefs) {
    Run();
}

}  // namespace SubgraphTestsDefinitions
