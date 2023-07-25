// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/eltwise_reshape_activation.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(EltwiseReshapeActivation, CompareWithRefs) {
    Run();
}

}  // namespace SubgraphTestsDefinitions
