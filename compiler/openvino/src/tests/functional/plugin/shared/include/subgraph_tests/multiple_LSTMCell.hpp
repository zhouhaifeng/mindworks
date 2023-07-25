// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/multiple_LSTMCell.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MultipleLSTMCellTest, CompareWithRefs) {
    Run();
};

} // namespace SubgraphTestsDefinitions
