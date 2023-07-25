// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/multiple_connect_split_concat.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MultipleConnectSplitConcatTest, CompareWithRefs) {
    Run();
};

}  // namespace SubgraphTestsDefinitions
