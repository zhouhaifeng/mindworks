// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/permute_concat_permute.hpp"

namespace SubgraphTestsDefinitions {

using PermuteConcatPermuteNeg = PermuteConcatPermute;

TEST_P(PermuteConcatPermute, CompareWithRefs) {
    Run();
}

TEST_P(PermuteConcatPermuteNeg, CompareWithRefs) {
    ExpectLoadNetworkToThrow("type: Concat, and concatenation axis(");
}

}  // namespace SubgraphTestsDefinitions
