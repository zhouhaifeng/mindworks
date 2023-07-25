// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/convert_pad_to_group_conv.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ConvertPadToConvTests, CompareWithRefs) {
    Run();
}
} // namespace SubgraphTestsDefinitions
