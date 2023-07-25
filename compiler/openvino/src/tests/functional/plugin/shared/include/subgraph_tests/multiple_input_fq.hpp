// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MULTIPLE_INPUT_HPP
#define MULTIPLE_INPUT_HPP

#include "shared_test_classes/subgraph/multiple_input_fq.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MultipleInputTest, CompareWithRefs) {
    Run();
};

} // namespace SubgraphTestsDefinitions

#endif // MULTIPLE_INPUT_HPP
