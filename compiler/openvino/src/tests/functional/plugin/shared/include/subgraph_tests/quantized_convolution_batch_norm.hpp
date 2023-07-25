// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/quantized_convolution_batch_norm.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(QuantizedConvolutionBatchNorm, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
}

} // namespace SubgraphTestsDefinitions
