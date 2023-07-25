// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/mat_mul.hpp"

namespace LayerTestsDefinitions {

TEST_P(MatMulTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
