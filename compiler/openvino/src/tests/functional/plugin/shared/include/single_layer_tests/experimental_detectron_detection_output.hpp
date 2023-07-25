// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <shared_test_classes/single_layer/experimental_detectron_detection_output.hpp>

namespace ov {
namespace test {
namespace subgraph {

TEST_P(ExperimentalDetectronDetectionOutputLayerTest, ExperimentalDetectronDetectionOutputLayerTests) {
    run();
}

} // namespace subgraph
} // namespace test
} // namespace ov
